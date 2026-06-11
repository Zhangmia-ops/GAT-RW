# Ensure your environment has torch, torch_geometric, numpy, scipy, networkx, tqdm installed
import json
import time
import random
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata, pearsonr
from collections import defaultdict
from tqdm import tqdm
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
# New addition
def load_reverse_node_mapping(mapping_file):
    reverse_map = {}

    with open(mapping_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            orig_id, std_id = line.strip().split(",")
            reverse_map[int(std_id)] = int(orig_id)

    return reverse_map

def merge_outputs(direct, exact, model):
    return direct + exact + model

def save_plain_predictions(results, output_file, reverse_map=None):
    with open(output_file, "w") as f:
        for u, v, w in results:
            if reverse_map is not None:
                u = reverse_map.get(u, u)
                v = reverse_map.get(v, v)

            f.write(f"{u},{v},{w}\n")

    print(f"✅ Inference results saved to {output_file}")

def load_component_index(path):
    with open(path, "r") as f:
        data = json.load(f)

    node_to_component = {int(k): int(v) for k, v in data["node_to_component"].items()}
    component_to_nodes = {
        int(k): set(v) for k, v in data["component_to_nodes"].items()
    }

    return node_to_component, component_to_nodes

def load_queries(query_file):
    queries = []

    with open(query_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split(",")
            u, v = int(parts[0]), int(parts[1])
            queries.append((u, v))

    return queries

def build_edge_lookup(adj_matrix):
    edge_dict = {}
    for u in adj_matrix:
        for v, w in adj_matrix[u].items():
            edge_dict[(u, v)] = w
            edge_dict[(v, u)] = w
    return edge_dict

def route_queries(queries, node_to_component, edge_lookup):
    """
    return:
    direct_outputs = [(u,v,0.0)]
    exact_edges = [(u,v,w)]
    need_model = [(u,v)]
    """

    direct_outputs = []
    exact_outputs = []
    need_model = []

    for u, v in queries:

        # Case 1: not same component → 0.0
        if node_to_component[u] != node_to_component[v]:
            direct_outputs.append((u, v, 0.0))
            continue

        # Case 2: edge already exists → return original weight
        if (u, v) in edge_lookup:
            exact_outputs.append((u, v, edge_lookup[(u, v)]))
            continue

        # Case 3: need model inference
        need_model.append((u, v))

    return direct_outputs, exact_outputs, need_model

# ==========================================
# PART 1: Basic graph structure (from teammate's code)
# ==========================================

class DynamicWeightedGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(dict)  # {u: {v: weight}}
        # Removed complex maintenance of reverse edges and coin flips, keep minimal structure needed for random walks to speed up
    
    def add_edge(self, u: int, v: int, weight: float) -> None:
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges[u][v] = weight
        # Undirected graph, add bidirectional
        self.edges[v][u] = weight

# ==========================================
# PART 2: Random walk logic (from teammate's code)
# ==========================================

def random_walk_mc(s: int, L: int, n_walks: int, graph: DynamicWeightedGraph, beta: float = 0.85):
    """Monte Carlo random walk"""
    visit_counts = defaultdict(int)
    if s not in graph.edges or not graph.edges[s]:
        return {}

    # Precompute neighbors and weights to speed up sampling
    neighbors = list(graph.edges[s].keys())
    weights = list(graph.edges[s].values())
    
    for _ in range(n_walks):
        curr = s
        for _ in range(L):
            if random.random() > beta: break # Teleport
            
            curr_neighbors = list(graph.edges[curr].keys())
            if not curr_neighbors: break
            
            curr_weights = list(graph.edges[curr].values())
            # Weighted sampling for next hop
            curr = random.choices(curr_neighbors, weights=curr_weights, k=1)[0]
            visit_counts[curr] += 1

    total = sum(visit_counts.values())
    if total == 0: return {}
    
    # Normalize
    return {node: count / total for node, count in visit_counts.items() if node != s}

# ==========================================
# PART 3: Revision strategy (Smooth Gradient Suppression - Master Edition)
# ==========================================

def revise_predictions(gat_predictions, rw_probs, alpha=0.15):
    """
    gat_predictions: list of (u, v, w_gat)
    rw_probs: dict {(u, v): prob}
    Returns: list of (u, v, w_revised)
    """
    weights = [item[2] for item in gat_predictions]
    # Get corresponding RW probability, 0 if not present
    probs = [rw_probs.get((item[0], item[1]), 0.0) for item in gat_predictions]
    
    if not weights: return []

    weights_np = np.array(weights)
    probs_np = np.array(probs)

    # Compute ranks
    rank_w = rankdata(weights_np, method='average') / len(weights_np)
    rank_p = rankdata(probs_np, method='average') / len(probs_np)

    new_weights = []
    
    for i, w_orig in enumerate(weights_np):
        r_gat = rank_w[i]
        r_rw = rank_p[i]
        diff = r_rw - r_gat
        
        # Smooth decay factor: (1-w)^2, protects high scores, suppresses low scores
        decay_factor = (1.0 - w_orig) ** 2
        
        final_delta = 0.0
        if diff < 0: # RW thinks weaker -> lower score
            final_delta = diff * alpha * decay_factor * 2.0
        else: # RW thinks stronger -> slightly raise score
            final_delta = diff * alpha * decay_factor * 0.1
            
        new_w = min(1.0, max(0.0, w_orig + final_delta))
        new_weights.append(new_w)
        
    # Reassemble results
    revised_results = []
    for i, (u, v, _) in enumerate(gat_predictions):
        revised_results.append((u, v, new_weights[i]))
        
    return revised_results

# ==========================================
# PART 4: Feature engineering (your new version - includes Laplacian spectral features)
# ==========================================

def compute_local_features(adj_matrix, num_nodes):
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    weighted_degree = torch.zeros(num_nodes, dtype=torch.float32)
    clustering = torch.zeros(num_nodes, dtype=torch.float32)

    for u in range(num_nodes):
        if u in adj_matrix:
            neighbors = adj_matrix[u]
            k = len(neighbors)
            degree[u] = k
            weighted_degree[u] = sum(neighbors.values())
            if k >= 2:
                triangle_count = 0
                neighbor_list = list(neighbors.keys())
                for i in range(k):
                    for j in range(i+1, k):
                        ni, nj = neighbor_list[i], neighbor_list[j]
                        if nj in adj_matrix[ni]: triangle_count += 1
                clustering[u] = 2 * triangle_count / (k * (k - 1))
    return torch.stack([degree, weighted_degree, clustering], dim=1)

def compute_laplacian_pe(adj_matrix, num_nodes, k=8):
    # Build sparse matrix
    row, col, data = [], [], []
    for u, neighbors in adj_matrix.items():
        for v, w in neighbors.items():
            row.append(u); col.append(v); data.append(w)
    
    if not row: return torch.zeros(num_nodes, k)

    adj_csr = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj_csr @ d_mat_inv_sqrt

    try:
        # Compute smallest k+1 eigenvalues
        vals, vecs = eigsh(laplacian, k=k+1, which='SM', tol=1e-2)
        # Sort and discard the first (zero/constant vector)
        idx = vals.argsort()
        pe = torch.from_numpy(vecs[:, idx][:, 1:k+1]).float()
        if pe.shape[1] < k: # Pad if not enough eigs
            padding = torch.zeros(num_nodes, k - pe.shape[1])
            pe = torch.cat([pe, padding], dim=1)
    except:
        pe = torch.zeros(num_nodes, k)
    return pe

def build_features(adj_matrix, num_nodes):
    local = compute_local_features(adj_matrix, num_nodes)
    pe = compute_laplacian_pe(adj_matrix, num_nodes, k=8)
    features = torch.cat([local, pe], dim=1)
    
    # Z-score normalization
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std

# ==========================================
# PART 5: GAT model (your code)
# ==========================================

class StaticGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, out_dim=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return torch.sigmoid(x)

def train_gat_model(model, data, max_epochs, lr, device, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Progress bar description
    pbar = tqdm(range(max_epochs), desc="Training GAT", leave=False)
    
    for epoch in pbar:
        optimizer.zero_grad()
        node_out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        src, dst = data.edge_index
        pred_edge = (node_out[src] + node_out[dst]) / 2
        loss = criterion(pred_edge, data.edge_attr.squeeze())
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        # --- Early Stopping mechanism ---
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0 # Reset counter when new minimum found
            best_epoch = epoch
        else:
            patience_counter += 1 # Increment counter if no improvement
            
        pbar.set_postfix({'loss': f"{current_loss:.6f}", 'best': f"{best_loss:.6f}", 'patience': patience_counter})
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.6f} at epoch {best_epoch}")
            break
            
    return model

# ==========================================
# PART 6: Main pipeline integration
# ==========================================

def run_inference_pipeline(train_file, query_file, output_file, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # 1. Load training graph
    adj_matrix = defaultdict(dict)
    rw_graph = DynamicWeightedGraph()

    max_node_idx = 0
    with open(train_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            u, v, w = line.strip().split(',')
            u, v, w = int(u), int(v), float(w)

            max_node_idx = max(max_node_idx, u, v)

            adj_matrix[u][v] = w
            adj_matrix[v][u] = w
            rw_graph.add_edge(u, v, w)

    num_nodes = max_node_idx + 1

    # This is the first location you asked about:
    # After loading the training graph, build the existing edge lookup table
    edge_lookup = build_edge_lookup(adj_matrix)

    # Read query
    queries = load_queries(query_file)

    # Read component_index
    node_to_component, _ = load_component_index(params["component_index"])

    # This is the second location you asked about:
    # Before model prediction, classify queries into three categories
    direct_outputs, exact_outputs, model_queries = route_queries(
        queries,
        node_to_component,
        edge_lookup
    )

    # If no queries need model prediction, save results directly
    if len(model_queries) == 0:
        final_results = direct_outputs + exact_outputs
        save_plain_predictions(final_results, output_file)
        return

    # 2. Feature engineering
    print("--- 2. Building Features (Local + Spectral) ---")
    features = build_features(adj_matrix, num_nodes).to(device)

    edge_list, edge_weights = [], []
    for u, nbrs in adj_matrix.items():
        for v, w in nbrs.items():
            edge_list.append([u, v])
            edge_weights.append(w)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1).to(device)
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

    # 3. Train GAT
    print(f"--- 3. Training GAT (Epochs={params['epochs']}) ---")
    model = StaticGAT(
        in_dim=features.shape[1],
        hidden_dim=params['hidden_dim'],
        heads=params.get('heads', 4)
    ).to(device)

    model = train_gat_model(
        model,
        data,
        params['epochs'],
        params['lr'],
        device,
        patience=20
    )

    # 4. GAT inference only on model_queries
    print("--- 4. GAT Inference on Model Queries ---")
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index, data.edge_attr).squeeze().cpu()

    gat_predictions = []
    test_pairs = []

    # This is the third location you asked about:
    # Originally iterated over test_file, now only over model_queries
    for u, v in model_queries:
        if u < num_nodes and v < num_nodes:
            w_pred = (node_embeddings[u] + node_embeddings[v]) / 2.0
            w_pred = float(w_pred.item())
        else:
            w_pred = 0.0

        gat_predictions.append((u, v, w_pred))
        test_pairs.append((u, v))

    # 5. Random walk only for model_queries
    print(f"--- 5. Random Walk Sampling (L={params['L']}, N={params['n_walks']}) ---")
    rw_probs = {}

    unique_test_nodes = set()
    for u, v in test_pairs:
        unique_test_nodes.add(u)
        unique_test_nodes.add(v)

    node_pi_cache = {}

    for node in tqdm(unique_test_nodes, desc="RW Sampling"):
        pi = random_walk_mc(node, params['L'], params['n_walks'], rw_graph)
        node_pi_cache[node] = pi

    for u, v in test_pairs:
        p_uv = node_pi_cache.get(u, {}).get(v, 0.0)
        p_vu = node_pi_cache.get(v, {}).get(u, 0.0)
        rw_probs[(u, v)] = (p_uv + p_vu) / 2.0

    # 6. Revision fusion
    print(f"--- 6. Revision (Alpha={params['alpha']}) ---")
    model_results = revise_predictions(gat_predictions, rw_probs, alpha=params['alpha'])

    # 7. Merge three categories of outputs and output in the original query order
    pred_dict = {}

    for u, v, w in direct_outputs:
        pred_dict[(u, v)] = w

    for u, v, w in exact_outputs:
        pred_dict[(u, v)] = w

    for u, v, w in model_results:
        pred_dict[(u, v)] = w

    final_results = []
    for u, v in queries:
        if (u, v) in pred_dict:
            final_results.append((u, v, pred_dict[(u, v)]))

    reverse_map = None
    if params.get("mapping_file") is not None:
        reverse_map = load_reverse_node_mapping(params["mapping_file"])

    save_plain_predictions(final_results, output_file, reverse_map)
# ==========================================
# Entry point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True, help="Path to subgraph edge file.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to query pair file.")
    parser.add_argument("--component_index", type=str, required=True, help="Path to component_index.json.")
    parser.add_argument("--output_file", type=str, default="predicted_edges.txt")
    parser.add_argument("--mapping_file", type=str, default=None, help="Path to M_node_dictionary.txt.")
    parser.add_argument("--dataset_name", type=str, default="Unknown")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--n_walks", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.2)

    args = parser.parse_args()
    params = vars(args)

    run_inference_pipeline(
        args.train_file,
        args.query_file,
        args.output_file,
        params
    )