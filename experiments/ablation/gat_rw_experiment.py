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
# This script trains a GAT-based edge-weight predictor and refines predictions with random-walk evidence.
# It builds graph features, performs Monte Carlo random walks, applies a smoothed rank correction, and evaluates test edges.
# It writes metrics, timing information, and raw prediction results to a JSON file.

class DynamicWeightedGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(dict)  # {u: {v: weight}}
    def add_edge(self, u: int, v: int, weight: float) -> None:
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges[u][v] = weight

        self.edges[v][u] = weight

def random_walk_mc(s: int, L: int, n_walks: int, graph: DynamicWeightedGraph, beta: float = 0.85):
    """蒙特卡洛随机游走"""
    visit_counts = defaultdict(int)
    if s not in graph.edges or not graph.edges[s]:
        return {}

    neighbors = list(graph.edges[s].keys())
    weights = list(graph.edges[s].values())
    
    for _ in range(n_walks):
        curr = s
        for _ in range(L):
            if random.random() > beta: break # Teleport
            
            curr_neighbors = list(graph.edges[curr].keys())
            if not curr_neighbors: break
            
            curr_weights = list(graph.edges[curr].values())
    
            curr = random.choices(curr_neighbors, weights=curr_weights, k=1)[0]
            visit_counts[curr] += 1

    total = sum(visit_counts.values())
    if total == 0: return {}

    return {node: count / total for node, count in visit_counts.items() if node != s}


def revise_predictions(gat_predictions, rw_probs, alpha=0.15):
    """
    gat_predictions: list of (u, v, w_gat)
    rw_probs: dict {(u, v): prob}
    return:list of (u, v, w_revised)
    """
    weights = [item[2] for item in gat_predictions]

    probs = [rw_probs.get((item[0], item[1]), 0.0) for item in gat_predictions]
    
    if not weights: return []

    weights_np = np.array(weights)
    probs_np = np.array(probs)

    rank_w = rankdata(weights_np, method='average') / len(weights_np)
    rank_p = rankdata(probs_np, method='average') / len(probs_np)

    new_weights = []
    
    for i, w_orig in enumerate(weights_np):
        r_gat = rank_w[i]
        r_rw = rank_p[i]
        diff = r_rw - r_gat
 
        decay_factor = (1.0 - w_orig) ** 2
        
        final_delta = 0.0
        if diff < 0: # 
            final_delta = diff * alpha * decay_factor * 2.0
        else: 
            final_delta = diff * alpha * decay_factor * 0.1
            
        new_w = min(1.0, max(0.0, w_orig + final_delta))
        new_weights.append(new_w)

    revised_results = []
    for i, (u, v, _) in enumerate(gat_predictions):
        revised_results.append((u, v, new_weights[i]))
        
    return revised_results


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
   
        vals, vecs = eigsh(laplacian, k=k+1, which='SM', tol=1e-2)
  
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
    
    # Z-score 
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std

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

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0 # 
            best_epoch = epoch
        else:
            patience_counter += 1 
            
        pbar.set_postfix({'loss': f"{current_loss:.6f}", 'best': f"{best_loss:.6f}", 'patience': patience_counter})
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.6f} at epoch {best_epoch}")
            break
            
    return model

def run_pipeline(train_file, test_file, output_json, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    print("--- 1. Loading Graph Data ---")
    adj_matrix = defaultdict(dict)
    rw_graph = DynamicWeightedGraph() 
    
    max_node_idx = 0
    with open(train_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v, w = line.strip().split(',')
            u, v, w = int(u), int(v), float(w)
            max_node_idx = max(max_node_idx, u, v)
            
            adj_matrix[u][v] = w
            adj_matrix[v][u] = w
            rw_graph.add_edge(u, v, w)
    
    num_nodes = max_node_idx + 1

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

    print(f"--- 3. Training GAT (Epochs={params['epochs']}) ---")
    t0 = time.time()
    model = StaticGAT(in_dim=features.shape[1], 
                      hidden_dim=params['hidden_dim'], 
                      heads=params.get('heads', 4)).to(device)
    model = train_gat_model(model, data, params['epochs'], params['lr'], device,patience=20)
    train_time = time.time() - t0

    print("--- 4. GAT Inference on Test Set ---")
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index, data.edge_attr).squeeze().cpu()
    
    gat_predictions = [] 
    true_values = []    
    test_pairs = []
    
    with open(test_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v, w = line.strip().split(',')
            u, v, w = int(u), int(v), float(w)
            
            # GAT Predict
            if u < num_nodes and v < num_nodes:
                w_pred = (node_embeddings[u] + node_embeddings[v]) / 2.0
                w_pred = float(w_pred.item())
            else:
                w_pred = 0.0 # 处理孤立/新节点
            
            gat_predictions.append((u, v, w_pred))
            true_values.append(w)
            test_pairs.append((u, v))
            
    inference_time = time.time() - t1

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

    print(f"--- 6. Revision (Alpha={params['alpha']}) ---")
    # gat_predictions: (u, v, w_gat) -> revised: (u, v, w_final)
    final_results_list = revise_predictions(gat_predictions, rw_probs, alpha=params['alpha'])

    print("--- 7. Calculating Metrics & Exporting ---")
    
    y_true = np.array(true_values)
    y_pred = np.array([item[2] for item in final_results_list])
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    
    raw_results = []
    for i, (u, v, w_final) in enumerate(final_results_list):
        raw_results.append({
            "u": u,
            "v": v,
            "true_w": round(float(true_values[i]), 6), 
            "pred_w": round(float(w_final), 6)         
        })
        
    output_data = {
        "meta_data": {
            "algorithm": "GAT_Spectral_RW_Correction",
            "dataset": params['dataset_name'],
            "experiment_type": "ablation_final",
            "mask_ratio": f"{params['mask_cnt']}_edges",
            "hyper_parameters": {
                "hidden_dim": params['hidden_dim'],
                "learning_rate": params['lr'],
                "epochs": params['epochs'],
                "rw_L": params['L'],
                "rw_n_walks": params['n_walks'],
                "alpha": params['alpha']
            }
        },
        "metrics": {
            "mse": round(float(mse), 6),       
            "rmse": round(float(rmse), 6),     
            "mae": round(float(mae), 6),       
            "pcc": round(float(pcc), 6),       
            "train_time_seconds": round(train_time, 6),      
            "inference_time_seconds": round(inference_time, 6) 
        },
        "raw_results": raw_results
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"✅ Results saved to {output_json}")
    print(f"📊 Final PCC: {pcc:.4f} | MSE: {mse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training graph .txt")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test edges .txt")
    parser.add_argument("--output_file", type=str, default="result_GAT_RW.json")
 
    parser.add_argument("--dataset_name", type=str, default="Unknown")
    parser.add_argument("--mask_cnt", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000) 
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--n_walks", type=int, default=3000) 
    parser.add_argument("--alpha", type=float, default=0.2)
    
    args = parser.parse_args()
         
    params = vars(args)
    run_pipeline(args.train_file, args.test_file, args.output_file, params)