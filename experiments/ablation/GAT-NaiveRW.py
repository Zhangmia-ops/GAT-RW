# GAT-NaiveRW.py
# 配置：GAT (谱特征) + RW (简单线性修正，无平滑策略)

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
# This script runs an ablation experiment combining a GAT model with simple random-walk correction.
# It builds local and spectral node features, trains the GAT, and adjusts predictions with rank-based random-walk scores.
# It saves evaluation metrics, timing data, and raw predictions to JSON.

class DynamicWeightedGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(dict)
    def add_edge(self, u: int, v: int, weight: float) -> None:
        self.nodes.add(u); self.nodes.add(v)
        self.edges[u][v] = weight; self.edges[v][u] = weight

def random_walk_mc(s: int, L: int, n_walks: int, graph, beta: float = 0.85):
    visit_counts = defaultdict(int)
    if s not in graph.edges or not graph.edges[s]: return {}
    for _ in range(n_walks):
        curr = s
        for _ in range(L):
            if random.random() > beta: break
            curr_neighbors = list(graph.edges[curr].keys())
            if not curr_neighbors: break
            curr_weights = list(graph.edges[curr].values())
            curr = random.choices(curr_neighbors, weights=curr_weights, k=1)[0]
            visit_counts[curr] += 1
    total = sum(visit_counts.values())
    if total == 0: return {}
    return {node: count / total for node, count in visit_counts.items() if node != s}

def revise_predictions_naive(gat_predictions, rw_probs, alpha=0.15):

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
        
        final_delta = diff * alpha 
        
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
        if pe.shape[1] < k: 
            pe = torch.cat([pe, torch.zeros(num_nodes, k - pe.shape[1])], dim=1)
    except: pe = torch.zeros(num_nodes, k)
    return pe

def build_features(adj_matrix, num_nodes):
    local = compute_local_features(adj_matrix, num_nodes)
    pe = compute_laplacian_pe(adj_matrix, num_nodes, k=8)
    features = torch.cat([local, pe], dim=1)
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
    best_loss = float('inf'); patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        node_out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        src, dst = data.edge_index
        pred_edge = (node_out[src] + node_out[dst]) / 2
        loss = criterion(pred_edge, data.edge_attr.squeeze())
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item(); patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= patience: break
    return model

def run_pipeline(train_file, test_file, output_json, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 [Ablation Model D] GAT + Simple RW (No Decay)")

    adj_matrix = defaultdict(dict)
    rw_graph = DynamicWeightedGraph()
    max_node_idx = 0
    with open(train_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v, w = line.strip().split(',')
            u, v, w = int(u), int(v), float(w)
            max_node_idx = max(max_node_idx, u, v)
            adj_matrix[u][v] = w; adj_matrix[v][u] = w
            rw_graph.add_edge(u, v, w)
    num_nodes = max_node_idx + 1
    
    features = build_features(adj_matrix, num_nodes).to(device)
    edge_list, edge_weights = [], []
    for u, nbrs in adj_matrix.items():
        for v, w in nbrs.items():
            edge_list.append([u, v]); edge_weights.append(w)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1).to(device)
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

    model = StaticGAT(features.shape[1], params['hidden_dim'], params['heads']).to(device)
    model = train_gat_model(model, data, params['epochs'], params['lr'], device)

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
            if u < num_nodes and v < num_nodes:
                w_pred = float((node_embeddings[u] + node_embeddings[v]) / 2.0)
            else: w_pred = 0.0
            gat_predictions.append((u, v, w_pred))
            true_values.append(w)
            test_pairs.append((u, v))

    # RW Sampling
    unique_test_nodes = set([u for u,v in test_pairs] + [v for u,v in test_pairs])
    rw_probs = {}
    node_pi_cache = {}
    for node in unique_test_nodes:
        node_pi_cache[node] = random_walk_mc(node, params['L'], params['n_walks'], rw_graph)
    for u, v in test_pairs:
        p_uv = node_pi_cache.get(u, {}).get(v, 0.0)
        p_vu = node_pi_cache.get(v, {}).get(u, 0.0)
        rw_probs[(u, v)] = (p_uv + p_vu) / 2.0

    final_results_list = revise_predictions_naive(gat_predictions, rw_probs, alpha=params['alpha'])

    y_true = np.array(true_values)
    y_pred = np.array([item[2] for item in final_results_list])
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    
    output_data = {
        "meta_data": {
            "algorithm": "GAT-NaiveRW_SimpleRW",
            "dataset": params['dataset_name'],
            "mask_cnt": params['mask_cnt']
        },
        "metrics": {"mse": round(float(mse), 6), "pcc": round(float(pcc), 6), "mae": round(float(mae), 6)},
        "raw_results": [] 
    }
    
    with open(output_json, 'w') as f: json.dump(output_data, f, indent=2)
    print(f"✅ Model D Finished. PCC: {pcc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default="result_ablation_D.json")
    parser.add_argument("--dataset_name", default="Unknown")
    parser.add_argument("--mask_cnt", default="0")

    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--n_walks", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.2)
    
    args = parser.parse_args()
    run_pipeline(args.train_file, args.test_file, args.output_file, vars(args))