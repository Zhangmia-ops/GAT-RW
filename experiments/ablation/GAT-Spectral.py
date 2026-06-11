# GAT-Spectral.py
# Local +  Spectral | (No RW)
# This script runs an ablation experiment using local and spectral graph features.
# It trains a static GAT model without random-walk correction for edge-weight prediction.
# It evaluates test edges and exports prediction metrics and raw results to JSON.

import json
import time
import random
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

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

def build_features_full(adj_matrix, num_nodes):
    local = compute_local_features(adj_matrix, num_nodes)
    pe = compute_laplacian_pe(adj_matrix, num_nodes, k=8)
    features = torch.cat([local, pe], dim=1) # 拼接
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std

# 2. GAT 模型 (同上)
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
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        node_out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        src, dst = data.edge_index
        pred_edge = (node_out[src] + node_out[dst]) / 2
        loss = criterion(pred_edge, data.edge_attr.squeeze())
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= patience: break
    return model

# 3. 主流程
def run_pipeline(train_file, test_file, output_json, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 [Ablation Model B] With Spectral Features | No RW")

    adj_matrix = defaultdict(dict)
    max_node_idx = 0
    with open(train_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v, w = line.strip().split(',')
            u, v, w = int(u), int(v), float(w)
            max_node_idx = max(max_node_idx, u, v)
            adj_matrix[u][v] = w
            adj_matrix[v][u] = w
    num_nodes = max_node_idx + 1
    
    features = build_features_full(adj_matrix, num_nodes).to(device)
    
    edge_list, edge_weights = [], []
    for u, nbrs in adj_matrix.items():
        for v, w in nbrs.items():
            edge_list.append([u, v])
            edge_weights.append(w)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1).to(device)
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

    model = StaticGAT(features.shape[1], params['hidden_dim'], params['heads']).to(device)
    model = train_gat_model(model, data, params['epochs'], params['lr'], device)

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr).squeeze().cpu()
    
    predictions = []
    true_values = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v, w = map(float, line.strip().split(','))
            u, v = int(u), int(v)
            if u < num_nodes and v < num_nodes:
                w_pred = (embeddings[u] + embeddings[v]) / 2.0
                w_pred = float(w_pred.item())
            else: w_pred = 0.0
            predictions.append((u, v, w_pred))
            true_values.append(w)

    y_true = np.array(true_values)
    y_pred = np.array([p[2] for p in predictions])
    mse = np.mean((y_true - y_pred) ** 2)
    pcc, _ = pearsonr(y_true, y_pred)
    
    output_data = {
        "meta_data": {"algorithm": "GAT-Spectral_SpectralNoRW", "dataset": params['dataset_name']},
        "metrics": {"mse": round(float(mse), 6), "pcc": round(float(pcc), 6)},
        "raw_results": [{"u": p[0], "v": p[1], "true_w": round(t,6), "pred_w": round(p[2],6)} for t, p in zip(true_values, predictions)]
    }
    with open(output_json, 'w') as f: json.dump(output_data, f, indent=2)
    print(f"✅ Model B Finished. PCC: {pcc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default="result_ablation_B.json")
    parser.add_argument("--dataset_name", default="Unknown")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    run_pipeline(args.train_file, args.test_file, args.output_file, vars(args))