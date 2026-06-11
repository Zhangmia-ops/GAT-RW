# GAT-Local.py
# Local Features Only No RW
# This script runs an ablation experiment that uses only local graph features.
# It trains a static GAT model to predict masked edge weights from training graph data.
# It evaluates predictions on a held-out test set and writes metrics plus raw results to JSON.

import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from scipy.stats import pearsonr
from collections import defaultdict

def compute_local_features(adj_matrix, num_nodes):
    # Computes local structural features for every node in the graph.
    # The returned tensor contains degree, weighted degree, and clustering coefficient columns.
    
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

def build_features_local_only(adj_matrix, num_nodes):
    # Builds normalized node features using only local structural statistics.
    # It applies z-score normalization with safeguards for near-zero standard deviation.
 
    features = compute_local_features(adj_matrix, num_nodes)
    
    # Z-score 
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


class StaticGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, out_dim=1):
        # Initializes the object and stores the layers or graph containers needed later.
        # It does not run training or inference by itself.
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
         # Runs the forward pass for the model.
        # It returns normalized edge or node predictions depending on the model class.
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return torch.sigmoid(x)

def train_gat_model(model, data, max_epochs, lr, device, patience=20):
  # Trains the GAT model on observed weighted edges.
    # It uses MSE loss on reconstructed edge weights and stops early when the loss stops improving.
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
        else:
            patience_counter += 1
        if patience_counter >= patience: break
    return model


def run_pipeline(train_file, test_file, output_json, params):
    # Runs the full experiment pipeline for one train/test split.
    # It loads data, trains the model, predicts held-out edge weights, evaluates metrics, and saves JSON output.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 [Ablation Model A] Local Features ONLY | No RW")
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
    
   
    features = build_features_local_only(adj_matrix, num_nodes).to(device)
    
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
        "meta_data": {"algorithm": "GAT-Local_LocalOnly", "dataset": params['dataset_name']},
        "metrics": {"mse": round(float(mse), 6), "pcc": round(float(pcc), 6)},
        "raw_results": [{"u": p[0], "v": p[1], "true_w": round(t,6), "pred_w": round(p[2],6)} for t, p in zip(true_values, predictions)]
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Model A Finished. PCC: {pcc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default="result_ablation_A.json")
    parser.add_argument("--dataset_name", default="Unknown")
    # 默认参数
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    run_pipeline(args.train_file, args.test_file, args.output_file, vars(args))