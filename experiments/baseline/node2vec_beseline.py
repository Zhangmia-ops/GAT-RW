import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
import os
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import PyG's Node2Vec
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader, TensorDataset

# ==================== 1. Data loading (two-file mode) ====================
def load_data_split(train_file, test_file):
    print(f"▶ Reading training data: {train_file} ...")
    print(f"▶ Reading test data: {test_file} ...")

    node_set = set()
    train_edges_list = [] # [(u,v,w)]
    test_edges_list = []  # [(u,v,w)]

    # Read training set
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                parts = line.split(',')
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2])
                node_set.add(u); node_set.add(v)
                train_edges_list.append((u, v, w))
            except ValueError: continue

    # Read test set
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                parts = line.split(',')
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2])
                node_set.add(u); node_set.add(v)
                test_edges_list.append((u, v, w))
            except ValueError: continue

    # Build mapping
    sorted_nodes = sorted(list(node_set))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    num_nodes = len(node_map)

    print(f"  - Total nodes: {num_nodes}")
    print(f"  - Training edges: {len(train_edges_list)}")
    print(f"  - Test edges: {len(test_edges_list)}")

    # 1. Build edge_index for Node2Vec (undirected, unweighted)
    # Node2Vec only needs to know which nodes are connected to perform walks
    n2v_edges = []
    for u, v, _ in train_edges_list:
        idx_u, idx_v = node_map[u], node_map[v]
        n2v_edges.append([idx_u, idx_v])
        n2v_edges.append([idx_v, idx_u]) # bidirectional

    n2v_edge_index = torch.tensor(n2v_edges, dtype=torch.long).t().contiguous()

    # 2. Build training data for regressor (u, v) -> w
    reg_train_idx = []
    reg_train_wt = []
    for u, v, w in train_edges_list:
        reg_train_idx.append([node_map[u], node_map[v]])
        reg_train_wt.append(w)
        # Optionally add bidirectional to double training data for regressor
        reg_train_idx.append([node_map[v], node_map[u]])
        reg_train_wt.append(w)

    reg_train_inputs = torch.tensor(reg_train_idx, dtype=torch.long)
    reg_train_targets = torch.tensor(reg_train_wt, dtype=torch.float)

    # 3. Build test data
    test_inputs_list = []
    test_targets_list = []
    test_raw_info = []

    for u, v, w in test_edges_list:
        test_inputs_list.append([node_map[u], node_map[v]])
        test_targets_list.append(w)
        test_raw_info.append({"u": u, "v": v, "true_w": w})

    test_inputs = torch.tensor(test_inputs_list, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.float)

    return num_nodes, n2v_edge_index, reg_train_inputs, reg_train_targets, test_inputs, test_targets, test_raw_info

# ==================== 2. Regressor model definition ====================
class LinkRegressor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(LinkRegressor, self).__init__()
        # Input is concatenation of two node embeddings
        self.lin1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, z_src, z_dst):
        cat = torch.cat([z_src, z_dst], dim=-1)
        x = torch.relu(self.lin1(cat))
        x = self.lin2(x)
        return torch.sigmoid(x).view(-1)

# ==================== 3. Main flow ====================
def main():
    parser = argparse.ArgumentParser()
    # Default paths
    default_train = "dataset/preprocessed/Fruit-Fly/training_graphs/graph_minus_200_edges.txt"
    default_test  = "dataset/preprocessed/Fruit-Fly/test_sets/graph_minus_200_REMOVED.txt"

    parser.add_argument('--train_file', type=str, default=default_train)
    parser.add_argument('--test_file', type=str, default=default_test)
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--epochs_n2v', type=int, default=5, help='Node2Vec training epochs')
    parser.add_argument('--epochs_reg', type=int, default=50, help='Regressor training epochs')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_json = "base_result/Node2Vec_Fruit-Fly_minus200.json"

    # 1. Load data
    data = load_data_split(args.train_file, args.test_file)
    num_nodes, n2v_edge_index, reg_train_inputs, reg_train_targets, test_inputs, test_targets, test_raw_info = data

    n2v_edge_index = n2v_edge_index.to(device)
    reg_train_inputs = reg_train_inputs.to(device)
    reg_train_targets = reg_train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    # ================= Phase 1: Train Node2Vec (feature extraction) =================
    print(f"\n🚀 [Phase 1] Training Node2Vec Embeddings ({args.epochs_n2v} epochs)...")
    train_start = time.time()

    n2v_model = Node2Vec(
        n2v_edge_index,
        embedding_dim=args.dim,
        walk_length=10,
        context_size=5,
        walks_per_node=10,
        num_negative_samples=1,
        p=1, q=1,
        sparse=True # Sparse gradient, speeds up training
    ).to(device)

    loader = n2v_model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer_n2v = torch.optim.SparseAdam(n2v_model.parameters(), lr=0.01)

    n2v_model.train()
    for epoch in range(args.epochs_n2v):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, desc=f"  N2V Epoch {epoch+1}"):
            optimizer_n2v.zero_grad()
            loss = n2v_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer_n2v.step()
            total_loss += loss.item()

    # Freeze embeddings as features
    n2v_model.eval()
    with torch.no_grad():
        node_embeddings = n2v_model().detach() # [N, dim]

    # ================= Phase 2: Train Regressor (edge weight prediction) =================
    print(f"\n🚀 [Phase 2] Training Link Regressor ({args.epochs_reg} epochs)...")

    regressor = LinkRegressor(args.dim).to(device)
    optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Build DataLoader
    reg_dataset = TensorDataset(reg_train_inputs, reg_train_targets)
    reg_loader = DataLoader(reg_dataset, batch_size=1024, shuffle=True)

    regressor.train()
    for epoch in range(args.epochs_reg):
        epoch_loss = 0
        for batch_idx, batch_y in reg_loader:
            optimizer_reg.zero_grad()

            # Look up embeddings
            src_emb = node_embeddings[batch_idx[:, 0]]
            dst_emb = node_embeddings[batch_idx[:, 1]]

            pred = regressor(src_emb, dst_emb)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer_reg.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"  Reg Epoch {epoch+1} Loss: {epoch_loss/len(reg_loader):.4f}")

    train_time = time.time() - train_start

    # ================= Phase 3: Inference and evaluation =================
    print("\n▶ Starting inference evaluation...")
    inference_start = time.time()

    regressor.eval()
    with torch.no_grad():
        src_emb = node_embeddings[test_inputs[:, 0]]
        dst_emb = node_embeddings[test_inputs[:, 1]]
        preds = regressor(src_emb, dst_emb)

    inference_time = time.time() - inference_start

    # Metric calculation
    y_pred = preds.cpu().numpy()
    y_true = test_targets.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    print("\n" + "="*40)
    print(f"📊 Node2Vec Results:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   PCC:  {pcc:.6f}")
    print("="*40)

    # ================= Phase 4: Export JSON =================
    # Build raw_results
    raw_results = []
    for i, info in enumerate(test_raw_info):
        raw_results.append({
            "u": info['u'],
            "v": info['v'],
            "true_w": float(info['true_w']),
            "pred_w": float(y_pred[i])
        })

    output_data = {
        "meta_data": {
            "algorithm": "Node2Vec",
            "dataset": "Amazon_Minus_200",
            "experiment_type": "baseline",
            "hyper_parameters": {
                "embedding_dim": args.dim,
                "n2v_epochs": args.epochs_n2v,
                "reg_epochs": args.epochs_reg,
                "walk_length": 10,
                "walks_per_node": 10
            }
        },
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "pcc": float(pcc),
            "train_time_seconds": float(train_time),
            "inference_time_seconds": float(inference_time)
        },
        "raw_results": raw_results
    }

    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {output_json}")
    except Exception as e:
        print(f"❌ Save failed: {e}")

if __name__ == "__main__":
    main()