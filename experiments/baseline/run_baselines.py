import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Import model definitions (ensure baseline_models.py is in the same directory)
from baseline_models import GCN_Predictor, SAGE_Predictor, MLP_Predictor

# ==================== 1. New data loading (two-file mode) ====================
def load_data_split(train_file, test_file):
    print(f"▶ Reading training data: {train_file} ...")
    print(f"▶ Reading test data: {test_file} ...")

    node_set = set()

    # Temporary storage lists
    train_edges_list = []   # [(u,v,w), ...]
    test_edges_list = []    # [(u,v,w), ...]

    # --- 1. Read training set ---
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

    # --- 2. Read test set ---
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                parts = line.split(',')
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2])
                # Note: nodes in test set must also be added to node_set,
                # otherwise mapping will fail if there are isolated nodes in test.
                node_set.add(u); node_set.add(v)
                test_edges_list.append((u, v, w))
            except ValueError: continue

    # --- 3. Build unified node mapping ---
    sorted_nodes = sorted(list(node_set))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    num_nodes = len(node_map)

    print(f"  - Total nodes (Union): {num_nodes}")
    print(f"  - Training edges: {len(train_edges_list)}")
    print(f"  - Test edges: {len(test_edges_list)}")

    # --- 4. Build Tensors ---

    # A. Build training graph (undirected, add bidirectional)
    train_edge_index_list = []
    train_edge_weight_list = []

    for u, v, w in train_edges_list:
        idx_u, idx_v = node_map[u], node_map[v]
        # Add (u, v)
        train_edge_index_list.append([idx_u, idx_v])
        train_edge_weight_list.append(w)
        # Add (v, u)
        train_edge_index_list.append([idx_v, idx_u])
        train_edge_weight_list.append(w)

    train_idx = torch.tensor(train_edge_index_list, dtype=torch.long).t().contiguous()
    train_wt = torch.tensor(train_edge_weight_list, dtype=torch.float)

    # B. Build test pairs (unidirectional, for prediction)
    test_idx_list = []
    test_wt_list = []
    test_edges_orig = [] # Store original info for JSON

    for u, v, w in test_edges_list:
        idx_u, idx_v = node_map[u], node_map[v]
        test_idx_list.append([idx_u, idx_v])
        test_wt_list.append(w)
        test_edges_orig.append({"u": u, "v": v, "true_w": w})

    test_idx = torch.tensor(test_idx_list, dtype=torch.long).t().contiguous()
    test_wt = torch.tensor(test_wt_list, dtype=torch.float)

    return num_nodes, train_idx, train_wt, test_idx, test_wt, test_edges_orig

# ==================== 2. Training and evaluation flow ====================
def train_and_evaluate(model_name, train_file, test_file, epochs=200, lr=0.01, device='cpu', output_json=None,dataset=None, mask_cnt=None):

    # 1. Load data
    num_nodes, train_idx, train_wt, test_idx, test_wt, test_info_list = load_data_split(train_file, test_file)

    train_idx, train_wt = train_idx.to(device), train_wt.to(device)
    test_idx, test_wt = test_idx.to(device), test_wt.to(device)

    # 2. Initialize model
    print(f"\n🚀 Initializing model: {model_name}")
    if model_name == 'GCN':
        model = GCN_Predictor(num_nodes, 64, 64)
    elif model_name == 'SAGE':
        model = SAGE_Predictor(num_nodes, 64, 64)
    elif model_name == 'MLP':
        model = MLP_Predictor(num_nodes, 64, 64)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Training
    print("▶ Starting training...")
    train_start = time.time()

    model.train()
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()
        # Forward pass using training graph structure
        pred = model(train_idx, train_wt, train_idx)
        loss = criterion(pred, train_wt)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    train_time = time.time() - train_start

    # 4. Inference
    print("▶ Starting inference...")
    inference_start = time.time()

    model.eval()
    with torch.no_grad():
        # Note: encode nodes using the training graph (train_idx),
        # then predict weights for test edges (test_idx)
        preds = model(train_idx, train_wt, test_idx)

    inference_time = time.time() - inference_start

    # Convert to numpy
    y_pred = preds.cpu().numpy()
    y_true = test_wt.cpu().numpy()

    # Metric calculation
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    print("\n" + "="*40)
    print(f"📊 {model_name} Results:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   PCC:  {pcc:.6f}")
    print("="*40)

    # 5. Export JSON
    if output_json:
        # Assemble raw_results
        raw_results = []
        for i, info in enumerate(test_info_list):
            item = {
                "u": info['u'],
                "v": info['v'],
                "true_w": float(info['true_w']),
                "pred_w": float(y_pred[i])
            }
            raw_results.append(item)

        output_data = {
            "meta_data": {
                "algorithm": model_name,
                "dataset": dataset,
                "experiment_type": "baseline_benchmark",
                "mask_ratio": f"{mask_cnt}_edges",
                "mask_cnt": int(mask_cnt),
                "random_seed": 42,
                "hyper_parameters": {
                    "embedding_dim": 64,
                    "hidden_dim": 64,
                    "learning_rate": lr,
                    "epochs": epochs
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

# ==================== Entry point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default file paths (modify to your actual absolute or relative paths)
    default_train = "data/ust_splits/amazon_weight/training_graphs/graph_minus_200_edges.txt"
    default_test  = "data/ust_splits/amazon_weight/test_sets/graph_minus_200_REMOVED.txt"

    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'SAGE', 'MLP'])
    parser.add_argument('--train_file', type=str, default=default_train, help='Training set path')
    parser.add_argument('--test_file', type=str, default=default_test, help='Test set path (with ground truth)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.train_file):
        print(f"❌ Error: training file not found at {args.train_file}")
        exit(1)
    if not os.path.exists(args.test_file):
        print(f"❌ Error: test file not found at {args.test_file}")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mask_cnt = os.path.basename(args.test_file).split("minus_")[-1].split("_")[0]

    # Output filename
    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_minus{mask_cnt}.json")

    train_and_evaluate(
        args.model,
        args.train_file,
        args.test_file,
        args.epochs,
        lr=0.01,
        device=device,
        output_json=output_json,
        dataset=args.dataset,
        mask_cnt=mask_cnt
    )