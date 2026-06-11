import math
import argparse
import json
import time
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. Data loading ====================
def load_graph_and_test_set(train_file, test_file):
    print(f"▶ Reading training graph: {train_file} ...")

    # 1. Build training graph (adjacency list: {u: {v1, v2...}})
    adj = {}
    nodes = set()

    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                parts = line.split(',')
                u, v = int(parts[0]), int(parts[1])
                # w = float(parts[2]) # CN/AA only looks at topology, not weights, or just existence

                nodes.add(u); nodes.add(v)

                # Undirected graph handling
                if u not in adj: adj[u] = set()
                if v not in adj: adj[v] = set()
                adj[u].add(v)
                adj[v].add(u)
            except ValueError: continue

    print(f"  - Training graph built: {len(nodes)} nodes")

    # 2. Read test set
    print(f"▶ Reading test set: {test_file} ...")
    test_pairs = [] # [(u, v, true_w)]

    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                parts = line.split(',')
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2])
                test_pairs.append((u, v, w))
            except ValueError: continue

    print(f"  - Number of test edges: {len(test_pairs)}")

    return adj, test_pairs

# ==================== 2. Algorithm implementations ====================

def run_common_neighbors(adj, u, v):
    """Common Neighbors: number of common neighbors"""
    if u not in adj or v not in adj:
        return 0.0

    # Python set intersection is very fast
    common = adj[u].intersection(adj[v])
    return float(len(common))

def run_adamic_adar(adj, u, v):
    """Adamic/Adar: sum of inverse log of common neighbor degrees"""
    if u not in adj or v not in adj:
        return 0.0

    common = adj[u].intersection(adj[v])
    score = 0.0
    for z in common:
        degree = len(adj[z])
        if degree > 1: # log(1)=0, avoid division by zero
            score += 1 / math.log(degree)
    return score

# ==================== 3. Main flow ====================
def main():
    parser = argparse.ArgumentParser()
    # Default paths
    default_train = "dataset/preprocessed/amazon_weight/training_graphs/graph_minus_200_edges.txt"
    default_test  = "dataset/preprocessed/amazon_weight/test_sets/graph_minus_200_REMOVED.txt"

    parser.add_argument('--method', type=str, required=True, choices=['CN', 'AA'], help='Choose algorithm: CN or AA')
    parser.add_argument('--train_file', type=str, default=default_train)
    parser.add_argument('--test_file', type=str, default=default_test)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--mask_cnt", type=int, required=True)
    args = parser.parse_args()

    # Output filename
    output_json = args.output_file

    # 1. Load data
    adj, test_pairs = load_graph_and_test_set(args.train_file, args.test_file)

    # 2. Run algorithm (and time it)
    print(f"\n🚀 Running {args.method} algorithm...")
    start_time = time.time()

    y_true = []
    y_scores = []
    raw_results_temp = [] # Temporary storage, not yet normalized

    for u, v, w in tqdm(test_pairs):
        if args.method == 'CN':
            score = run_common_neighbors(adj, u, v)
        else: # AA
            score = run_adamic_adar(adj, u, v)

        y_true.append(w)
        y_scores.append(score)
        raw_results_temp.append({"u": u, "v": v, "true_w": w})

    inference_time = time.time() - start_time

    # 3. Normalization and metric calculation
    # Note: CN/AA scores range in [0, +inf), while true weights are in [0, 1].
    # To compute MSE, we must map scores to the [0, 1] interval.
    # PCC does not need normalization, it only looks at linear relationship.

    y_scores_np = np.array(y_scores)
    y_true_np = np.array(y_true)

    min_s, max_s = y_scores_np.min(), y_scores_np.max()

    if max_s - min_s == 0:
        y_pred_norm = np.zeros_like(y_scores_np)
    else:
        # Min-Max Normalization
        y_pred_norm = (y_scores_np - min_s) / (max_s - min_s)

    # Compute metrics
    mse = mean_squared_error(y_true_np, y_pred_norm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np, y_pred_norm)
    pcc, _ = pearsonr(y_true_np, y_scores_np) # Use raw scores for PCC

    print("\n" + "="*40)
    print(f"📊 {args.method} Results:")
    print(f"   MSE:  {mse:.6f} (after Min-Max normalization)")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   PCC:  {pcc:.6f}")
    print(f"   Time: {inference_time:.4f}s")
    print("="*40)

    # 4. Export JSON
    # Store normalized pred_w in raw_results for easy comparison later
    final_raw_results = []
    for i, item in enumerate(raw_results_temp):
        item["pred_w"] = float(y_pred_norm[i]) # Store normalized value
        # item["raw_score"] = float(y_scores[i]) # Optional: if you want to keep raw CN/AA scores
        final_raw_results.append(item)

    output_data = {
        "meta_data": {
            "algorithm": "CommonNeighbors" if args.method == 'CN' else "AdamicAdar",
            "dataset": args.dataset,
            "mask_cnt": args.mask_cnt,
            "mask_ratio": f"{args.mask_cnt}_edges",
            "experiment_type": "heuristic_baseline",
            "hyper_parameters": {
                "method": args.method,
                "normalization": "Min-Max"
            }
        },
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "pcc": float(pcc),
            "inference_time_seconds": float(inference_time)
        },
        "raw_results": final_raw_results
    }

    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {output_json}")
    except Exception as e:
        print(f"❌ Save failed: {e}")

if __name__ == "__main__":
    main()