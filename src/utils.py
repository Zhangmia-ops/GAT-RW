import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from graph import DynamicWeightedGraph
from scipy.stats import rankdata, pearsonr # Need this for rank calculation


# ==================== Evaluation and plotting functions ====================

def load_ground_truth(filename: str) -> Dict[Tuple[int, int], float]:
    """
    Load ground truth edge weights.
    Returns dictionary: {(min(u,v), max(u,v)): weight}
    """
    truth = {}
    print(f"Loading ground truth from {filename}...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',')
                if len(parts) < 3: continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                    w = float(parts[2])
                    key = tuple(sorted((u, v)))
                    truth[key] = w
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: Ground truth file {filename} not found!")
    return truth

def evaluate_and_plot(revised_data: List[Tuple], ground_truth_path: str, plot_filename: str):
    """
    Compute MSE and plot.
    """
    truth_dict = load_ground_truth(ground_truth_path)
    
    y_true = []
    y_orig = []
    y_revised = []
    missed_count = 0
    
    for item in revised_data:
        # item: (u, v, w_orig, raw_prob, p_mapped, w_revised)
        u, v, w_orig, _, _, w_revised = item
        key = tuple(sorted((u, v)))
        
        if key in truth_dict:
            t = truth_dict[key]
            y_true.append(t)
            y_orig.append(w_orig)
            y_revised.append(w_revised)
        else:
            missed_count += 1

    if not y_true:
        print("❌ Cannot compute MSE: No ground truth found for any completed edge in the Ground Truth file.")
        return

    y_true = np.array(y_true)
    y_orig = np.array(y_orig)
    y_revised = np.array(y_revised)
    
    # --- Compute metrics ---
    mse_orig = np.mean((y_true - y_orig) ** 2)
    mse_revised = np.mean((y_true - y_revised) ** 2)
    mae_orig = np.mean(np.abs(y_true - y_orig))
    mae_revised = np.mean(np.abs(y_true - y_revised))
    
    pcc_orig, _ = pearsonr(y_true, y_orig)
    pcc_revised, _ = pearsonr(y_true, y_revised)

    mse_imp = (mse_orig - mse_revised) / mse_orig * 100
    pcc_imp = (pcc_revised - pcc_orig) / abs(pcc_orig) * 100 # PCC larger is better
    
    print("\n" + "="*50)
    print(f"📊 Evaluation results (based on {len(y_true)} edges with ground truth)")
    print(f"   (Edges without ground truth: {missed_count})")
    print("-" * 50)
    print(f"Metric       | Before Rev. | After Rev.  | Change")
    print("-" * 50)
    print(f"MSE (↓)    | {mse_orig:.6f}  | {mse_revised:.6f}  | {mse_imp:+.2f}%")
    print(f"MAE (↓)    | {mae_orig:.6f}  | {mae_revised:.6f}  |")
    print(f"PCC (↑)    | {pcc_orig:.6f}  | {pcc_revised:.6f}  | {pcc_imp:+.2f}%")
    print("="*50 + "\n")

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter comparison
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_orig, alpha=0.5, label=f'Original (MSE={mse_orig:.4f})', c='blue', s=20)
    plt.scatter(y_true, y_revised, alpha=0.6, label=f'Revised (MSE={mse_revised:.4f})', c='red', s=20, marker='x')
    
    # Draw y=x diagonal line
    min_val = min(np.min(y_true), np.min(y_orig), np.min(y_revised))
    max_val = max(np.max(y_true), np.max(y_orig), np.max(y_revised))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Weight')
    plt.ylabel('Predicted / Revised Weight')
    plt.title('Weight Correction Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Error distribution histogram
    plt.subplot(1, 2, 2)
    error_orig = np.abs(y_true - y_orig)
    error_rev = np.abs(y_true - y_revised)
    
    plt.hist(error_orig, bins=20, alpha=0.5, label='Original Error', color='blue')
    plt.hist(error_rev, bins=20, alpha=0.5, label='Revised Error', color='red')
    plt.xlabel('Absolute Error |True - Pred|')
    plt.ylabel('Count')
    plt.title('Error Distribution Histogram')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"📈 Evaluation chart saved to: {plot_filename}")

# ==================== Core logic functions ====================

def read_graph_from_file(filename: str) -> Tuple[DynamicWeightedGraph, List[Tuple[int, int, float]]]:
    graph = DynamicWeightedGraph()
    completion_edges = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line: continue
            is_completion = line.endswith('#') and ',#' not in line
            if is_completion: line = line[:-1].strip()
            try:
                u, v, w = map(float, line.split(','))
                u, v = int(u), int(v)
            except ValueError: continue
            if is_completion: completion_edges.append((u, v, w))
            else:
                graph.add_edge(u, v, w)
                graph.add_edge(v, u, w)
    return graph, completion_edges

def calibrate_temperatures(graph, pi_all, neg_ratio=2.0):
    """
    Calibrate random walk frequencies using Platt Scaling
    """
    calibrated_pi = {u: {} for u in pi_all}
    for u in tqdm(pi_all, desc="Calibrating temperatures"):
        existing_edges = graph.edges[u]
        pos_probs = [pi_all[u].get(v, 0.0) for v in existing_edges]
        if not pos_probs:
            calibrated_pi[u] = pi_all[u]; continue
        num_pos = len(pos_probs)
        num_neg = int(num_pos * neg_ratio)
        non_neighbors = graph.get_non_neighbors(u, num_neg)
        neg_probs = [pi_all[u].get(w, 0.0) for w in non_neighbors]
        probs = np.array(pos_probs + neg_probs)
        labels = np.array([1] * num_pos + [0] * num_neg)
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs / (1 - probs))
        
        def nll_loss(T):
            cal_probs = 1 / (1 + np.exp(-logits / T))
            cal_probs = np.clip(cal_probs, epsilon, 1 - epsilon)
            losses = - (labels * np.log(cal_probs) + (1 - labels) * np.log(1 - cal_probs))
            return np.mean(losses)
        try:
            res = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
            T_u = res.x if res.success else 1.0
        except: T_u = 1.0
        
        for v, p in pi_all[u].items():
            p_clipped = max(epsilon, min(1 - epsilon, p))
            logit = np.log(p_clipped / (1 - p_clipped))
            cal_p = 1 / (1 + np.exp(-logit / T_u))
            calibrated_pi[u][v] = cal_p
    return calibrated_pi



# def revise_completion_edges(completion_edges, pi_all, alpha=0.05, output_file="revised_edges.txt"):
#     """
#     [Battle Royale Edition: Directional Lock Ratchet Revision]
#     Strategy:
#     1. Very small Alpha (0.05): Since RW is noisy, give it very little influence.
#     2. Directional Lock:
#        - Prevent pulling down high scores.
#        - Prevent pulling up low scores.
#        - Only allow "Consistency Boosting".
#     """
#     revised_data_list = []
#     weights = []
#     probs = []
#
#     # --- 1. Collect data ---
#     for u, v, w in completion_edges:
#         pi_uv = pi_all.get(u, {}).get(v, 0.0)
#         pi_vu = pi_all.get(v, {}).get(u, 0.0)
#         avg_prob = (pi_uv + pi_vu) / 2
#         weights.append(w)
#         probs.append(avg_prob)
#
#     if not weights: return []
#
#     weights_np = np.array(weights)
#     probs_np = np.array(probs)
#
#     # --- 2. Compute relative ranks (0.0 ~ 1.0) ---
#     # We still trust ranks more than numerical values
#     rank_w = rankdata(weights_np, method='average') / len(weights_np)
#     rank_p = rankdata(probs_np, method='average') / len(probs_np)
#
#     # --- 3. Core revision logic ---
#     new_weights = []
#
#     for i, w_orig in enumerate(weights_np):
#         r_gat = rank_w[i]
#         r_rw = rank_p[i]
#
#         # Raw adjustment based on rank difference
#         raw_delta = alpha * (r_rw - r_gat)
#
#         final_delta = 0.0
#
#         # ======= Core: Directional Lock logic =======
#
#         # Scenario A: GAT thinks it's a strong edge (> 0.65)
#         if w_orig > 0.65:
#             if raw_delta > 0:
#                 # GAT strong, RW thinks stronger -> allow icing on the cake
#                 final_delta = raw_delta
#             else:
#                 # GAT strong, RW thinks weak -> forbid pulling down! (protect bridges)
#                 final_delta = 0.0
#
#         # Scenario B: GAT thinks it's a weak edge (< 0.35)
#         elif w_orig < 0.35:
#             if raw_delta < 0:
#                 # GAT weak, RW thinks weaker -> allow kicking when down
#                 final_delta = raw_delta
#             else:
#                 # GAT weak, RW thinks strong -> forbid pulling up! (suppress false positives)
#                 final_delta = 0.0
#
#         # Scenario C: Fuzzy middle zone (0.35 ~ 0.65)
#         else:
#             # Only allow modification if RW signal is very strong
#             # e.g., RW rank in top 10% or bottom 10%
#             if r_rw > 0.9 or r_rw < 0.1:
#                 final_delta = raw_delta * 0.5 # conservative
#             else:
#                 final_delta = 0.0 # otherwise no change
#
#         # ===============================
#
#         new_w = w_orig + final_delta
#
#         # Clip
#         new_w = min(1.0, max(0.0, new_w))
#         new_weights.append(new_w)
#
#     # --- 4. Save ---
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write("u,v,original_weight,raw_prob,final_delta,revised_weight\n")
#
#         for i, (u, v, w_orig) in enumerate(completion_edges):
#             raw_p = probs[i]
#             new_w = new_weights[i]
#             delta = new_w - w_orig
#
#             f.write(f"{u},{v},{w_orig:.4f},{raw_p:.6f},{delta:.6f},{new_w:.6f}\n")
#             revised_data_list.append((u, v, w_orig, raw_p, delta, new_w))
#
#     print(f"Stats: number of edges modified: {np.sum(np.abs(np.array(new_weights) - weights_np) > 1e-6)}")
#     print(f"Stats: number of upward revisions (increase): {np.sum(np.array(new_weights) > weights_np)}")
#     print(f"Stats: number of downward revisions (decrease): {np.sum(np.array(new_weights) < weights_np)}")
#
#     return revised_data_list

from scipy.stats import rankdata
import numpy as np

def revise_completion_edges(completion_edges, pi_all, alpha=0.15, output_file="revised_edges.txt"):
    """
    [Master Edition: Smooth Gradient Suppression]
    Goal: Optimize both MSE and PCC.
    Method: Abandon hard thresholds, use (1-w)^2 as dynamic decay coefficient.
    Effect:
    1. Automatically apply maximum pressure in low score range (0~0.2) -> reduce MSE.
    2. Smooth transition throughout, no breakpoints -> improve PCC.
    3. High score range automatically tends to zero -> protect bridges.
    """
    revised_data_list = []
    weights = []
    probs = []
    
    # --- 1. Collect data ---
    for u, v, w in completion_edges:
        pi_uv = pi_all.get(u, {}).get(v, 0.0)
        pi_vu = pi_all.get(v, {}).get(u, 0.0)
        avg_prob = (pi_uv + pi_vu) / 2
        weights.append(w)
        probs.append(avg_prob)

    if not weights: return []
    
    weights_np = np.array(weights)
    probs_np = np.array(probs)

    # --- 2. Compute ranks ---
    rank_w = rankdata(weights_np, method='average') / len(weights_np)
    rank_p = rankdata(probs_np, method='average') / len(probs_np)
    
    # --- 3. Core revision logic ---
    new_weights = []
    
    # Slightly increase alpha, because multiplied by decay factor, actual strength will be smaller
    # Suggested try alpha = 0.15 or 0.2
    base_alpha = alpha 
    
    for i, w_orig in enumerate(weights_np):
        r_gat = rank_w[i]
        r_rw = rank_p[i]
        
        diff = r_rw - r_gat # RW - GAT
        
        # [Core innovation] Smooth decay factor
        # The closer to 0, the larger the factor (close to 1); the closer to 1, the smaller (close to 0)
        # Square makes it decay slower in low range (maintain high pressure), faster in high range
        decay_factor = (1.0 - w_orig) ** 2
        
        final_delta = 0.0
        
        # =========================================
        #   Logic A: RW wants to lower score (diff < 0)
        # =========================================
        if diff < 0:
            # This is a smooth "penalty", no abrupt threshold like if w < 0.4
            # Multiply by an extra factor to maintain previous suppression strength
            final_delta = diff * base_alpha * decay_factor * 2.0
            
        # =========================================
        #   Logic B: RW wants to raise score (diff > 0)
        # =========================================
        else:
            # Even in low range, we almost never raise score.
            # Give a very small coefficient, allow only tiny perturbations to maintain monotonicity
            final_delta = diff * base_alpha * decay_factor * 0.1
        
        # =========================================
        
        new_w = w_orig + final_delta
        new_w = min(1.0, max(0.0, new_w))
        new_weights.append(new_w)

    # --- 4. Save ---
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("u,v,original_weight,raw_prob,final_delta,revised_weight\n")
        
        for i, (u, v, w_orig) in enumerate(completion_edges):
            raw_p = probs[i]
            new_w = new_weights[i]
            delta = new_w - w_orig
            f.write(f"{u},{v},{w_orig:.4f},{raw_p:.6f},{delta:.6f},{new_w:.6f}\n")
            revised_data_list.append((u, v, w_orig, raw_p, delta, new_w))
            
    return revised_data_list

def save_probability_distributions_to_file(probability_distributions, filename):
    with open(filename, 'w') as file:
        for u in probability_distributions:
            file.write(f"Node {u}:\n")
            for v, p in probability_distributions[u].items():
                file.write(f"  {v}: {p:.6f}\n")
            file.write("\n")