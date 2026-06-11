import os
import json
import argparse
import sys
import re

def find_best_combination(dataset_name):
    # 1. Determine results directory
    base_results_dir = f"../results/parameter_experiment/{dataset_name}"
    
    if not os.path.exists(base_results_dir):
        print(f"❌ Error: cannot find results directory: {base_results_dir}")
        sys.exit(1)

    print(f"🔍 Scanning results directory: {base_results_dir} ...")

    all_experiments = []

    # 2. Recursively traverse all JSON files
    for root, dirs, files in os.walk(base_results_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        params = data.get("meta_data", {}).get("hyper_parameters", {})
                        
                        if "pcc" in metrics:
                            all_experiments.append({
                                "file_name": file,
                                "metrics": metrics,
                                "params": params
                            })
                except Exception as e:
                    pass

    if not all_experiments:
        print("❌ No valid JSON result files found.")
        sys.exit(1)

    print(f"✅ Successfully loaded {len(all_experiments)} experiment results.\n")

    # 3. Find best results
    best_pcc_exp = max(all_experiments, key=lambda x: x["metrics"]["pcc"])
    best_mse_exp = min(all_experiments, key=lambda x: x["metrics"]["mse"])

    # 4. Print results
    print("=" * 60)
    print(f"🏆 Dataset: {dataset_name} Parameter Sensitivity Analysis Report")
    print("=" * 60)

    print(f"\n🥇 [PCC Champion] (Highest correlation)")
    print(f"   📂 File name: {best_pcc_exp['file_name']}")
    print(f"   📈 PCC: {best_pcc_exp['metrics']['pcc']:.6f}")
    print(f"   📉 MSE: {best_pcc_exp['metrics']['mse']:.6f}")
    
    print("-" * 60)

    # 5. Show best values per parameter category (fix Unknown issue)
    print("\n📊 Best performance per parameter dimension (based on PCC):")
    
    # Define parameters to analyze and their possible aliases in JSON
    # Format: "display_name": ["JSON_key1", "JSON_key2"]
    param_map = {
        "hidden_dim": ["hidden_dim"],
        "heads":      ["heads"], 
        "lr":         ["lr", "learning_rate"], 
        "L":          ["L", "rw_L"], 
        "n_walks":    ["n_walks", "rw_n_walks"], 
        "alpha":      ["alpha"]
    }
    
    for p_display, p_keys in param_map.items():
        # Find all experiments that contain this parameter in their filename
        relevant_exps = [e for e in all_experiments if p_display in e['file_name']]
        
        if relevant_exps:
            # Find the one with highest PCC in this group
            best_in_group = max(relevant_exps, key=lambda x: x["metrics"]["pcc"])
            
            # --- Core fix logic: get parameter value ---
            val = "Unknown"
            
            # 1. Try to get from JSON params (support aliases)
            for key in p_keys:
                if key in best_in_group['params']:
                    val = best_in_group['params'][key]
                    break
            
            # 2. If not in JSON (e.g., heads), try to extract from filename
            if val == "Unknown":
                # Regex extract: parameter name + (digits or p combination)
                match = re.search(f"{p_display}(\d+p?\d*)", best_in_group['file_name'])
                if match:
                    val_str = match.group(1)
                    val = val_str.replace('p', '.') # convert 0p01 to 0.01
            
            print(f"   🔹 {p_display:10s} Best value -> {val} (PCC: {best_in_group['metrics']['pcc']:.4f})")
        else:
            # If no relevant file found, skip
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    find_best_combination(args.dataset)