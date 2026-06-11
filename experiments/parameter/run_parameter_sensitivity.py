import os
import subprocess
import argparse
import sys

def run_sensitivity_experiment(dataset_name, data_dir, output_dir):
    """
    Run a full parameter sensitivity experiment for the minus_100 version of the specified dataset.
    """
    
    # ================= Configuration =================
    
    # 1. Path configuration (hardcoded according to your screenshot)
    base_data_dir = data_dir
    train_file = os.path.join(base_data_dir, "training_graphs", "graph_minus_100_edges.txt")
    test_file  = os.path.join(base_data_dir, "test_sets", "graph_minus_100_REMOVED.txt")
    
    # 2. Check if files exist (avoid running on empty)
    if not os.path.exists(train_file):
        print(f"❌ Error: training file not found: {train_file}")
        sys.exit(1)
    if not os.path.exists(test_file):
        print(f"❌ Error: test file not found: {test_file}")
        sys.exit(1)

    # 3. Base default parameters (Baseline)
    default_params = {
        "hidden_dim": 32,
        "heads":      4,
        "lr":         0.01,
        "L":          6,
        "n_walks":    3000,
        "alpha":      0.2,
        "epochs":     1000  # Fixed large value, rely on early stopping
    }

    # 4. Parameter grid to test
    param_grid = {
        "hidden_dim": [16, 32],
        "heads":      [2, 4, 6],
        "lr":         [0.01, 0.005, 0.001],
        "L":          [4, 5, 6, 7],
        "n_walks":    [ 2000, 3000, 4000, 5000],
        "alpha":      [0.05, 0.1, 0.15, 0.2, 0.25]
    }
    
    # 5. Parameter name mapping
    arg_map = {
        "hidden_dim": "--hidden_dim",
        "heads":      "--heads",
        "lr":         "--lr",
        "L":          "--L",
        "n_walks":    "--n_walks",
        "alpha":      "--alpha"
    }
    
    # ===========================================

    # Result save path: ../results/parameter_result/{dataset_name}/
    base_output_dir = output_dir
    model_name = "GAT_RW"
    experiment_type = "Sensitivity"

    print(f"🚀 Starting parameter sensitivity analysis")
    print(f"🎯 Dataset: {dataset_name} (Mask: minus_100)")
    print(f"📂 Results will be saved to: {base_output_dir}/")

    # Loop over each parameter to test
    for param_name, values in param_grid.items():
        print(f"\n==================================================")
        print(f">>> Testing parameter: {param_name} (range: {values})")
        print(f"==================================================")
        
        for val in values:
            # 1. Format value for safe filename (e.g., 0.01 -> 0p01)
            val_str_safe = str(val).replace('.', 'p')
            
            # 2. Build subfolder: ../results/parameter_result/dblp_weight/lr/lr0p01/
            sub_dir_name = f"{param_name}{val_str_safe}"
            save_dir = os.path.join(base_output_dir, param_name, sub_dir_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 3. Build output filename
            json_filename = f"{model_name}_{dataset_name}_{experiment_type}_{param_name}{val_str_safe}.json"
            output_path = os.path.join(save_dir, json_filename)

            # 4. Check if exists
            if os.path.exists(output_path):
                print(f"  [Skip] File already exists: {json_filename}")
                continue

            # 5. Assemble command
            cmd = [
                "python", "src/gat_rw_experiment.py",
                "--train_file", train_file,
                "--test_file", test_file,
                "--output_file", output_path,
                "--dataset_name", dataset_name,
                "--mask_cnt", "100", # Explicitly tell GAT code this is minus 100
                "--epochs", str(default_params["epochs"])
            ]

            # Dynamic parameter assembly
            for p_key, p_flag in arg_map.items():
                if p_key == param_name:
                    cmd.extend([p_flag, str(val)])
                else:
                    cmd.extend([p_flag, str(default_params[p_key])])
            
            # 6. Execute command
            print(f"  ▶️  Running: {param_name} = {val} ...")
            try:
                # Capture output, show only on error
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"     ✅ Finished. Result: {output_path}")
                else:
                    print(f"     ❌ Failed! Error log:\n{result.stderr}")
            except Exception as e:
                print(f"     ❌ Execution exception: {e}")

    print("\n🎉 All parameter sensitivity experiments completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Only need dataset name (e.g., dblp_weight)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    run_sensitivity_experiment(args.dataset, args.data_dir, args.output_dir)