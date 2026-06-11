# This script orchestrates the full ablation-study run across several masking levels.
# It calls each ablation model script with shared hyperparameters and dataset paths.
# It skips completed outputs and stores each model result in a structured output directory.

import os
import subprocess
import argparse
import sys

def run_ablation_suite(dataset_name, data_dir, output_dir):

    mask_counts = [20, 50, 100, 150, 200]

    best_params = {
        "hidden_dim": 32,  
        "heads": 4,
        "lr": 0.01,
        "epochs": 1000,

        "L": 6,
        "n_walks": 3000,
        "alpha": 0.2
    }

    models = {
        "Model_A": "GAT-Local.py",  
        "Model_B": "GAT-Spectral.py", 
        "Model_C": "gat_rw_experiment.py",   
        "Model_D": "GAT-NaiveRW.py"
    }
    

    base_output_dir = output_dir

    data_root = data_dir

    print(f"🚀 Starting full ablation study: {dataset_name}")
    print(f"📂 Results will be saved to: {base_output_dir}/")
    print(f"⚙️  Testing difficulty levels: {mask_counts}")

    for mask in mask_counts:
        print(f"\n------------------------------------------------")
        print(f"🔥 Processing difficulty: Minus {mask}")
        print(f"------------------------------------------------")

        # Build data paths
        train_file = os.path.join(data_root, "training_graphs", f"graph_minus_{mask}_edges.txt")
        test_file  = os.path.join(data_root, "test_sets", f"graph_minus_{mask}_REMOVED.txt")

        # Check if data exists
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"⚠️  Skipping Minus {mask}: data files not found.")
            continue

        for model_name, script_name in models.items():
            
            model_out_dir = os.path.join(base_output_dir, model_name)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)
            
            json_name = f"{dataset_name}_{model_name}_minus{mask}.json"
            output_path = os.path.join(model_out_dir, json_name)
            
            if os.path.exists(output_path):
                print(f"  [Skip] {model_name} already exists.")
                continue

            print(f"  ▶️  Running model: {model_name} ...")

            cmd = [
                "python", os.path.join("experiments", "ablation", script_name),
                "--train_file", train_file,
                "--test_file", test_file,
                "--output_file", output_path,
                "--dataset_name", dataset_name,
                "--hidden_dim", str(best_params["hidden_dim"]),
                "--heads", str(best_params["heads"]),
                "--lr", str(best_params["lr"]),
                "--epochs", str(best_params["epochs"])
            ]

            if model_name in ["Model_C", "Model_D"]:
                cmd.extend([
                    "--mask_cnt", str(mask), 
                    "--L", str(best_params["L"]),
                    "--n_walks", str(best_params["n_walks"]),
                    "--alpha", str(best_params["alpha"])
                ])

            try:
                # Run
                subprocess.run(cmd, check=True)
                print(f"     ✅ {model_name} finished.")
            except subprocess.CalledProcessError as e:
                print(f"     ❌ {model_name} failed! (Exit code: {e.returncode})")
            except Exception as e:
                print(f"     ❌ {model_name} exception: {e}")

    print("\n🎉 Full ablation study finished! Time to draw line charts!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    run_ablation_suite(args.dataset, args.data_dir, args.output_dir)