import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

def calculate_metrics_from_raw(raw_data):
    """Compute RMSE and MAE from raw_results"""
    if not raw_data:
        return None, None
    
    y_true = [item['true_w'] for item in raw_data]
    y_pred = [item['pred_w'] for item in raw_data]
    
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return rmse, mae

def load_ablation_data(root_dir):
    records = []
    print(f"Scanning ablation experiment data: {root_dir} ...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Get model name (parent folder name)
                model_name = os.path.basename(os.path.dirname(file_path))
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 1. Extract minus value
                    filename = os.path.splitext(file)[0]
                    if 'minus' in filename:
                        # Parse like amazon_weight_Model_A_minus20 -> 20
                        minus_val = int(filename.split('minus')[-1])
                    else:
                        continue 

                    metrics = data.get('metrics', {})
                    raw_res = data.get('raw_results', [])
                    
                    # 2. Get base metrics
                    rmse = metrics.get('rmse')
                    mae = metrics.get('mae')
                    mse = metrics.get('mse')
                    pcc = metrics.get('pcc')
                    
                    # ==================== Core modification for missing values ====================
                    
                    # Case A: If RMSE missing but MSE present -> compute sqrt (for Model D)
                    if rmse is None and mse is not None:
                        rmse = np.sqrt(mse)
                        # print(f"Filled RMSE via sqrt(MSE): {model_name} - minus{minus_val}")

                    # Case B: If RMSE/MAE missing but raw_results available -> recalculate (for Model A/B)
                    # Note: only enter here if not already filled, or specifically for MAE
                    calc_rmse, calc_mae = calculate_metrics_from_raw(raw_res)
                    
                    if rmse is None and calc_rmse is not None:
                        rmse = calc_rmse
                        
                    if mae is None and calc_mae is not None:
                        mae = calc_mae

                    # ========================================================

                    records.append({
                        'Model': model_name,
                        'Minus Edges': minus_val,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MSE': mse,
                        'PCC': pcc
                    })
                    
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    
    return pd.DataFrame(records)

# ================= Modified plotting function =================
def plot_ablation_results(df):
    """Plot ablation experiment comparison (Times New Roman, legend on top, no subplot titles)"""
    if df.empty:
        print("No data, cannot plot")
        return

    # 1. Set global style and font (Times New Roman)
    sns.set_theme(style="whitegrid", font_scale=1.5, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False
    })
    
    # 2. Define metrics (Y-axis labels)
    metric_configs = [
        ('RMSE', 'RMSE'),
        ('MAE', 'MAE'),
        ('MSE', 'MSE'),
        ('PCC', 'PCC')
    ]
    
    # 3. Create 2x2 canvas (adjust figsize to be more compact)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Remove overall title (uncomment below if needed)
    # fig.suptitle('Ablation Study: Model Component Analysis', fontsize=22, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()
    num_models = df['Model'].nunique()

    for i, (col, ylabel) in enumerate(metric_configs):
        ax = axes_flat[i]
        
        if col in df.columns and df[col].notna().any():
            sns.lineplot(
                data=df,
                x='Minus Edges',
                y=col,
                hue='Model',       
                style='Model',     
                markers=True, 
                dashes=False,
                linewidth=3,
                markersize=10,
                ax=ax,
                legend='brief' if i == 0 else False # Only generate legend handles on first subplot
            )
            
            # --- Key modification: remove ax.set_title ---
            # Set axis labels and their font size
            ax.set_xlabel('Removed Edges', fontsize=30, family='serif')
            ax.set_ylabel(ylabel, fontsize=30, family='serif')
            
            # Set ticks
            ax.set_xticks(sorted(df['Minus Edges'].unique()))
            ax.tick_params(labelsize=30)
        else:
            ax.text(0.5, 0.5, f'No Data: {col}', ha='center', family='serif')

        # Remove subplot internal legends after plotting
        if ax.get_legend():
            ax.get_legend().remove()

    # 4. Extract legend handles (from the first subplot that had legend)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    
    # ================= Layout and legend fine-tuning =================
    
    # 5. Adjust layout: rect=[left, bottom, right, top]
    # top=0.90 reserves 10% top space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if handles:
        # 6. Legend on top
        # bbox_to_anchor=(0.5, 0.92) means legend bottom edge at 92% of figure height, close to subplots
        fig.legend(
            handles=handles, 
            labels=labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.91), 
            ncol=num_models,    # One row for all models
            frameon=True, 
            fontsize=30,
            #title="Model Variant",
            #title_fontsize=26,
            columnspacing=1.5,
            borderaxespad=0.
        )
    
    # 7. Save image (bbox_inches='tight' ensures legend not cut off)
    save_path = os.path.join(OUTPUT_DIR, 'ablation_fixed_style.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ablation experiment chart saved (Times New Roman/top legend): {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation experiment results.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example amazon_weight or Flickr.")
    parser.add_argument("--data-root", default="ABLATION_EXPERIMENT", help="Root directory of ablation experiment results.")
    parser.add_argument("--output-root", default="ablation_plots", help="Root directory for saving plots.")

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.dataset)
    OUTPUT_DIR = os.path.join(args.output_root, args.dataset)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_ablation_data(DATA_DIR)

    if not df.empty:
        df = df.sort_values(by=['Model', 'Minus Edges'])
        print("Data extracted successfully. Plotting...")
        plot_ablation_results(df)

        # Save the intermediate table for checking.
        df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_data_check.csv'), index=False)
    else:
        print("No data extracted.")