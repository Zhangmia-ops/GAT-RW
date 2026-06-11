import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_baseline_data(root_dir):
    records = []
    print(f"Scanning baseline data: {root_dir} ...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    meta = data.get('meta_data', {})
                    metrics = data.get('metrics', {})
                    
                    # 1. Get algorithm name and dataset name
                    # Prefer reading from meta_data because filename might be non-standard
                    algo = meta.get('algorithm') or meta.get('method') # some have algorithm, some method
                    dataset = meta.get('dataset')
                    
                    # If not in json, try to parse from filename (e.g., AA_amazon_minus100.json)
                    if not algo or not dataset:
                        parts = file.split('_')
                        # Simple heuristic rule, adjust according to your actual filename
                        # Assume dataset always appears before 'minus', algo is first part
                        # This may need minor tuning based on actual filenames
                        if 'minus' in file:
                            dataset_candidate = [p for p in parts if p in ['amazon', 'dblp', 'Flickr', 'Fruit-Fly', 'amazon_weight']]
                            if dataset_candidate:
                                dataset = dataset_candidate[0]
                                algo = parts[0] # assume first part is algorithm name

                    # 2. Get Minus value
                    # Prefer from mask_cnt or mask_ratio, or filename
                    minus_val = meta.get('mask_cnt')
                    if minus_val is None:
                         if 'minus' in file:
                            import re
                            match = re.search(r'minus(\d+)', file)
                            if match:
                                minus_val = int(match.group(1))

                    if algo and dataset and minus_val is not None:
                        records.append({
                            'Algorithm': algo,
                            'Dataset': dataset,
                            'Minus Edges': int(minus_val),
                            'RMSE': metrics.get('rmse'),
                            'MAE': metrics.get('mae'),
                            'MSE': metrics.get('mse'),
                            'PCC': metrics.get('pcc')
                        })

                except Exception as e:
                    print(f"Error parsing {file}: {e}")
                    
    return pd.DataFrame(records)

# ================= Modified plotting function =================
# ================= Modified plotting function =================
def plot_single_dataset_comparison(df, target_dataset='amazon'):
    """Scenario 1: Performance comparison on a specific graph structure (dataset) (removed subplot titles, optimized spacing)"""
    subset = df[df['Dataset'].astype(str).str.contains(target_dataset, case=False)].copy()
    
    if subset.empty:
        print(f"No records found for dataset {target_dataset}")
        return

    name_map = {
    'AdamicAdar': 'AA',
    'CommonNeighbors': 'CN',
    'GAT_Spectral_RW_Correction': 'GAT-RW',
    'Node2Vec': 'Node2VecV',
    'GCN': 'GCN'
    }
    # Apply mapping
    subset['Algorithm'] = subset['Algorithm'].replace(name_map)

    # 1. Set font and global font size
    sns.set_theme(style="whitegrid", font_scale=1.8, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False
    })
    
    # 2. Adjust canvas
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Metric mapping
    metrics = [('RMSE', 'RMSE'), ('PCC', 'PCC')]
    
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(
            data=subset, 
            x='Minus Edges', 
            y=metric, 
            hue='Algorithm', 
            style='Algorithm', 
            markers=True, 
            dashes=False, 
            linewidth=3, 
            markersize=10, 
            ax=ax,
            legend='brief' 
        )
        
        # --- Key modification: remove subplot title ---
        # ax.set_title(...)  <-- this line deleted
        
        # 3. Enhance axis labels (since no title, Y-axis labels need to be clear enough)
        ax.set_xlabel('Minus Edges', fontsize=30, family='serif')
        ax.set_ylabel(ylabel, fontsize=30, family='serif')
        
        # Tick settings
        ax.set_xticks(sorted(subset['Minus Edges'].unique()))
        ax.tick_params(axis='both', which='major', labelsize=30) 
        
        if ax.get_legend():
            ax.get_legend().remove()

    # 4. Extract legend handles
    handles, labels = axes[0].get_legend_handles_labels()
    
    # ================= Layout optimization (compress whitespace) =================
    
    # 5. Raise subplot area upper limit (from 0.88 to 0.94, because titles removed, space can go upward)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if handles:
        # 6. Place legend at new upper position (bbox_to_anchor Y value = 0.93)
        fig.legend(
            handles=handles, 
            labels=labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.93), 
            ncol=len(labels), 
            frameon=True, 
            fontsize=24,
            #title="Algorithm",
            #title_fontsize=24,
            columnspacing=1.6,
            borderaxespad=0.
        )
    
    # 7. Save
    save_path = os.path.join(OUTPUT_DIR, f'baseline_{target_dataset}_no_subtitle.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimized chart saved: {save_path}")
    plt.show()

# def plot_average_performance(df):
#     """Scenario 2: Average performance across datasets"""
#     # Group by Algorithm and Minus Edges, compute average
#     avg_df = df.groupby(['Algorithm', 'Minus Edges'])[['RMSE', 'MAE', 'MSE', 'PCC']].mean().reset_index()
#     
#     sns.set_theme(style="whitegrid", font_scale=1.1)
#     
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     fig.suptitle('Average Performance Across All Datasets', fontsize=18, y=1.05)
#     
#     # RMSE Average
#     sns.lineplot(data=avg_df, x='Minus Edges', y='RMSE', hue='Algorithm', style='Algorithm', 
#                  markers=True, dashes=False, linewidth=2.5, markersize=9, ax=axes[0])
#     axes[0].set_title('Average RMSE')
#     axes[0].set_xticks(sorted(avg_df['Minus Edges'].unique()))
#     
#     # PCC Average
#     sns.lineplot(data=avg_df, x='Minus Edges', y='PCC', hue='Algorithm', style='Algorithm', 
#                  markers=True, dashes=False, linewidth=2.5, markersize=9, ax=axes[1])
#     axes[1].set_title('Average PCC')
#     axes[1].set_xticks(sorted(avg_df['Minus Edges'].unique()))
#     
#     plt.tight_layout()
#     save_path = os.path.join(OUTPUT_DIR, 'baseline_average_comparison.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Average performance comparison chart saved: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot baseline comparison for one dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example amazon_weight or Flickr.")
    parser.add_argument("--base-root", default="BASE_RESULT", help="Root directory of baseline results.")
    parser.add_argument("--output-root", default="baseline_plots", help="Root directory for saving plots.")

    args = parser.parse_args()

    BASE_DIR = os.path.join(args.base_root, args.dataset)
    OUTPUT_DIR = os.path.join(args.output_root, args.dataset)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_baseline_data(BASE_DIR)

    if not df.empty:
        # Normalize dataset names.
        df['Dataset'] = df['Dataset'].replace({'Fruit-FLy': 'Fruit-Fly', 'amazon_weight': 'amazon'})
        df['Dataset'] = df['Dataset'].str.lower()

        # Plot the target dataset.
        target_dataset = args.dataset.replace("_weight", "").lower()
        plot_single_dataset_comparison(df, target_dataset=target_dataset)

        # Save the summary table if needed.
        # df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_summary.csv'), index=False)
    else:
        print("No baseline data extracted.")