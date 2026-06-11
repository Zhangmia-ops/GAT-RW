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
                    algo = meta.get('algorithm') or meta.get('method')
                    dataset = meta.get('dataset')
                    if not algo or not dataset:
                        parts = file.replace('.json', '').split('_')
                        if 'minus' in file:
                            if 'amazon' in file.lower(): dataset = 'amazon'
                            elif 'dblp' in file.lower(): dataset = 'dblp'
                            elif 'flickr' in file.lower(): dataset = 'flickr'
                            elif 'fruit' in file.lower(): dataset = 'fruit-fly'
                            if not algo: algo = parts[0]
                    minus_val = meta.get('mask_cnt')
                    if minus_val is None and 'minus' in file:
                        import re
                        match = re.search(r'minus(\d+)', file)
                        if match: minus_val = int(match.group(1))
                    if algo and dataset and minus_val is not None:
                        records.append({
                            'Algorithm': algo, 'Dataset': dataset, 'Minus Edges': int(minus_val),
                            'RMSE': metrics.get('rmse'), 'MAE': metrics.get('mae'),
                            'MSE': metrics.get('mse'), 'PCC': metrics.get('pcc')
                        })
                except Exception as e: print(f"Error parsing {file}: {e}")
    return pd.DataFrame(records)

def clean_data(df):
    df['Dataset'] = df['Dataset'].str.lower()
    df['Dataset'] = df['Dataset'].replace('amazon_weight', 'amazon')
    df['Dataset'] = df['Dataset'].str.replace('fruit-fly', 'fruit-fly', case=False) 
    df['Dataset'] = df.apply(lambda x: 'fruit-fly' if 'fruit' in x['Dataset'] else x['Dataset'], axis=1)
    return df

# ================= Plotting function with legend display fix =================
# ================= Modified plotting function =================
def plot_global_average_performance(df):
    # Set theme
    sns.set_theme(style="whitegrid", font_scale=1.8, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False 
    })

    name_map = {
    'AdamicAdar': 'AA',
    'CommonNeighbors': 'CN',
    'GAT_Spectral_RW_Correction': 'GAT-RW',
    'Node2Vec': 'Node2VecV',
    'GCN': 'GCN'
    }
    # Apply mapping
    df['Algorithm'] = df['Algorithm'].replace(name_map)

    metrics_map = [
        ('RMSE', 'RMSE'),
        ('PCC', 'PCC')
    ]
    
    # Canvas size stays as before
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for i, (metric, ylabel) in enumerate(metrics_map):
        ax = axes[i]
        sns.lineplot(
            data=df, 
            x='Minus Edges', 
            y=metric, 
            hue='Algorithm', 
            style='Algorithm',
            markers=True, 
            dashes=False, 
            linewidth=3,
            markersize=10, 
            errorbar='sd',    
            err_kws={'alpha': 0.15},
            ax=ax,
            legend='brief' 
        )
        
        # --- Key modification: remove subplot titles ---
        # ax.set_title(...)  <-- this line deleted
        
        # Enhance Y-axis labels (since there is no title, Y-axis label is the only way to distinguish metrics)
        ax.set_xlabel('Minus Edges', fontsize=30)
        ax.set_ylabel(ylabel, fontsize=30) # slightly larger Y-axis font
        
        ax.set_xticks(sorted(df['Minus Edges'].unique()))
        ax.tick_params(axis='both', which='major', labelsize=30) 
        # Remove subplot internal legends
        if ax.get_legend():
            ax.get_legend().remove()

    # Extract legend handles
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()

    # ================= Spacing optimization adjustment =================
    
    # 1. Adjust subplot height range:
    # Since titles removed, we can make subplots taller. Raise top from 0.85 to 0.92
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    # 2. Redraw legend:
    # Raise bbox_to_anchor Y coordinate to 0.91 to place it close to subplot top, eliminating whitespace
    if handles:
        fig.legend(
            handles=handles, 
            labels=labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.91), 
            ncol=len(labels), 
            frameon=True, 
            fontsize=26,
            #title="Algorithm",
            #title_fontsize=24,
            columnspacing=2.0
        )

    save_path = os.path.join(OUTPUT_DIR, 'baseline_no_subtitle_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved (no subtitle clean version): {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot global average baseline performance.")
    parser.add_argument("--dataset", default="all", help="Output folder name, for example all or amazon_weight.")
    parser.add_argument("--base-root", default="BASE_RESULT", help="Root directory of baseline results.")
    parser.add_argument("--output-root", default="baseline_plots", help="Root directory for saving plots.")

    args = parser.parse_args()

    BASE_DIR = args.base_root
    OUTPUT_DIR = os.path.join(args.output_root, args.dataset)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_baseline_data(BASE_DIR)

    if not df.empty:
        df = clean_data(df)
        plot_global_average_performance(df)
    else:
        print("No data extracted.")