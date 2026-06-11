import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Mapping from folder name to internal JSON parameter name
PARAM_MAPPING = {
    'alpha': 'alpha',
    'heads': 'heads',           
    'hidden_dim': 'hidden_dim',
    'L': 'rw_L',               
    #'lr': 'learning_rate',     
    'n_walks': 'rw_n_walks'    
}

# ================= Data loading logic =================

def load_data(root_dir):
    records = []
    print(f"Scanning directory: {root_dir} ...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                parts = os.path.normpath(file_path).split(os.sep)
                
                category = None
                for part in parts:
                    if part in PARAM_MAPPING:
                        category = part
                        break
                
                if category is None:
                    continue 

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    meta = data.get('meta_data', {})
                    metrics = data.get('metrics', {})
                    params = meta.get('hyper_parameters', {})
                    
                    param_key = PARAM_MAPPING[category]
                    param_value = params.get(param_key, None)
                    
                    if param_value is not None:
                        records.append({
                            'category': category,       
                            'param_name': param_key,    
                            'param_value': param_value, 
                            'rmse': metrics.get('rmse'),
                            'mse': metrics.get('mse'),
                            'mae': metrics.get('mae'),
                            'pcc': metrics.get('pcc')
                        })
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    return pd.DataFrame(records)

# ================= Core plotting logic (1 figure, 2 subplots) =================

# ================= Modified core plotting logic (2x2 layout) =================

# ================= Modified core plotting logic (large font, no titles, streamlined labels) =================

def plot_dual_view(df):
    # Get all parameter categories
    categories = df['category'].unique()
    
    # [Keep high font size settings]
    sns.set_theme(style="whitegrid", font_scale=2.0, rc={
        "font.family": "serif", 
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False
    })
    
    # Change to process each parameter individually, each parameter generates a 1x2 figure
    for cat in categories:
        subset = df[df['category'] == cat].copy()
        if subset.empty: continue
        
        subset = subset.sort_values(by='param_value')
        param_real_name = subset['param_name'].iloc[0]
        
        # [Modification point] Create 1x2 canvas, adjust figsize to make it more horizontally rectangular
        fig, (ax_err, ax_pcc) = plt.subplots(1, 2, figsize=(22, 8))
        
        # ---------------- Left: Error Metrics ----------------
        error_data = subset[['param_value', 'rmse', 'mse', 'mae']].melt(
            id_vars='param_value', var_name='Metric', value_name='Error Value'
        )
        error_data['Metric'] = error_data['Metric'].str.upper()
        
        sns.lineplot(
            data=error_data, x='param_value', y='Error Value', hue='Metric',
            style='Metric', markers=True, dashes=False, linewidth=4,
            markersize=14, ax=ax_err, palette='tab10'
        )
        
        # Set labels, remove parentheses, use extra large font
        ax_err.set_ylabel('Error Value', fontsize=40)
        ax_err.set_xlabel(param_real_name, fontsize=40)
        ax_err.legend(title=None, fontsize=26)
        
        # ---------------- Right: Correlation ----------------
        sns.lineplot(
            data=subset, x='param_value', y='pcc', 
            color='#1f77b4', marker='o', markersize=14, linewidth=4, ax=ax_pcc
        )
        
        ax_pcc.set_ylabel('PCC', fontsize=40) 
        ax_pcc.set_xlabel(param_real_name, fontsize=40)

        # ---------------- Tick and detail settings ----------------
        ax_err.tick_params(axis='both', which='major', labelsize=40)
        ax_pcc.tick_params(axis='both', which='major', labelsize=40)

        if pd.api.types.is_numeric_dtype(subset['param_value']):
            unique_vals = subset['param_value'].unique()
            if len(unique_vals) <= 15:
                ax_err.set_xticks(unique_vals)
                ax_pcc.set_xticks(unique_vals)

        # Tight layout
        plt.tight_layout()
        
        # [Modification point] Save filename directly as parameter name, convenient for LaTeX referencing
        save_path = os.path.join(OUTPUT_DIR, f'sensitivity_{cat}_clean.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart for parameter {cat} saved: {save_path}")

def plot_2x2_grid(df):
    """
    Generate a 2x2 grid combination plot, containing sensitivity analysis for four parameters: alpha, hidden_dim, L, n_walks
    """
    # Specify the four parameters to plot and their order in the canvas
    target_params = ['alpha', 'hidden_dim', 'L', 'n_walks']
    
    # Keep large font and academic font settings
    sns.set_theme(style="whitegrid", font_scale=2.0, rc={
        "font.family": "serif", 
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False
    })
    
    # Create 2x2 canvas, adjust figsize to give each subplot enough display space
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten() # flatten for easy iteration
    
    for i, cat in enumerate(target_params):
        ax = axes[i]
        subset = df[df['category'] == cat].copy()
        
        if subset.empty:
            ax.set_title(f"No Data for {cat}")
            continue
            
        subset = subset.sort_values(by='param_value')
        param_real_name = subset['param_name'].iloc[0]
        
        # Reshape error metrics data
        error_data = subset[['param_value', 'rmse', 'mse', 'mae']].melt(
            id_vars='param_value', var_name='Metric', value_name='Error Value'
        )
        error_data['Metric'] = error_data['Metric'].str.upper()
        
        # Plot line chart
        sns.lineplot(
            data=error_data, x='param_value', y='Error Value', hue='Metric',
            style='Metric', markers=True, dashes=False, linewidth=4,
            markersize=14, ax=ax, palette='tab10'
        )
        
        # Set axis labels, use large font size
        ax.set_ylabel('Error Value', fontsize=36)
        ax.set_xlabel(param_real_name, fontsize=36)
        
        # Adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Optimize X-axis tick display (if not too many data points, force display all actual values)
        if pd.api.types.is_numeric_dtype(subset['param_value']):
            unique_vals = subset['param_value'].unique()
            if len(unique_vals) <= 15:
                ax.set_xticks(unique_vals)
        
        # Remove legend from each subplot, will add a unified legend on top
        if ax.get_legend():
            ax.get_legend().remove()

    # Extract legend handles and labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Adjust tight layout to reserve space for top global legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Add shared legend at top center of 2x2 canvas
    if handles:
        fig.legend(
            handles=handles, labels=labels, loc='upper center',
            bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=30,
            columnspacing=2.0
        )
        
    save_path = os.path.join(OUTPUT_DIR, 'sensitivity_2x2_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 2x2 grid combination chart saved: {save_path}")

def export_summary_table(df):
    """Save extracted experiment results as a CSV summary table"""
    csv_path = os.path.join(OUTPUT_DIR, 'experiment_summary.csv')
    # Define export column order
    cols = ['category', 'param_name', 'param_value', 'rmse', 'mse', 'mae', 'pcc']
    # Ensure only existing columns are selected, sort by category and param_value
    df_to_save = df[cols].sort_values(by=['category', 'param_value'])
    df_to_save.to_csv(csv_path, index=False)
    print(f"✅ Summary table saved to: {csv_path}")
# ================= Main program =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter sensitivity experiment results.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example amazon_weight or Flickr.")
    parser.add_argument("--root-dir", default="parameter_experiment", help="Root directory of parameter experiment results.")
    parser.add_argument("--output-root", default="analysis_results_dual", help="Output root directory for saving analysis results.")

    args = parser.parse_args()

    ROOT_DIR = os.path.join(args.root_dir, args.dataset)
    OUTPUT_DIR = os.path.join(args.output_root, args.dataset)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(ROOT_DIR):
        print(f"Error: directory not found: '{ROOT_DIR}'.")
    else:
        df = load_data(ROOT_DIR)

        if not df.empty:
            print(f"Successfully extracted {len(df)} experiment records.")
            export_summary_table(df)
            plot_dual_view(df)
            plot_2x2_grid(df)
            print(f"\nAll processing finished. Please check the folder: {OUTPUT_DIR}")

        else:
            print("No valid data extracted.")