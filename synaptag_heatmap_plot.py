import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def parse_dir_name(dir_name):
    """Extracts learning rate, sparsity, and layer_wise flag from directory name."""
    # Match patterns like lr0.1_sparsity0.01_layerwise or lr0.1_sparsity0.01_nolayerwise
    match = re.match(r'lr([\d.eE-]+)_sparsity([\d.eE-]+)(?:_(layerwise|nolayerwise))?', dir_name)
    if match:
        lr = float(match.group(1))
        sparsity = float(match.group(2))
        layer_wise_str = match.group(3)
        
        # Determine layer_wise flag
        if layer_wise_str == "nolayerwise":
            layer_wise = False
        else:
            # Default to True if the tag is 'layerwise' or missing (original format)
            layer_wise = True 
            
        return lr, sparsity, layer_wise
    return None, None, None

def read_rfa(file_path):
    """Reads the RFA value from the first line of the file."""
    try:
        with open(file_path, 'r') as f:
            line = f.readline()
            return float(line.strip())
    except FileNotFoundError:
        print(f"Warning: RFA file not found: {file_path}")
        return np.nan
    except ValueError:
        print(f"Warning: Could not parse RFA value in: {file_path}")
        return np.nan
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return np.nan

def collect_data(base_dir):
    """Collects LR, sparsity, layer_wise flag, and RFA data from subdirectories."""
    data = []
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return pd.DataFrame(data)

    for dir_name in os.listdir(base_dir):
        lr, sparsity, layer_wise = parse_dir_name(dir_name)
        if lr is not None and sparsity is not None and layer_wise is not None:
            rfa_file = os.path.join(base_dir, dir_name, 'best_rfa.txt')
            rfa = read_rfa(rfa_file)
            if not np.isnan(rfa):
                data.append({'learning_rate': lr, 'sparsity': sparsity, 'layer_wise': layer_wise, 'rfa': rfa})

    return pd.DataFrame(data)

def plot_heatmap(df, title, output_path):
    """Generates and saves a heatmap from the DataFrame."""
    if df.empty:
        print(f"No data to plot for {title}")
        return

    try:
        # Pivot the data for heatmap format
        heatmap_data = df.pivot_table(index='learning_rate', columns='sparsity', values='rfa')
        heatmap_data = heatmap_data.sort_index(ascending=False) # Sort LR descending
        heatmap_data = heatmap_data.sort_index(axis=1) # Sort sparsity ascending

        plt.figure(figsize=(12, 10))
        # Use LogNorm for better color differentiation if values span orders of magnitude
        # norm = LogNorm(vmin=df['rfa'].min(), vmax=df['rfa'].max()) if df['rfa'].min() > 0 else None
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", # norm=norm,
                    cbar_kws={'label': 'RFA (Retain-Forget Accuracy)'})

        plt.title(title)
        plt.xlabel("Sparsity")
        plt.ylabel("Learning Rate")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Heatmap saved to {output_path}")

    except Exception as e:
        print(f"Error generating heatmap for {title}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Define base directories relative to the script location or use absolute paths
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust base_search_dir if your checkpoints are elsewhere
    base_search_dir = os.path.join("checkpoints", "hyperparam_search")
    output_plot_dir = os.path.join("plots")
    os.makedirs(output_plot_dir, exist_ok=True)

    datasets = ["cifar10", "cifar100"] # Add "tinyimagenet" if needed
    model = "resnet18"
    method = "synaptag_NG"

    for dataset in datasets:
        dataset_dir = os.path.join(base_search_dir, dataset, model, method)
        print(f"\nProcessing {dataset} data from: {dataset_dir}")
        
        all_data = collect_data(dataset_dir)
        if all_data.empty:
            print(f"No data found for {dataset}.")
            continue

        # Separate data based on layer_wise flag
        data_lw = all_data[all_data['layer_wise'] == True]
        data_nlw = all_data[all_data['layer_wise'] == False]

        # Plot heatmap for LayerWise = True
        if not data_lw.empty:
            plot_heatmap(data_lw,
                         f"Synaptag Hyperparameter Search ({dataset.upper()} / {model.capitalize()}) - LayerWise=True\nRFA vs. Learning Rate and Sparsity",
                         os.path.join(output_plot_dir, f"{dataset}_{method}_layerwise_heatmap.png"))
        else:
            print(f"No data found for {dataset} with LayerWise=True.")

        # Plot heatmap for LayerWise = False
        if not data_nlw.empty:
            plot_heatmap(data_nlw,
                         f"Synaptag Hyperparameter Search ({dataset.upper()} / {model.capitalize()}) - LayerWise=False\nRFA vs. Learning Rate and Sparsity",
                         os.path.join(output_plot_dir, f"{dataset}_{method}_nolayerwise_heatmap.png"))
        else:
            print(f"No data found for {dataset} with LayerWise=False.")


    print("\nScript finished.")
