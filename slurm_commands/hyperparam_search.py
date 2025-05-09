#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import itertools
import subprocess
import json
from datetime import datetime

# Determine project directory dynamically
script_location = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_location)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Slurm scripts for hyperparameter search on unlearning methods')
    
    parser.add_argument('--output_dir', type=str, default="hyperparam_jobs",
                        help='Directory to save Slurm job scripts')
    parser.add_argument('--save_dir', type=str, default="checkpoints/hyperparam_search",
                        help='Directory to save the model checkpoints')
    parser.add_argument('--data_path', type=str, default="/u/kdkyum/ptmp_link/.torchvision",
                        help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the pre-trained model to start unlearning from (set to None for non-continual learning setup)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory to save the generated masks (for RL method)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the jobs to Slurm after creating the scripts')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing results to find best hyperparameters')
    
    return parser.parse_args()

def generate_learning_rates():
    return [0.1, 0.01, 0.001, 1e-4, 1e-5]

def generate_sparsity_values():
    """Generate 10 sparsity values between 0.001 and 0.1 on a log scale"""
    # Start with log-spaced values between 0.001 and 0.1
    return [0.2, 0.3, 0.4, 0.5]
    # return [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

def get_last_class(dataset):
    """Get the last class index for the dataset"""
    if dataset == "cifar10":
        return 9
    elif dataset == "cifar100":
        return [99]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def generate_slurm_script(args, dataset, model, method, lr, sparsity=None, layer_wise=True):
    """Generate a Slurm job script for a specific hyperparameter configuration"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{timestamp}_{dataset}_{model}_{method}_lr{lr:.6f}"
    if sparsity is not None:
        job_id += f"_sparsity{sparsity:.6f}"
    if "synaptag" in method:
        job_id += f"_{'layerwise' if layer_wise else 'nolayerwise'}"
    
    # Create directories
    output_dir = os.path.join(args.output_dir, method)
    os.makedirs(output_dir, exist_ok=True)
    method_dir = os.path.join(args.save_dir, model, dataset, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # For RL method, create the mask directory
    mask_dir = args.mask_dir
    if method == "SalUn" and mask_dir is None:
        mask_dir = os.path.join(args.save_dir, model, dataset, method, f"lr{lr:.6f}", "masks")
    if method == "SalUn":
        os.makedirs(mask_dir, exist_ok=True)
    
    script_path = os.path.join(output_dir, f"{job_id}.sh")
    
    # Determine the last class
    last_class = get_last_class(dataset)
    
    # Prepare the save directory for this specific run

    save_path_base = os.path.join(args.save_dir, model, dataset, method, f"lr{lr:.6f}")
    if sparsity is not None:
        save_path_base += f"_sparsity{sparsity:.6f}"
    if "synaptag" in method:
        save_path = f"{save_path_base}_{'layerwise' if layer_wise else 'nolayerwise'}"
    else:
        save_path = save_path_base
    os.makedirs(save_path, exist_ok=True)
    
    # Create the Slurm script
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash -l
# Standard output and error:
#SBATCH -o {output_dir}/{job_id}.out.%j
#SBATCH -e {output_dir}/{job_id}.err.%j
# Initial working directory:
#SBATCH -D {project_dir}  # Set Slurm working directory explicitly
# Job name
#SBATCH -J {job_id}
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --time=4:00:00

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate engram

export WANDB_MODE=offline
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

# Explicitly change to the project directory
echo "Changing directory to {project_dir}"
cd {project_dir} || exit 1  # Exit if cd fails

""")
        
        if method == "SalUn":
            if dataset == "cifar100":
                f.write(f"""# Generate mask for SalUn first
echo "Generating mask for {dataset} {model} class {last_class}"
python -u generate_mask.py \\
    --dataset {dataset} \\
    --data {args.data_path} \\
    --arch resnet{model.split('resnet')[1]} \\
    --class_to_replace {" ".join(map(str, last_class))} \\
    --save_dir {mask_dir} \\
    --num_classes 100
""")
            else:
                f.write(f"""# Generate mask for SalUn first
echo "Generating mask for {dataset} {model} class {last_class}"
python -u generate_mask.py \\
    --dataset {dataset} \\
    --data {args.data_path} \\
    --arch resnet{model.split('resnet')[1]} \\
    --class_to_replace {last_class} \\
    --save_dir {mask_dir} \\
""")
            f.write("\n")
            f.write(f"""# Run hyperparameter search
echo "Running unlearning with {method}, lr={lr}"
python -u main_random.py \\
    --dataset {dataset} \\
    --data {args.data_path} \\
    --arch resnet{model.split('resnet')[1]} \\
    --save_dir {save_path} \\""")
        
            if isinstance(last_class, list):
                f.write(f"""
    --class_to_replace {" ".join(map(str, last_class))} \\""")
            else:
                f.write(f"""
    --class_to_replace {last_class} \\""")
        
            f.write(f"""
    --unlearn {method} \\
    --unlearn_lr {lr} \\
    --unlearn_epochs 10 \\
""")
            if dataset in ["cifar100"]:
                num_classes = 100
                f.write(f"    --num_classes {num_classes} \\\n")

            mask_path = os.path.join(mask_dir, f"with_0.5.pt")
            f.write(f"    --mask_path {mask_path} \\\n")
        
        else: # Add the main unlearning command
            f.write(f"""# Run hyperparameter search
echo "Running unlearning with {method}, lr={lr}"
python -u main_forget.py \\
    --dataset {dataset} \\
    --data {args.data_path} \\
    --arch resnet{model.split('resnet')[1]} \\
    --save_dir {save_path} \\""")
        
            if isinstance(last_class, list):
                f.write(f"""
    --class_to_replace {" ".join(map(str, last_class))} \\""")
            else:
                f.write(f"""
    --class_to_replace {last_class} \\""")
        
            f.write(f"""
    --unlearn {method} \\
    --unlearn_lr {lr} \\
    --unlearn_epochs 10 \\
""")
            if layer_wise and "synaptag" in method:
                f.write(f"    --layer_wise \\\n")
        
            if dataset in ["cifar100"]:
                num_classes = 100
                f.write(f"    --num_classes {num_classes} \\\n")
        
            if "synaptag" in method and sparsity is not None:
                f.write(f"    --sparsity {sparsity} \\\n")
        
        # Add code to extract and save the best RFA
        f.write(f"""
# Extract the best RFA from the results
RESULTS_FILE="{save_path}/model_best.pth.tar"
if [ -f "$RESULTS_FILE" ]; then
    # Extract RFA using PyTorch to load the checkpoint
    BEST_RFA=$(python -c "import torch; data=torch.load('$RESULTS_FILE', map_location='cpu'); print(data.get('rfa', 'N/A'))")
    echo "Best RFA: $BEST_RFA"
    
    # Save the best RFA to a simple text file for easy parsing
    echo "$BEST_RFA" > "{save_path}/best_rfa.txt"
    
    # Also save the retain and forget accuracy for reference
    RETAIN_ACC=$(python -c "import torch; data=torch.load('$RESULTS_FILE', map_location='cpu'); print(data.get('retain_acc', 'N/A'))")
    FORGET_ACC=$(python -c "import torch; data=torch.load('$RESULTS_FILE', map_location='cpu'); print(data.get('forget_acc', 'N/A'))")
    echo "Retain Accuracy: $RETAIN_ACC, Forget Accuracy: $FORGET_ACC" >> "{save_path}/best_rfa.txt"
else
    echo "No results file found at $RESULTS_FILE"
fi

echo "Hyperparameter search completed"
""")
    
    print(f"Generated Slurm script: {script_path}")
    return script_path

def get_user_job_count():
    """Get the number of jobs the current user has in the Slurm queue"""
    try:
        # Get username
        import getpass
        username = getpass.getuser()
        
        # Run squeue command to count user's jobs
        result = subprocess.run(
            ["squeue", "-u", username, "-h"],
            capture_output=True, text=True
        )
        
        # Count lines in the output (each line is a job)
        job_count = len(result.stdout.strip().split('\n'))
        if job_count == 1 and result.stdout.strip() == '':
            job_count = 0
            
        return job_count
    except Exception as e:
        print(f"Error getting job count: {e}")
        return 0

def submit_jobs_with_limit(scripts, max_jobs=50, check_interval=60):
    """Submit jobs while respecting the maximum job limit"""
    if not scripts:
        print("No scripts to submit")
        return
        
    total_jobs = len(scripts)
    submitted = 0
    
    print(f"Starting to submit {total_jobs} jobs (maximum {max_jobs} at a time)")
    
    while submitted < total_jobs:
        # Check current job count
        current_job_count = get_user_job_count()
        available_slots = max_jobs - current_job_count
        
        if available_slots <= 0:
            print(f"Currently at job limit ({current_job_count}/{max_jobs}). Waiting {check_interval} seconds...")
            time.sleep(check_interval)
            continue
        
        # Submit as many jobs as we can
        jobs_to_submit = min(available_slots, total_jobs - submitted)
        
        for i in range(jobs_to_submit):
            script_path = scripts[submitted + i]
            print(f"Submitting job {submitted + i + 1}/{total_jobs}: {script_path}")
            subprocess.run(["sbatch", script_path])
            
        submitted += jobs_to_submit
        
        print(f"Progress: {submitted}/{total_jobs} jobs submitted")
        
        # Small delay between submissions to avoid overwhelming the scheduler
        if submitted < total_jobs:
            time.sleep(2)
    
    print(f"All {total_jobs} jobs have been submitted!")

def analyze_results(args):
    """Analyze existing results to find the best hyperparameters"""
    datasets = ["cifar10", "cifar100"]
    models = ["resnet18", "resnet50"]
    # methods = ["SalUn", "synaptag_NG"]
    methods = ["RL", "GA", "NG", "SalUn", "FT", "synaptag_RL", "synaptag_NG"]
    
    best_params = {}
    
    for dataset, model, method in itertools.product(datasets, models, methods):
        method_dir = f"{args.save_dir}/{model}/{dataset}/{method}"
        print(f"\nAnalyzing {dataset} {model} {method}")
        if not os.path.exists(method_dir):
            print(f"  Directory not found: {method_dir}")
            continue
            
        # Initialize best parameters for the current method
        best_rfa = -float('inf')
        best_lr = None
        best_sparsity = None
        
        # Specific tracking for synaptag layer-wise options
        best_rfa_lw = -float('inf')
        best_lr_lw = None
        best_sparsity_lw = None
        best_rfa_nlw = -float('inf')
        best_lr_nlw = None
        best_sparsity_nlw = None
        
        # Search through all result directories
        for result_dir in os.listdir(method_dir):
            full_result_path = os.path.join(method_dir, result_dir)
            if not os.path.isdir(full_result_path):
                continue

            rfa_file = os.path.join(full_result_path, "best_rfa.txt")
            if not os.path.exists(rfa_file):
                continue

            try:
                with open(rfa_file, 'r') as f:
                    content = f.readlines()
                    if not content:
                        continue
                    try:
                        rfa = float(content[0].strip())
                    except ValueError:
                        continue
            except Exception as e:
                print(f"  Error reading {rfa_file}: {e}")
                continue
                    
            # Extract hyperparameters from directory name
            try:
                parts = result_dir.split('_')
                lr = None
                sparsity = None
                current_layer_wise = None # Default for non-synaptag or if not specified

                # Find LR
                lr_part_found = False
                for part in parts:
                    if part.startswith("lr"):
                        lr_str = part[2:]
                        lr = float(lr_str)
                        lr_part_found = True
                        break
                if not lr_part_found: continue # Skip if LR not found

                # Find Sparsity (if applicable)
                for part in parts:
                    if part.startswith("sparsity"):
                        sparsity_str = part[8:]
                        sparsity = float(sparsity_str)
                        break

                # Determine layer_wise for synaptag
                if "synaptag" in method:
                    if "nolayerwise" in parts:
                        current_layer_wise = False
                    elif "layerwise" in parts:
                         current_layer_wise = True
                    # If neither is present, maybe default to True or skip? Assuming default True if not specified.
                    elif sparsity is not None: # Only consider if sparsity is also found
                         current_layer_wise = True # Default assumption if tag missing

            except (ValueError, IndexError) as e:
                print(f"  Could not parse hyperparameters from directory name: {result_dir} - {e}")
                continue
            
            # Update best parameters
            if "synaptag" in method:
                if current_layer_wise is True:
                    if rfa > best_rfa_lw:
                        best_rfa_lw = rfa
                        best_lr_lw = lr
                        best_sparsity_lw = sparsity
                elif current_layer_wise is False:
                    if rfa > best_rfa_nlw:
                        best_rfa_nlw = rfa
                        best_lr_nlw = lr
                        best_sparsity_nlw = sparsity
            else: # For methods other than synaptag
                if rfa > best_rfa:
                    best_rfa = rfa
                    best_lr = lr
                    best_sparsity = sparsity # Sparsity might be None, which is fine
        
        # Store results
        if "synaptag" in method:
            if best_lr_lw is not None:
                config_lw = { "best_rfa": best_rfa_lw, "best_lr": best_lr_lw }
                if best_sparsity_lw is not None: config_lw["best_sparsity"] = best_sparsity_lw
                config_lw["layer_wise"] = True
                best_params[(dataset, model, method + "_layerwise")] = config_lw
                print(f"  Best (LayerWise=True): RFA = {best_rfa_lw:.4f}, LR = {best_lr_lw:.6f}" +
                      (f", Sparsity = {best_sparsity_lw:.6f}" if best_sparsity_lw is not None else ""))
            else:
                 print(f"  No valid results found for LayerWise=True.")

            if best_lr_nlw is not None:
                config_nlw = { "best_rfa": best_rfa_nlw, "best_lr": best_lr_nlw }
                if best_sparsity_nlw is not None: config_nlw["best_sparsity"] = best_sparsity_nlw
                config_nlw["layer_wise"] = False
                best_params[(dataset, model, method + "_nolayerwise")] = config_nlw
                print(f"  Best (LayerWise=False): RFA = {best_rfa_nlw:.4f}, LR = {best_lr_nlw:.6f}" +
                      (f", Sparsity = {best_sparsity_nlw:.6f}" if best_sparsity_nlw is not None else ""))
            else:
                 print(f"  No valid results found for LayerWise=False.")

        else: # For other methods
            if best_lr is not None:
                config = { "best_rfa": best_rfa, "best_lr": best_lr }
                if best_sparsity is not None: config["best_sparsity"] = best_sparsity
                best_params[(dataset, model, method)] = config
                print(f"  Best RFA = {best_rfa:.4f}, LR = {best_lr:.6f}" +
                      (f", Sparsity = {best_sparsity:.6f}" if best_sparsity is not None else ""))
            else:
                print(f"  No valid results found.")

    # Save results to JSON file
    results_file = f"{args.save_dir}/best_hyperparameters.json"
    with open(results_file, 'w') as f:
        # Convert tuple keys to strings for JSON serialization
        serializable_best_params = {
            f"{dataset}_{model}_{method_key}": config  # method_key includes _layerwise/_nolayerwise for synaptag
            for (dataset, model, method_key), config in best_params.items()
        }
        json.dump(serializable_best_params, f, indent=2, sort_keys=True)
    
    print(f"\nBest hyperparameters saved to {results_file}")

def main():
    args = parse_args()
    
    if args.analyze:
        analyze_results(args)
        return
    
    # Define the search space
    datasets = ["cifar100"]
    models = ["resnet18", "resnet50"]
    # methods = ["synaptag_RL", "synaptag_NG"]
    methods = ["RL", "GA", "NG", "SalUn", "FT", "synaptag_RL", "synaptag_NG"]
    learning_rates = generate_learning_rates()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate all job scripts
    all_scripts = []
    
    # For methods other than synaptag
    for dataset, model, method in itertools.product(datasets, models, methods):
        if "synaptag" not in method:
            for lr in learning_rates:
                script_path = generate_slurm_script(args, dataset, model, method, lr, layer_wise=True)
                all_scripts.append(script_path)
        else:
            # For synaptag, search sparsity and layer_wise options
            sparsity_values = generate_sparsity_values()
            for lr, sparsity, layer_wise_flag in itertools.product(learning_rates, sparsity_values, [True]):
                script_path = generate_slurm_script(args, dataset, model, method, lr, sparsity, layer_wise=layer_wise_flag)
                all_scripts.append(script_path)
    
    print(f"Generated {len(all_scripts)} Slurm job scripts")
    
    # Submit jobs if requested
    if args.submit:
        submit_jobs_with_limit(all_scripts, max_jobs=32, check_interval=30)
    else:
        print("To submit all jobs, run with --submit flag")
    
    print("After all jobs complete, run with --analyze flag to find the best hyperparameters")

if __name__ == "__main__":
    main()
