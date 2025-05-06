#!/usr/bin/env python3

"""
Script to generate Slurm jobs for continual unlearning experiments using various methods.

This script supports multiple unlearning methods:
- RL (SalUn): Requires saliency map generation before unlearning
- GA, NG, FT: Direct unlearning methods
- boundary_expanding, boundary_shrink: Boundary-based unlearning methods
- synaptag: Synapse-based unlearning with layer-wise option

And multiple datasets:
- CIFAR-10: Unlearning classes incrementally from 0 to 7
- CIFAR-100: Unlearning classes in groups of 5, from 0-4 to 0-95
"""

import os
import argparse
import subprocess
import time
import json
import sys

# Add the parent directory to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Slurm script for continual unlearning experiments')
    
    parser.add_argument('--method', type=str, choices=['RL', 'GA', 'NG', 'FT', 'boundary_expanding', 'boundary_shrink', 'synaptag', 'all'],
                        default='all', help='Unlearning method to use (default: all)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'all'], 
                        default='all', help='Dataset to use (default: all)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the pre-trained model to start unlearning from')
    parser.add_argument('--data_path', type=str, default="/u/kdkyum/ptmp_link/.torchvision",
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="slurm_logs",
                        help='Directory to save Slurm output logs')
    parser.add_argument('--gpu_count', type=int, default=1, choices=[1, 2, 4],
                        help='Number of GPUs to use (1, 2, or 4)')
    parser.add_argument('--time_limit', type=str, default="11:59:00",
                        help='Time limit for the Slurm job (HH:MM:SS)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the jobs to Slurm after creating the scripts')
    parser.add_argument('--hp_file', type=str, default="checkpoints/hyperparam_search/best_hyperparameters.json",
                        help='Path to the JSON file with best hyperparameters')
    
    # For CIFAR-10
    parser.add_argument('--max_classes_cifar10', type=int, default=8,
                        help='Maximum number of classes to unlearn for CIFAR-10 (0 to max_classes-1)')
    
    # For CIFAR-100
    parser.add_argument('--max_class_cifar100', type=int, default=95,
                        help='Maximum class to unlearn for CIFAR-100 (up to this class in increments of 5)')
    parser.add_argument('--increment_cifar100', type=int, default=5,
                        help='Increment size for classes to unlearn at each step for CIFAR-100')

    # For RL (SalUn) method
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for the saliency mask (for RL/SalUn method)')
    return parser.parse_args()


def load_hyperparameters(hp_file):
    """Load the best hyperparameters from the JSON file"""
    try:
        with open(hp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading hyperparameters from {hp_file}: {e}")
        sys.exit(1)


def get_hyperparams(hyperparams, dataset, model, method):
    """Get the hyperparameters for a specific dataset, model, and method"""
    key = f"{dataset}_{model}_{method}"
    if method == 'synaptag':
        key += "_layerwise"  # We're only using layer_wise=True
        
    if key not in hyperparams:
        print(f"Warning: No hyperparameters found for {key}. Using default values.")
        return {
            'best_lr': 0.01,
            'best_sparsity': 0.05 if method == 'synaptag' else None
        }
    
    return hyperparams[key]


def generate_slurm_script(args, dataset, model, method, hp):
    """Generate the Slurm script for a specific dataset, model, and method"""
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine resource allocation based on GPU count
    if args.gpu_count == 1:
        gpu_line = "#SBATCH --gres=gpu:a100:1"
        cpus_per_task = 18
        mem = 125000
    elif args.gpu_count == 2:
        gpu_line = "#SBATCH --gres=gpu:a100:2"
        cpus_per_task = 36
        mem = 250000
    else:  # 4 GPUs
        gpu_line = "#SBATCH --gres=gpu:a100:4"
        cpus_per_task = 72
        mem = 500000
    
    # Set parameters based on dataset
    if dataset == 'cifar10':
        max_classes = args.max_classes_cifar10
        dataset_specific_params = f"MAX_CLASSES={max_classes}"
        num_classes_arg = ""
        # For CIFAR-10, we increment by 1
        increment = 1
    else:  # cifar100
        max_class = args.max_class_cifar100
        increment = args.increment_cifar100
        dataset_specific_params = f"MAX_CLASS={max_class}\nINCREMENT={increment}"
        num_classes_arg = "--num_classes 100"
    
    # Set method-specific parameters
    unlearn_lr = hp.get('best_lr', 0.01)
    method_specific_params = ""
    
    if method == 'synaptag':
        sparsity = hp.get('best_sparsity', 0.05)
        method_specific_params = f"SPARSITY={sparsity}"
    elif method == 'RL':  # SalUn method
        method_specific_params = f"THRESHOLD={args.threshold}"
    
    # Determine save directory and method command
    save_dir = f"checkpoints/{method}_continual_unlearn/{dataset}"
    mask_dir = f"{save_dir}/masks" if method == 'RL' else None
    
    # Create a job name
    job_name = f"{dataset}_{method}_continual_unlearn"
    
    # Create a unique script name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    script_path = f"{args.output_dir}/{job_name}_{timestamp}.sh"
    
    # Generate the Slurm script content - start with header
    slurm_script = f"""#!/bin/bash -l
# Standard output and error:
#SBATCH -o {args.output_dir}/{job_name}_%j.out
#SBATCH -e {args.output_dir}/{job_name}_%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J {job_name}
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# GPU configuration based on selection
{gpu_line}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#
#SBATCH --mail-type=none
#SBATCH --time={args.time_limit}

# Load required modules
module purge
module load cuda/12.6
module load python-waterboa/2024.06

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate engram

# Set environment variables
export WANDB_MODE=offline
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

# Define base parameters
BASE_SAVE_DIR="{save_dir}"
UNLEARN_LR={unlearn_lr}
UNLEARN_EPOCHS=10
DATA_PATH="{args.data_path}"
{dataset_specific_params}
{method_specific_params}

# Create the output directory
mkdir -p ${{BASE_SAVE_DIR}}
"""
    
    # Add mask directory creation for RL/SalUn method
    if method == 'RL':
        slurm_script += f"""
# Create mask directory for SalUn method
MASK_DIR="{mask_dir}"
mkdir -p ${{MASK_DIR}}
"""
    
    # Add model path handling
    if args.model_path:
        slurm_script += f"""
# Use the provided pre-trained model as starting point
CURRENT_MODEL_PATH="{args.model_path}"
echo "Starting with pre-trained model: ${{CURRENT_MODEL_PATH}}"
"""
    else:
        slurm_script += """
# No pre-trained model provided, will use randomly initialized model
CURRENT_MODEL_PATH=""
echo "No pre-trained model provided, will start with random initialization"
"""

    # Add the unlearning loop based on dataset
    if dataset == 'cifar10':
        slurm_script += """
# Cumulative unlearning loop
for MAX_CLASS_ID in $(seq 0 $((MAX_CLASSES-1))); do
    echo "=========================================================="
    echo "Starting cumulative unlearning step for classes 0-${MAX_CLASS_ID}"
    
    # Define the save directory for this step
    STEP_DIR="${BASE_SAVE_DIR}/0-${MAX_CLASS_ID}"
    mkdir -p ${STEP_DIR}
    
    # Build the cumulative class list (0 to MAX_CLASS_ID)
    CLASS_LIST=""
    for CLASS_ID in $(seq 0 ${MAX_CLASS_ID}); do
        CLASS_LIST="${CLASS_LIST} ${CLASS_ID}"
    done
    CLASS_LIST=$(echo ${CLASS_LIST} | xargs)  # Trim leading/trailing spaces
    
    # Extract the last new class for mask generation (for SalUn/RL method)
    CLASS_TO_REPLACE=${MAX_CLASS_ID}
    
    echo "Unlearning classes: ${CLASS_LIST}"
"""
    else:  # cifar100
        slurm_script += """
# Cumulative unlearning loop with increments of $INCREMENT
for END_CLASS in $(seq $INCREMENT $INCREMENT $MAX_CLASS); do
    START_CLASS=0
    PREV_END=$((END_CLASS - INCREMENT))
    
    echo "=========================================================="
    echo "Starting cumulative unlearning step for classes $START_CLASS-$((END_CLASS-1))"
    
    # Define the save directory for this step
    STEP_DIR="${BASE_SAVE_DIR}/0-$((END_CLASS-1))"
    mkdir -p ${STEP_DIR}
    
    # Build the cumulative class list (0 to END_CLASS-1)
    CLASS_LIST=""
    for CLASS_ID in $(seq $START_CLASS $((END_CLASS-1))); do
        CLASS_LIST="${CLASS_LIST} ${CLASS_ID}"
    done
    CLASS_LIST=$(echo ${CLASS_LIST} | xargs)  # Trim leading/trailing spaces
    
    # Build the list of new classes in this increment
    NEW_CLASSES=""
    for CLASS_ID in $(seq $((PREV_END)) $((END_CLASS-1))); do
        NEW_CLASSES="${NEW_CLASSES} ${CLASS_ID}"
    done
    NEW_CLASSES=$(echo ${NEW_CLASSES} | xargs)  # Trim leading/trailing spaces
    
    echo "Unlearning classes: ${CLASS_LIST}"
    echo "New classes in this increment: ${NEW_CLASSES}"
"""
    
    # For SalUn/RL method, add mask generation step
    if method == 'RL':
        if dataset == 'cifar10':
            slurm_script += """
    # Step 1: Generate the saliency map for SalUn/RL method
    MASK_SUBDIR="${MASK_DIR}/0-${MAX_CLASS_ID}"
    mkdir -p ${MASK_SUBDIR}
    MASK_PATH="${MASK_SUBDIR}/with_${THRESHOLD}.pt"
    
    echo "Generating saliency map for class: ${NEW_CLASSES}"
    MASK_COMMAND="python generate_mask.py --save_dir ${MASK_SUBDIR} \\
        --class_to_replace ${NEW_CLASSES} --unlearn_epochs 1 --data ${DATA_PATH}"
    
    # Add model path if we have a previous model
    if [ ! -z "${CURRENT_MODEL_PATH}" ]; then
        MASK_COMMAND="${MASK_COMMAND} --model_path ${CURRENT_MODEL_PATH}"
    fi
    
    echo "Running command: ${MASK_COMMAND}"
    eval "${MASK_COMMAND}"
    
    # Save exit status
    EXIT_STATUS=$?
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
        echo "Error: Saliency map generation for classes 0-${MAX_CLASS_ID} failed with exit status ${EXIT_STATUS}"
        exit ${EXIT_STATUS}
    fi
"""
        else:  # cifar100
            slurm_script += """
    # Step 1: Generate the saliency map for SalUn/RL method
    MASK_SUBDIR="${MASK_DIR}/0-$((END_CLASS-1))"
    mkdir -p ${MASK_SUBDIR}
    MASK_PATH="${MASK_SUBDIR}/with_${THRESHOLD}.pt"
    
    echo "Generating saliency map using class: ${NEW_CLASSES}"
    
    MASK_COMMAND="python generate_mask.py --save_dir ${MASK_SUBDIR} \\
        --class_to_replace ${NEW_CLASSES} --unlearn_epochs 1 \\
        --dataset cifar100 --num_classes 100 --data ${DATA_PATH}"
    
    # Add model path if we have a previous model
    if [ ! -z "${CURRENT_MODEL_PATH}" ]; then
        MASK_COMMAND="${MASK_COMMAND} --model_path ${CURRENT_MODEL_PATH}"
    fi
    
    echo "Running command: ${MASK_COMMAND}"
    eval "${MASK_COMMAND}"
    
    # Save exit status
    EXIT_STATUS=$?
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
        echo "Error: Saliency map generation for classes 0-$((END_CLASS-1)) failed with exit status ${EXIT_STATUS}"
        exit ${EXIT_STATUS}
    fi
"""
    
    # Add the appropriate unlearning command based on method
    if method == 'RL':  # SalUn method uses main_random.py
        if dataset == 'cifar10':
            slurm_script += """
    # Step 2: Perform SalUn unlearning with main_random.py
    echo "Performing SalUn unlearning for classes: ${CLASS_LIST} with mask: ${MASK_PATH}"
    UNLEARN_COMMAND="python main_random.py --save_dir ${STEP_DIR} \\
        --unlearn RL --class_to_replace ${CLASS_LIST} \\
        --unlearn_epochs ${UNLEARN_EPOCHS} --unlearn_lr ${UNLEARN_LR} \\
        --mask_path ${MASK_PATH} --data ${DATA_PATH}"
"""
        else:  # cifar100
            slurm_script += """
    # Step 2: Perform SalUn unlearning with main_random.py
    echo "Performing SalUn unlearning for classes: ${CLASS_LIST} with mask: ${MASK_PATH}"
    UNLEARN_COMMAND="python main_random.py --save_dir ${STEP_DIR} \\
        --unlearn RL --class_to_replace ${CLASS_LIST} \\
        --unlearn_epochs ${UNLEARN_EPOCHS} --unlearn_lr ${UNLEARN_LR} \\
        --mask_path ${MASK_PATH} --dataset cifar100 --num_classes 100 --data ${DATA_PATH}"
"""
    else:  # Other methods use main_forget.py
        method_args = ""
        if method == 'synaptag':
            method_args = "--sparsity ${SPARSITY} --layer_wise"
        
        if dataset == 'cifar10':
            slurm_script += f"""
    # Perform {method} unlearning using main_forget.py
    echo "Performing {method} unlearning for classes: ${{CLASS_LIST}}"
    UNLEARN_COMMAND="python main_forget.py --save_dir ${{STEP_DIR}} \\
        --unlearn {method} --class_to_replace ${{CLASS_LIST}} \\
        --unlearn_epochs ${{UNLEARN_EPOCHS}} --unlearn_lr ${{UNLEARN_LR}} \\
        {method_args} --data ${{DATA_PATH}}"
"""
        else:  # cifar100
            slurm_script += f"""
    # Perform {method} unlearning using main_forget.py
    echo "Performing {method} unlearning for classes: ${{CLASS_LIST}}"
    UNLEARN_COMMAND="python main_forget.py --save_dir ${{STEP_DIR}} \\
        --unlearn {method} --class_to_replace ${{CLASS_LIST}} \\
        --unlearn_epochs ${{UNLEARN_EPOCHS}} --unlearn_lr ${{UNLEARN_LR}} \\
        {method_args} --dataset cifar100 --num_classes 100 --data ${{DATA_PATH}}"
"""
    
    # Add model path to the unlearning command if we have a previous model
    slurm_script += """
    # Add model path if we have a previous model
    if [ ! -z "${CURRENT_MODEL_PATH}" ]; then
        UNLEARN_COMMAND="${UNLEARN_COMMAND} --model_path ${CURRENT_MODEL_PATH}"
    fi
    
    echo "Running command: ${UNLEARN_COMMAND}"
    eval "${UNLEARN_COMMAND}"
    
    # Save exit status
    EXIT_STATUS=$?
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
"""
    
    # Error message based on dataset
    if dataset == 'cifar10':
        slurm_script += """
        echo "Error: Unlearning for classes 0-${MAX_CLASS_ID} failed with exit status ${EXIT_STATUS}"
        exit ${EXIT_STATUS}
    fi
    
    # Update the model path for the next iteration
    CURRENT_MODEL_PATH="${STEP_DIR}/model_best.pth.tar"
    echo "Updated model path for next step: ${CURRENT_MODEL_PATH}"
    
    # Validate that the model file exists
    if [ ! -f "${CURRENT_MODEL_PATH}" ]; then
        echo "Error: Expected model file not found at ${CURRENT_MODEL_PATH}"
        exit 1
    fi
    
    echo "Successfully unlearned classes 0-${MAX_CLASS_ID}"
    echo "=========================================================="
done

echo "Cumulative unlearning completed for all classes 0-$((MAX_CLASSES-1))!"
"""
    else:  # cifar100
        slurm_script += """
        echo "Error: Unlearning for classes 0-$((END_CLASS-1)) failed with exit status ${EXIT_STATUS}"
        exit ${EXIT_STATUS}
    fi
    
    # Update the model path for the next iteration
    CURRENT_MODEL_PATH="${STEP_DIR}/model_best.pth.tar"
    echo "Updated model path for next step: ${CURRENT_MODEL_PATH}"
    
    # Validate that the model file exists
    if [ ! -f "${CURRENT_MODEL_PATH}" ]; then
        echo "Error: Expected model file not found at ${CURRENT_MODEL_PATH}"
        exit 1
    fi
    
    echo "Successfully unlearned classes 0-$((END_CLASS-1))"
    echo "=========================================================="
done

echo "Cumulative unlearning completed for all CIFAR-100 classes up to $((MAX_CLASS-1))!"
"""
    
    # Write the script to file
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    args = parse_args()
    
    # Load hyperparameters
    hyperparams = load_hyperparameters(args.hp_file)
    
    # Define datasets and methods based on user selection
    datasets = ['cifar10', 'cifar100'] if args.dataset == 'all' else [args.dataset]
    model = 'resnet18'  # Only using resnet18 as specified
    methods = ['RL', 'GA', 'NG', 'FT', 'synaptag'] if args.method == 'all' else [args.method]
    
    # Track all generated scripts
    generated_scripts = []
    
    # Generate scripts for each dataset and method combination
    for dataset in datasets:
        for method in methods:
            # Get hyperparameters for this combination
            hp = get_hyperparams(hyperparams, dataset, model, method)
            
            print(f"Generating script for {dataset} with {method} method...")
            script_path = generate_slurm_script(args, dataset, model, method, hp)
            generated_scripts.append(script_path)
            print(f"Generated script: {script_path}")
    
    # Submit jobs if requested
    if args.submit and generated_scripts:
        print(f"Submitting {len(generated_scripts)} jobs to Slurm...")
        for script_path in generated_scripts:
            print(f"Submitting: {script_path}")
            subprocess.run(['sbatch', script_path])
            # Small delay to avoid overwhelming the scheduler
            time.sleep(1)
        print("All jobs submitted!")
    elif generated_scripts:
        print("\nTo submit the jobs, run the following commands:")
        for script_path in generated_scripts:
            print(f"sbatch {script_path}")


if __name__ == "__main__":
    main()
