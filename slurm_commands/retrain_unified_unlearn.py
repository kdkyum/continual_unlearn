#!/usr/bin/env python3

"""
Unified script to generate Slurm jobs for retraining ResNet-18 or ResNet-50 on CIFAR-10 or CIFAR-100
for unlearning specific classes.

For CIFAR-10: Unlearn classes 0-7 (classes 0, 0-1, 0-2, ..., 0-7)
For CIFAR-100: Unlearn classes in increments of 5 (0-4, 0-9, 0-14, ..., 0-94)

The script generates a Slurm array job that runs multiple retrain jobs in parallel.
"""

import os
import argparse
import subprocess
import time
import sys

# Add the parent directory to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Slurm script for retraining unlearning')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                        help='Dataset to use (cifar10 or cifar100)')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                        help='Model architecture to use (resnet18 or resnet50)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the retrained models (defaults based on architecture and dataset)')
    parser.add_argument('--data_path', type=str, default="/u/kdkyum/ptmp_link/.torchvision",
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="slurm_logs",
                        help='Directory to save Slurm output logs')
    parser.add_argument('--gpu_count', type=int, default=1, choices=[1, 2, 4],
                        help='Number of GPUs to use (1, 2, or 4)')
    parser.add_argument('--time_limit', type=str, default="11:59:00",
                        help='Time limit for the Slurm job (HH:MM:SS)')
    parser.add_argument('--job_name', type=str, default=None,
                        help='Name for the Slurm job (defaults based on architecture and dataset)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the job to Slurm after creating the script')
    args = parser.parse_args()
    
    # Set default save_dir based on architecture and dataset
    if args.save_dir is None:
        args.save_dir = f"checkpoints/retrain_continual_unlearn/{args.arch}/{args.dataset}"
    
    # Set default job_name based on architecture and dataset
    if args.job_name is None:
        args.job_name = f"{args.dataset}_{args.arch}_retrain"
    
    return args


def generate_slurm_script(args):
    """Generate the Slurm script based on provided arguments"""
    
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
    
    # Set dataset-specific configuration
    if args.dataset == 'cifar10':
        array_range = "0-7"
        class_selection_code = """# Set the target classes to unlearn based on array task ID
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    CLASS_IDS="0"
    SAVE_DIR="${BASE_SAVE_DIR}/0-0"
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    CLASS_IDS="0 1"
    SAVE_DIR="${BASE_SAVE_DIR}/0-1"
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    CLASS_IDS="0 1 2"
    SAVE_DIR="${BASE_SAVE_DIR}/0-2"
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
    CLASS_IDS="0 1 2 3"
    SAVE_DIR="${BASE_SAVE_DIR}/0-3"
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then
    CLASS_IDS="0 1 2 3 4"
    SAVE_DIR="${BASE_SAVE_DIR}/0-4"
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then
    CLASS_IDS="0 1 2 3 4 5"
    SAVE_DIR="${BASE_SAVE_DIR}/0-5"
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
    CLASS_IDS="0 1 2 3 4 5 6"
    SAVE_DIR="${BASE_SAVE_DIR}/0-6"
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
    CLASS_IDS="0 1 2 3 4 5 6 7"
    SAVE_DIR="${BASE_SAVE_DIR}/0-7"
fi"""
        dataset_params = "--dataset cifar10 --num_classes 10 \\"
        desc = "Starting retraining unlearning for classes ${CLASS_IDS}..."
    else:  # cifar100
        array_range = "0-18"
        class_selection_code = """# Calculate class IDs and save directory based on array task ID
INCREMENT=5
START_CLASS=0
END_CLASS=$((START_CLASS + (SLURM_ARRAY_TASK_ID + 1) * INCREMENT - 1))
CLASS_IDS=$(seq -s " " $START_CLASS $END_CLASS)
SAVE_DIR=${BASE_SAVE_DIR}/0-${END_CLASS}

"""
        dataset_params = "--dataset cifar100 --num_classes 100 \\"
        desc = "Starting CIFAR-100 retraining unlearning for classes ${CLASS_IDS}..."
    
    # Create a unique script name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    script_path = f"{args.output_dir}/{args.dataset}_{args.arch}_retrain_unlearn_{timestamp}.sh"
    
    # Generate the Slurm script content
    slurm_script = f"""#!/bin/bash -l
# Standard output and error:
#SBATCH -o {args.output_dir}/{args.job_name}_%A_%a.out
#SBATCH -e {args.output_dir}/{args.job_name}_%A_%a.err
# Initial working directory:
#SBATCH -D /raven/ptmp/kdkyum/workdir/continual_unlearn
# Job name
#SBATCH -J {args.job_name}
#
# Run as array job
#SBATCH --array={array_range}
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
BASE_SAVE_DIR="{args.save_dir}"
EPOCHS=182
LR=0.1
DATA_PATH="{args.data_path}"
ARCH="{args.arch}"

{class_selection_code}

echo "{desc}"
echo "Architecture: ${{ARCH}}"
echo "Saving model to ${{SAVE_DIR}}"

python main_forget.py --save_dir ${{SAVE_DIR}} \\
    --arch ${{ARCH}} \\
    {dataset_params}
    --unlearn retrain --class_to_replace ${{CLASS_IDS}} \\
    --unlearn_epochs ${{EPOCHS}} --unlearn_lr ${{LR}} \\
    --data ${{DATA_PATH}}

echo "Retrain unlearning complete for architecture ${{ARCH}}, classes ${{CLASS_IDS}}!"
"""
    
    # Write the script to file
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    args = parse_args()
    
    dataset_desc = "CIFAR-10" if args.dataset == 'cifar10' else "CIFAR-100"
    if args.dataset == 'cifar10':
        classes_desc = "0 through 0-7"
    else:
        classes_desc = "in increments of 5 (0-4, 0-9, ..., 0-94)"
    
    print(f"Generating Slurm array job for {args.arch} on {dataset_desc} to unlearn classes {classes_desc}...")
    script_path = generate_slurm_script(args)
    print(f"Slurm script generated at: {script_path}")
    
    if args.submit:
        print(f"Submitting job to Slurm...")
        subprocess.run(['sbatch', script_path])
        print(f"Job submitted!")
    else:
        print(f"To submit the job, run: sbatch {script_path}")


if __name__ == "__main__":
    main()
