#!/usr/bin/env python3

"""
Script to generate a Slurm job for retraining ResNet-18 on CIFAR-100
to unlearn classes in increments of 5 (0-4, 0-9, 0-14, etc. up to 0-94) using the retrain method.

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
    parser = argparse.ArgumentParser(description='Generate Slurm script for CIFAR-100 retraining unlearning')
    parser.add_argument('--save_dir', type=str, default="checkpoints/retrain_continual_unlearn",
                        help='Base directory to save the retrained models')
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50"],
                        help='Model architecture to use (resnet18 or resnet50)')
    parser.add_argument('--data_path', type=str, default="/u/kdkyum/ptmp_link/.torchvision",
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default="slurm_logs",
                        help='Directory to save Slurm output logs')
    parser.add_argument('--gpu_count', type=int, default=1, choices=[1, 2, 4],
                        help='Number of GPUs to use (1, 2, or 4)')
    parser.add_argument('--time_limit', type=str, default="11:59:00",
                        help='Time limit for the Slurm job (HH:MM:SS)')
    parser.add_argument('--job_name', type=str, default="cifar100_retrain_unlearn",
                        help='Name for the Slurm job')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the job to Slurm after creating the script')
    return parser.parse_args()


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
    
    # Create a unique script name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    script_path = f"{args.output_dir}/cifar100_retrain_unlearn_{timestamp}.sh"
    
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
# Run as array job for each forget class combination (0-4, 0-9, 0-14, etc. up to 0-94)
#SBATCH --array=0-19
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
BASE_SAVE_DIR="{args.save_dir}/{args.arch}/cifar100"
EPOCHS=182
LR=0.1
DATA_PATH="{args.data_path}"
ARCH="{args.arch}"

# Calculate class IDs and save directory based on array task ID
INCREMENT=1
START_CLASS=0
END_CLASS=$((START_CLASS + (SLURM_ARRAY_TASK_ID + 1) * INCREMENT - 1))
CLASS_IDS=$(seq -s " " $START_CLASS $END_CLASS)
SAVE_DIR="${{BASE_SAVE_DIR}}/0-${{END_CLASS}}"

echo "Starting CIFAR-100 retraining unlearning for classes ${{CLASS_IDS}} using ${{ARCH}}..."
echo "Saving model to ${{SAVE_DIR}}"

python main_forget.py --save_dir ${{SAVE_DIR}} \\
    --dataset cifar100 --num_classes 100 --arch ${{ARCH}} \\
    --unlearn retrain --class_to_replace ${{CLASS_IDS}} \\
    --unlearn_epochs ${{EPOCHS}} --unlearn_lr ${{LR}} \\
    --data ${{DATA_PATH}}

echo "CIFAR-100 retrain unlearning complete for classes ${{CLASS_IDS}}!"
"""
    
    # Write the script to file
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    args = parse_args()
    
    print(f"Generating Slurm array job for CIFAR-100 parallel retraining with {args.arch} to unlearn classes in increments (0, 0-1, 0-2, ..., 0-19)...")
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