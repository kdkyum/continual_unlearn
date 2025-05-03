#!/usr/bin/env python3

"""
Script to generate a Slurm job for running the SynapTag algorithm on CIFAR-100
for continual unlearning of classes in increments of 5 (0-4, 0-9, 0-14, etc.).

In this script, each step unlearns all classes up to the current increment:
- Step 0: Unlearn classes 0-4
- Step 1: Unlearn classes 0-9
- Step 2: Unlearn classes 0-14
And so on, until the specified maximum class is unlearned.
"""

import os
import argparse
import subprocess
import time
import sys

# Add the parent directory to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Slurm script for cumulative SynapTag unlearning on CIFAR-100')
    parser.add_argument('--save_dir', type=str, default="checkpoints/cifar100_synaptag_continual_unlearn",
                        help='Directory to save the SynapTag models')
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
    parser.add_argument('--sparsity', type=float, default=0.003,
                        help='Sparsity of the mask (for SynapTag)')
    parser.add_argument('--unlearn_lr', type=float, default=0.01,
                        help='Learning rate for unlearning')
    parser.add_argument('--unlearn_epochs', type=int, default=50,
                        help='Number of epochs for unlearning')
    parser.add_argument('--max_class', type=int, default=95,
                        help='Maximum class to unlearn (up to this class in increments of 5)')
    parser.add_argument('--increment', type=int, default=5,
                        help='Increment size for classes to unlearn at each step')
    parser.add_argument('--job_name', type=str, default="cifar100_synaptag_continual_unlearn",
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
    script_path = f"{args.output_dir}/cifar100_synaptag_cumulative_unlearn_{timestamp}.sh"
    
    # Generate the Slurm script content
    slurm_script = f"""#!/bin/bash -l
# Standard output and error:
#SBATCH -o {args.output_dir}/{args.job_name}_%j.out
#SBATCH -e {args.output_dir}/{args.job_name}_%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J {args.job_name}
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
SPARSITY={args.sparsity}
UNLEARN_LR={args.unlearn_lr}
UNLEARN_EPOCHS={args.unlearn_epochs}
DATA_PATH="{args.data_path}"
MAX_CLASS={args.max_class}
INCREMENT={args.increment}

# Create the output directory
mkdir -p ${{BASE_SAVE_DIR}}

# Initialize the model path
"""

    # Handle the initial model path
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

    # Add the cumulative unlearning loop
    slurm_script += """
# Cumulative unlearning loop with increments of $INCREMENT
for END_CLASS in $(seq $INCREMENT $INCREMENT $MAX_CLASS); do
    START_CLASS=0
    PREV_END=$((END_CLASS - INCREMENT))
    
    echo "=========================================================="
    echo "Starting cumulative unlearning step for classes $START_CLASS-$END_CLASS"
    
    # Define the save directory for this step
    STEP_DIR="${BASE_SAVE_DIR}/0-${END_CLASS}"
    mkdir -p ${STEP_DIR}
    
    # Build the cumulative class list (0 to END_CLASS)
    CLASS_LIST=""
    for CLASS_ID in $(seq $START_CLASS $END_CLASS); do
        CLASS_LIST="${CLASS_LIST} ${CLASS_ID}"
    done
    CLASS_LIST=$(echo ${CLASS_LIST} | xargs)  # Trim leading/trailing spaces
    
    # Build the list of new classes in this increment (for reporting only)
    NEW_CLASSES=""
    for CLASS_ID in $(seq $((PREV_END + 1)) $END_CLASS); do
        NEW_CLASSES="${NEW_CLASSES} ${CLASS_ID}"
    done
    NEW_CLASSES=$(echo ${NEW_CLASSES} | xargs)  # Trim leading/trailing spaces
    
    echo "Unlearning classes: ${CLASS_LIST}"
    echo "New classes in this increment: ${NEW_CLASSES}"
    
    # Run SynapTag unlearning using main_forget.py
    COMMAND="python main_forget.py --save_dir ${STEP_DIR} \\
        --unlearn synaptag \\
        --class_to_replace ${CLASS_LIST} \\
        --unlearn_epochs ${UNLEARN_EPOCHS} \\
        --unlearn_lr ${UNLEARN_LR} \\
        --sparsity ${SPARSITY} \\
        --layer_wise \\
        --dataset cifar100 --num_classes 100 \\
        --data ${DATA_PATH}"
    
    # Add model path if we have a previous model
    if [ ! -z "${CURRENT_MODEL_PATH}" ]; then
        COMMAND="${COMMAND} --model_path ${CURRENT_MODEL_PATH}"
    fi
    
    echo "Running command: ${COMMAND}"
    eval "${COMMAND}"
    
    # Save exit status
    EXIT_STATUS=$?
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
        echo "Error: Unlearning for classes 0-${END_CLASS} failed with exit status ${EXIT_STATUS}"
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
    
    echo "Successfully unlearned classes 0-${END_CLASS}"
    echo "=========================================================="
done

echo "Cumulative unlearning completed for all CIFAR-100 classes up to $MAX_CLASS!"
"""
    
    # Write the script to file
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    args = parse_args()
    
    print(f"Generating Slurm job for cumulative SynapTag unlearning of CIFAR-100 classes in increments of {args.increment} up to class {args.max_class}...")
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