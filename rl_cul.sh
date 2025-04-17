#!/bin/bash

# Define base parameters
MODEL_PATH="checkpoints/base_model/0checkpoint.pth.tar"
BASE_SAVE_DIR="checkpoints/RL_model/forget_class"

# Continual unlearning for classes 1 to 8 using GA
PREV_CLASSES="c0"
SAVE_DIR="${BASE_SAVE_DIR}_c0"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3 c4"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3_c4"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 4 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3 c4 c5"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3_c4_c5"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 4 5 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3 c4 c5 c6"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3_c4_c5_c6"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 4 5 6 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3 c4 c5 c6 c7"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3_c4_c5_c6_c7"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 4 5 6 7 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"

PREV_CLASSES="c0 c1 c2 c3 c4 c5 c6 c7 c8"
SAVE_DIR="${BASE_SAVE_DIR}_c0_c1_c2_c3_c4_c5_c6_c7_c8"
python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
    --unlearn RL --class_to_replace 0 1 2 3 4 5 6 7 8 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
MODEL_PATH="${SAVE_DIR}/RLcheckpoint.pth.tar"