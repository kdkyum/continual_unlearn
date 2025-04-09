# Continual-Unlearning for Classification
The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.


## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

    A simple example for ResNet-18 on CIFAR-10.
    ```bash
    python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182 --data ~/.torchvision/dataset --save_dir checkpoints/base_model
    ```

2. Generate Saliency Map
    ```bash
    python generate_mask.py --save_dir checkpoints/salun/mask --model_path checkpoints/base_model --class_to_replace ${class_ids} --unlearn_epochs 1 --data ~/.torchvision/dataset
    ```

3. Unlearn
    *  SalUn
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --class_to_replace ${class_ids} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path} --data ~/.torchvision/dataset
    ```

    A simple example for ResNet-18 on CIFAR-10 to unlearn class 0.
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --class_to_replace 0 --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path mask/with_0.5.pt --data ~/.torchvision/dataset
    ```

    For continual unlearning of multiple classes one by one:
    ```bash
    # First unlearn class 0
    python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --class_to_replace 0 --model_path ${origin_model_path} --save_dir ${save_dir}_class0 --mask_path mask/with_0.5.pt --data ~/.torchvision/dataset
    
    # Then unlearn class 1 from the previously unlearned model
    python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --class_to_replace 1 --model_path ${save_dir}_class0/model_best.pth.tar --save_dir ${save_dir}_class0_1 --mask_path mask/with_0.5.pt --data ~/.torchvision/dataset
    
    # Continue for other classes as needed
    ```

    To compute UA, we need to subtract the forget accuracy from 100 in the evaluation results. As for MIA, it corresponds to multiplying SVC_MIA_forget_efficacy['confidence'] by 100 in the evaluation results. For a detailed clarification on MIA, please refer to Appendix C.3 at the following link: https://arxiv.org/abs/2304.04934.


    * Retrain
    ```bash
    python main_forget.py --save_dir checkpoints/retrain_model/forget_class_${class_ids} --model_path checkpoints/base_model/0model_SA_best.pth.tar --unlearn retrain --class_to_replace ${class_ids} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --data ~/.torchvision/dataset
    ```

    Example for retraining with 8 models using a for loop:
    ```bash
    BASE_SAVE_DIR="checkpoints/retrain_model"
    MODEL_PATH="checkpoints/base_model/0model_SA_best.pth.tar"
    EPOCHS=182
    LR=0.1

    for i in {1..8}; do
        CLASS_IDS=$(seq -s ',' 0 $((i-1)))
        SAVE_DIR="${BASE_SAVE_DIR}/forget_class_${CLASS_IDS//,/}"
        python main_forget.py --save_dir ${SAVE_DIR} --model_path ${MODEL_PATH} \
            --unlearn retrain --class_to_replace ${CLASS_IDS} --unlearn_epochs ${EPOCHS} --unlearn_lr ${LR} --data ~/.torchvision/dataset
    done
    ```

    * FT
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT --class_to_replace ${class_ids} --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
    ```

    * GA
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn GA --class_to_replace ${class_ids} --unlearn_epochs 5 --unlearn_lr 1e-4 --data ~/.torchvision/dataset
    ```

    * IU
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn wfisher --class_to_replace ${class_ids} --alpha ${alpha} --data ~/.torchvision/dataset
    ```

    * l1-sparse
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT_prune --class_to_replace ${class_ids} --alpha ${alpha} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --data ~/.torchvision/dataset
    ```
    
    # Continual Unlearning
    
    To perform continual unlearning where each class is forgotten sequentially, use this approach:
    
    ```bash
    # Define base parameters
    MODEL_PATH="path/to/original/model.pth.tar"
    BASE_SAVE_DIR="continual_unlearn"
    
    # Unlearn class 0
    python main_forget.py --save_dir ${BASE_SAVE_DIR}_c0 --model_path ${MODEL_PATH} \
        --unlearn FT --class_to_replace 0 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
        
    # Unlearn class 1 using the model that already forgot class 0
    python main_forget.py --save_dir ${BASE_SAVE_DIR}_c0_c1 --model_path ${BASE_SAVE_DIR}_c0/model_best.pth.tar \
        --unlearn FT --class_to_replace 0,1 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
        
    # Continue with class 2
    python main_forget.py --save_dir ${BASE_SAVE_DIR}_c0_c1_c2 --model_path ${BASE_SAVE_DIR}_c0_c1/model_best.pth.tar \
        --unlearn FT --class_to_replace 0,1,2 --unlearn_epochs 10 --unlearn_lr 0.01 --data ~/.torchvision/dataset
    ```
    
    This process can be repeated for all classes you wish to unlearn, creating a chain of unlearning operations.