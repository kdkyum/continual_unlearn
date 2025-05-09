import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from copy import deepcopy

import torch.nn as nn

from models.synaptag import SupermaskNet, SupermaskConv, SupermaskLinear
from pruner import extract_mask, prune_model_custom, remove_prune


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def load_best_model_checkpoint(model, device, save_dir):
    """Load the best model checkpoint based on RFA score"""
    best_model_path = os.path.join(save_dir, "model_best.pth.tar")
    if not os.path.exists(best_model_path):
        print("Best model checkpoint not found.")
        return None, None
    
    checkpoint = torch.load(best_model_path, map_location=device)
    if "state_dict" not in checkpoint:
        print("Invalid checkpoint format - missing state_dict.")
        return None, None
    
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Return model and metadata
    metadata = {
        "rfa": checkpoint.get("rfa"),
        "retain_acc": checkpoint.get("retain_acc"),
        "forget_acc": checkpoint.get("forget_acc"),
        "epoch": checkpoint.get("epoch")
    }
    
    return model, metadata


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)

        if 'synaptag' in args.unlearn:
            model = SupermaskNet(deepcopy(model), sparsity=args.sparsity, layer_wise=args.layer_wise).cuda()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            trainable_params = model.parameters() 
            
        optimizer = torch.optim.SGD(
            trainable_params,
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
                
        # Early stopping parameters
        best_rfa = -float('inf')
        epochs_without_improvement = 0
        # early_stop_patience = 20  # Default patience for early stopping
        
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
            )
            
            # Calculate RFA using validation loader
            val_loader = data_loaders['val']
            
            # Handle multiple forget classes
            forget_classes = [args.class_to_replace[-1]]  # Default: take last class
                
            retain_classes = [c for c in np.arange(args.num_classes) if c not in args.class_to_replace]
            
            # Calculate accuracy for forget and retain classes
            forget_correct = 0
            forget_total = 0
            retain_correct = 0
            retain_total = 0
            
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    
                    # Accuracy for forget classes
                    forget_indices = torch.zeros_like(targets, dtype=torch.bool)
                    for forget_class in forget_classes:
                        forget_indices = forget_indices | (targets == forget_class)
                    
                    forget_correct += (predicted[forget_indices] == targets[forget_indices]).sum().item()
                    forget_total += forget_indices.sum().item()
                    
                    # Accuracy for retain classes
                    retain_indices = torch.zeros_like(targets, dtype=torch.bool)
                    for retain_class in retain_classes:
                        retain_indices = retain_indices | (targets == retain_class)
                    
                    retain_correct += (predicted[retain_indices] == targets[retain_indices]).sum().item()
                    retain_total += retain_indices.sum().item()
            
            forget_acc = forget_correct / max(forget_total, 1)
            retain_acc = retain_correct / max(retain_total, 1)
            rfa = retain_acc - forget_acc
            
            print(f"Epoch {epoch} - Retain Acc: {retain_acc:.4f}, Forget Acc: {forget_acc:.4f}, RFA: {rfa:.4f}")
            
            # Check if RFA improved
            if rfa > best_rfa:
                best_rfa = rfa
                epochs_without_improvement = 0

                state = {
                    "rfa": rfa, 
                    "retain_acc": retain_acc,
                    "forget_acc": forget_acc,
                    "epoch": epoch
                }
                # Save the best model based on RFA
                if "synaptag" in args.unlearn:
                    mask = model.get_masks()
                    # Create a copy of the base model
                    base_model_copy = deepcopy(model.base_model)

                    # Map SupermaskNet modules to base model modules
                    supermask_modules = [(n, m) for n, m in model.named_modules() 
                                        if isinstance(m, (SupermaskConv, SupermaskLinear))]
                    base_modules = [(n, m) for n, m in base_model_copy.named_modules() 
                                   if isinstance(m, (nn.Conv2d, nn.Linear))]

                    # Apply masks to base model weights
                    if len(supermask_modules) == len(base_modules):
                        for (sm_name, sm_module), (base_name, base_module) in zip(supermask_modules, base_modules):
                            # Calculate binary mask (1 = keep, 0 = prune)
                            subnet = (sm_module.scores.abs() >= sm_module.threshold).float()
                            
                            # Ensure mask shape matches weight shape
                            if subnet.shape == base_module.weight.shape:
                                # Apply mask to base model weights (zeroing out pruned weights)
                                base_module.weight.data *= subnet
                            else:
                                print(f"Warning: Mask shape mismatch for {sm_name} and {base_name}")
                    else:
                        print(f"Warning: Module count mismatch: SupermaskNet has {len(supermask_modules)}, base model has {len(base_modules)}")

                    # Update the state dictionary with the masked base model
                    state["state_dict"] = base_model_copy.state_dict()
                else:
                    state["state_dict"] = model.state_dict()
                utils.save_checkpoint(state, False, args.save_dir, "", "model_best.pth.tar")
                print(f"New best RFA: {best_rfa:.4f}, model saved as model_best.pth.tar")
            else:
                epochs_without_improvement += 1
                
            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
