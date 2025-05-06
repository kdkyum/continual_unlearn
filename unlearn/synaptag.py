import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import time
import utils
from copy import deepcopy

from .impl import iterative_unlearn


@iterative_unlearn
def synaptag(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)

    try:
        forget_dataset.targets = - np.ones_like(forget_dataset.targets)
    except:
        forget_dataset.dataset.targets = - np.ones_like(forget_dataset.targets)

    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_indices = torch.randperm(len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    # switch to train mode
    model.train()
    device = next(model.parameters()).device

    for i, (image, target) in enumerate(train_loader):
        # Create masks to distinguish between retain and forget samples
        image = image.to(device)
        target = target.to(device)

        retain_mask = target >= 0
        forget_mask = target < 0

        optimizer.zero_grad()
        output_clean = model(image)

        # Initialize uniform distribution for forget samples
        num_classes = args.num_classes
        uniform_soft_label = torch.ones(num_classes, device=device) / num_classes

        # Convert targets to one-hot encoding (use clamp to ensure valid indices)
        target_one_hot = F.one_hot(target.clamp(min=0), num_classes=num_classes).float()

        # Replace forget sample labels with uniform distribution
        if forget_mask.sum() > 0:
            target_one_hot[forget_mask] = uniform_soft_label.expand(forget_mask.sum(), num_classes)

        # Compute loss
        if retain_mask.sum() > 0:
            retain_loss = F.cross_entropy(output_clean[retain_mask], target[retain_mask])
        else:
            retain_loss = torch.tensor(0.0, device=device)

        if forget_mask.sum() > 0:
            forget_loss = F.cross_entropy(output_clean[forget_mask], target_one_hot[forget_mask])
        else:
            forget_loss = torch.tensor(0.0, device=device)

        loss = retain_loss + forget_loss

        # Update model
        loss.backward()
        optimizer.step()

        # Update model's thresholds after optimization
        model._update_thresholds()

        # Record statistics
        losses.update(loss.item(), image.size(0))
        if retain_mask.any():
            acc1 = utils.accuracy(output_clean[retain_mask], target[retain_mask], topk=(1,))[0]
            top1.update(acc1.item(), retain_mask.sum().item())

@iterative_unlearn
def synaptag_RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    # Combine forget and retain datasets
    forget_dataset = deepcopy(forget_loader.dataset)
    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_indices = torch.randperm(len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    # switch to train mode
    model.train()
    device = next(model.parameters()).device
    
    # Get all class indices
    num_classes = args.num_classes
    all_classes = np.arange(num_classes)
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        # Get original targets before modifications
        original_target = target.clone()
        
        # Identify samples from forget dataset
        forget_samples = []
        for idx, (x, y) in enumerate(forget_loader):
            forget_samples.extend(y.tolist())
        forget_samples = set(forget_samples)
        
        forget_mask = torch.tensor([t.item() in forget_samples for t in original_target], device=device)
        retain_mask = ~forget_mask
        
        # Assign random labels to forget samples
        if forget_mask.sum() > 0:
            forget_targets = target[forget_mask]
            random_labels = []
            
            for orig_label in forget_targets:
                available_classes = [c for c in all_classes if c != orig_label.item()]
                random_label = np.random.choice(available_classes)
                random_labels.append(random_label)
            
            random_labels = torch.tensor(random_labels, dtype=torch.long, device=device)
            target[forget_mask] = random_labels
        
        # Forward pass and compute loss
        optimizer.zero_grad()
        output = model(image)
        
        loss = criterion(output, target)
        
        # Update model
        loss.backward()
        optimizer.step()
        
        # Update model's thresholds after optimization
        model._update_thresholds()
        
        # Record statistics
        losses.update(loss.item(), image.size(0))
        if retain_mask.any():
            acc1 = utils.accuracy(output[retain_mask], original_target[retain_mask], topk=(1,))[0]
            top1.update(acc1.item(), retain_mask.sum().item())


@iterative_unlearn
def synaptag_GA(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    # Combine forget and retain datasets
    forget_dataset = deepcopy(forget_loader.dataset)
    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_indices = torch.randperm(len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    # switch to train mode
    model.train()
    device = next(model.parameters()).device
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        # Identify samples from forget dataset
        forget_samples = []
        for idx, (x, y) in enumerate(forget_loader):
            forget_samples.extend(y.tolist())
        forget_samples = set(forget_samples)
        
        forget_mask = torch.tensor([t.item() in forget_samples for t in target], device=device)
        retain_mask = ~forget_mask
        
        # Forward pass
        optimizer.zero_grad()
        output = model(image)
        
        # Compute losses separately for retain and forget samples
        if retain_mask.sum() > 0:
            retain_loss = F.cross_entropy(output[retain_mask], target[retain_mask])
        else:
            retain_loss = torch.tensor(0.0, device=device)
        
        if forget_mask.sum() > 0:
            forget_loss = F.cross_entropy(output[forget_mask], target[forget_mask])
        else:
            forget_loss = torch.tensor(0.0, device=device)
        
        # Gradient ascent on forget samples by subtracting the forget loss
        loss = retain_loss - forget_loss
        
        # Update model
        loss.backward()
        optimizer.step()
        
        # Update model's thresholds after optimization
        model._update_thresholds()
        
        # Record statistics (using the absolute value of loss for logging)
        losses.update(loss.item(), image.size(0))
        if retain_mask.any():
            acc1 = utils.accuracy(output[retain_mask], target[retain_mask], topk=(1,))[0]
            top1.update(acc1.item(), retain_mask.sum().item())


