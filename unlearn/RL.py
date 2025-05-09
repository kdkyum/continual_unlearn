import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        try:
            # Save original targets
            original_targets = deepcopy(forget_dataset.targets)
            # Generate random targets different from originals
            random_targets = np.zeros_like(original_targets)
            for i, orig_label in enumerate(original_targets):
                # Generate a random label different from original
                possible_labels = [j for j in range(args.num_classes) if j != orig_label]
                random_targets[i] = np.random.choice(possible_labels)
            forget_dataset.targets = random_targets
        except:
            # For nested dataset structure
            original_targets = deepcopy(forget_dataset.dataset.targets)
            random_targets = np.zeros_like(original_targets)
            for i, orig_label in enumerate(original_targets):
                possible_labels = [j for j in range(args.num_classes) if j != orig_label]
                random_targets[i] = np.random.choice(possible_labels)
            forget_dataset.dataset.targets = random_targets
    
        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
      
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)

            loss = criterion(output_clean, target)
      
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
      
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
      
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
      
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            original_target = target.cuda()
            
            # Generate random targets that are different from original ones
            new_target = torch.zeros_like(original_target)
            for j in range(original_target.size(0)):
                possible_labels = [k for k in range(args.num_classes) if k != original_target[j].item()]
                new_target[j] = torch.tensor(np.random.choice(possible_labels))
            target = new_target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

    return top1.avg