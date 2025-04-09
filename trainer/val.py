import torch
import utils
from imagenet import get_x_y_from_data_dict
import pandas as pd


def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def validate_class_wise(val_loader, model, args):
    """
    Run evaluation with class-wise accuracy
    """
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    
    # switch to evaluate mode
    model.eval()
    
    if args.imagenet_arch:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                
                correct = (pred == target).float()
                
                # Update class-wise accuracy
                for c in range(args.num_classes):
                    class_mask = target == c
                    class_correct[c] += torch.sum(correct * class_mask.float()).item()
                    class_total[c] += torch.sum(class_mask.float()).item()
                
    else:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()
            
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                
                correct = (pred == target).float()
                
                # Update class-wise accuracy
                for c in range(args.num_classes):
                    class_mask = target == c
                    class_correct[c] += torch.sum(correct * class_mask.float()).item()
                    class_total[c] += torch.sum(class_mask.float()).item()
    
    # Calculate accuracy for each class and prepare results
    class_accuracy = torch.zeros(args.num_classes)
    results = []
    for i in range(args.num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of class {i}: {class_accuracy[i]:.2f}%')
        else:
            print(f'Class {i}: No samples')
        results.append({
            "class": i,
            "total_samples": int(class_total[i].item()),
            "correct_counts": int(class_correct[i].item()),
            "accuracy": class_accuracy[i].item()
        })
    
    # Create a pandas DataFrame for the results
    results_df = pd.DataFrame(results)
    print(results_df)
    
    return results_df
