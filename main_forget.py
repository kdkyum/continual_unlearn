import copy
import os
from collections import OrderedDict
from copy import deepcopy

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate, validate_class_wise
import torchvision.transforms as transforms
import pandas as pd

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    
    # Extract the class to forget (last element if multiple)
    class_to_forget = None
    if hasattr(args, 'class_to_replace') and len(args.class_to_replace) > 0:
        class_to_forget = args.class_to_replace[-1]
        marked_value = -(class_to_forget + 1)  # Convert to negative representation
    
    if args.dataset == "svhn":
        try:
            if class_to_forget is not None:
                marked = forget_dataset.targets == marked_value
            else:
                marked = forget_dataset.targets < 0
        except:
            if class_to_forget is not None:
                marked = forget_dataset.labels == marked_value
            else:
                marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            if class_to_forget is not None:
                marked = retain_dataset.targets != marked_value
            else:
                marked = retain_dataset.targets >= 0
        except:
            if class_to_forget is not None:
                marked = retain_dataset.labels != marked_value
            else:
                marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            if class_to_forget is not None:
                marked = forget_dataset.targets == marked_value
            else:
                marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
        except:
            if class_to_forget is not None:
                marked = forget_dataset.targets == marked_value
            else:
                marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        if args.unlearn != "retrain":
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    utils.dataset_convert_to_test(train_loader_full, args)
    train_class_wise_acc = validate_class_wise(
        train_loader_full, model, args
    )
    test_class_wise_acc = validate_class_wise(
        test_loader, model, args
    )
    # Combine train and test results into a single file
    combined_results = pd.concat(
        [train_class_wise_acc.assign(dataset="train"), test_class_wise_acc.assign(dataset="test")],
        ignore_index=True
    )
    combined_results.to_csv(
        os.path.join(args.save_dir, "class_wise_accuracy.csv"), index=False
    )

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        if retain_len > test_len:
            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        else:
            shadow_train = retain_dataset

        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()