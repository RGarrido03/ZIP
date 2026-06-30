from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose
import os, sys
from argparse import ArgumentParser
from typing import Union, Tuple, List, Dict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import datasets


def calc_bin_center(
    bins: List[Tuple[float, float]],
    count_stats: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    """
    Calculate the representative value for each bin based on the count statistics.

    `bins` may look like: [(0, 0), (1, 1), (2, 3), (4, 6), (7, float('inf'))]
    `count_stats` may look like: {0: 10, 1: 20, 2: 30, 3: 40, 4: 50, 5: 60, 6: 70, 7: 80, 8: 90, 9: 100}
    In this example, for bin (2, 3), we have 30 samples of 2 and 40 samples of 3 that fall into this bin.
    The representative value for this bin is (30 * 2 + 40 * 3) / (30 + 40) = 2.6.

    The returned list will have the same length as `bins`, and each element is the representative value for the corresponding bin.
    """
    bin_counts = [0] * len(bins)
    bin_sums = [0] * len(bins)
    for k, v in count_stats.items():
        for i, (start, end) in enumerate(bins):
            if start <= int(k) <= end:
                bin_counts[i] += int(v)
                bin_sums[i] += int(v) * int(k)
                break
    assert all(c > 0 for c in bin_counts), f"Expected all bin_counts to be greater than 0, got {bin_counts}. Consider to re-design the bins {bins}."
    bin_centers = [s / c for s, c in zip(bin_sums, bin_counts)]
    return bin_centers, bin_counts


def get_dataloader(args: ArgumentParser, split: str = "train") -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    ddp = args.nprocs > 1
    if split == "train":  # train, strong augmentation
        transforms = [
            datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.aug_min_scale, args.aug_max_scale)),
            datasets.RandomHorizontalFlip(),
        ]
        if args.aug_brightness > 0 or args.aug_contrast > 0 or args.aug_saturation > 0 or args.aug_hue > 0:
            transforms.append(datasets.ColorJitter(
                brightness=args.aug_brightness, contrast=args.aug_contrast, saturation=args.aug_saturation, hue=args.aug_hue
            ))
        if args.aug_blur_prob > 0 and args.aug_kernel_size > 0:
            transforms.append(datasets.RandomApply([
                datasets.GaussianBlur(kernel_size=args.aug_kernel_size),
            ], p=args.aug_blur_prob))
        if args.aug_saltiness > 0 or args.aug_spiciness > 0:
            transforms.append(datasets.PepperSaltNoise(
                saltiness=args.aug_saltiness, spiciness=args.aug_spiciness,
            ))
        transforms = Compose(transforms)

    elif args.sliding_window and args.resize_to_multiple:
        transforms = datasets.Resize2Multiple(args.window_size, stride=args.stride)

    else:
        transforms = None

    dataset_class = datasets.InMemoryCrowd if args.in_memory_dataset else datasets.Crowd
    prefetch_factor = None if args.num_workers == 0 else 3
    persistent_workers = False if args.num_workers == 0 else True
    
    dataset = dataset_class(
        dataset=args.dataset,
        split=split,
        transforms=transforms,
        sigma=args.sigma,
        return_filename=False,
        num_crops=args.num_crops if split == "train" else 1,
    )

    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset, num_replicas=args.nprocs, rank=args.local_rank, shuffle=True, seed=args.seed+args.local_rank)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        return data_loader, sampler

    elif (not ddp) and split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        return data_loader, None

    elif ddp and split == "val":
        sampler = DistributedSampler(dataset, num_replicas=args.nprocs, rank=args.local_rank, shuffle=False)
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        return data_loader
    
    else:  # (not ddp) and split == "val"
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        return data_loader
