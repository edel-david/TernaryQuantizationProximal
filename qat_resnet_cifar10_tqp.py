# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple example of QAT using DDP (ResNet on CIFAR-10)
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as T

from .model import resnet


DEVICE = torch.device("cuda:0")
SEED = 32
def main():
    torch.set_float32_matmul_precision("high")  # improves matmul speed
    torch.manual_seed(SEED)

    train_loader, val_loader = create_data_loaders(
        "~/data",
        128,
        6,
        False,
        SEED,
    )
    steps_per_epoch = len(train_loader)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # specify number of quantization bits for different parameter groups
    params_quant, params_no_wd, params_wd = split_param_groups(model)
    param_groups = [
        {"params": params_quant, "quant_bits": 0},
        {"params": params_no_wd, "weight_decay": 0},
        {"params": params_wd},
    ]

    # construct the base optimizer
    base_optimizer = torch.optim.SGD(
        param_groups,
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    evaluate: bool,
    seed: int,
):
    normalize_transform = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    )
    if not evaluate:
        train_set = datasets.CIFAR10(
            data_dir,
            train=True,
            transform=T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4),
                    T.ToTensor(),
                    normalize_transform,
                ]
            ),
            download=True,
        )
        train_sampler = DistributedSampler(train_set, seed=seed)
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        train_loader = None

    val_set = datasets.CIFAR10(
        data_dir,
        train=False,
        transform=T.Compose([T.ToTensor(), normalize_transform]),
    )
    val_sampler = SequentialSampler(val_set)
    val_loader = DataLoader(
        val_set,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    main()
