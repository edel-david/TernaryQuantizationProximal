# %%
# Copyright (c) Meta Platforms, Inc. and affiliates.
# and David Edel
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple example of QAT using DDP (ResNet on CIFAR-10)
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10"""

import torch

import json
import torch.nn as nn
from torchvision import transforms as T
from torch import optim
from torchvision.models import get_model

import wandb
import time

from tqpmod.logging_utils import init_loger_and_folder

import math  # for nan check

from torchao.prototype.parq.optim import QuantOptimizer, ProxPARQ
from torchao.prototype.parq.quant import UnifQuantizer, TernaryUnifQuantizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
torch.set_float32_matmul_precision("high")  # improves matmul speed


# %%
# NOTE: the previous soft-binning regularizer (SoftBinningScale /
# calc_reg_loss_scale_layer) has been removed. Quantization-aware training
# is now handled entirely by PARQ's QuantOptimizer, which projects weights
# onto the quantized grid directly inside optimizer.step() -- see main()
# below for how the optimizer, quantizer (UnifQuantizer) and proximal map
# (ProxPARQ) are configured for ternary ({-1, 0, +1}) quantization.


# %%
from tqdm import tqdm


def train_epoch_normal(
    model, loader, criterion, optimizer, scaler, lr_scheduler, device, amp=True
):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    pbar = tqdm(loader, desc="Training")

    optimizer.zero_grad(set_to_none=True)
    for batch in pbar:

        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        with torch.no_grad():
            running_loss += loss.detach()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum()
            total += labels.size(0)
        # Step optimizer once all k steps have finished accumulating
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        pbar.set_postfix(
            {"loss": loss.detach().item()}
        )  # slower becuase of .item(), but ok

    return running_loss.item() / (len(loader)), 100.0 * correct.item() / total


def train_epoch_imagenet_kaccum_chunked(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    device,
    amp=True,
    k: int = 8,
):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0

    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for batch in pbar:

        all_images = batch["pixel_values"].to(device, non_blocking=True)
        all_labels = batch["label"].to(device, non_blocking=True)

        img_chunks = torch.chunk(all_images, k, dim=0)
        label_chunks = torch.chunk(all_labels, k, dim=0)

        torch.compiler.cudagraph_mark_step_begin()
        for images, labels in zip(img_chunks, label_chunks):
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels) / k
            scaler.scale(loss).backward()

            with torch.no_grad():
                running_loss += loss.detach()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum()
                total += labels.size(0)
        # Step optimizer once all k steps have finished accumulating
        # (this also triggers PARQ's quantization proximal-map update)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        pbar.set_postfix(
            {"loss": (loss.detach() * k).item()}
        )  # slower becuase of .item(), but ok
        #break
    return running_loss.item() / (len(loader)), 100.0 * correct.item() / total


# Evaluation function
@torch.no_grad()
def evaluate(
    model, loader, criterion, device, amp=True
):  # transform_train_x_dtype = torch.float32
    model.eval()
    running_loss = torch.zeros((), device=device, requires_grad=False)
    correct = torch.zeros((), device=device, requires_grad=False)
    total = 0

    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        torch.compiler.cudagraph_mark_step_begin()  # probably not needed
        images = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        images: torch.Tensor
        labels: torch.Tensor
        # images, labels = images.to(device,transform_train_x_dtype, non_blocking=True), labels.to(
        #    device, non_blocking=True
        # )
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.detach()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().detach()
        total += labels.size(0)

    avg_loss = running_loss.item() / len(loader)
    test_accuracy = 100.0 * correct.item() / total
    return avg_loss, test_accuracy



import numpy as np

num_epochs = 90
lr_warmup_epochs = 5


def build_lr_factor(lr_warmup_epochs, num_epochs):
    num_epochs_lr_schedule = num_epochs

    def get_lr_factor(epoch):
        if epoch < lr_warmup_epochs:
            return (epoch + 1) / lr_warmup_epochs
        elif epoch > num_epochs_lr_schedule:
            return get_lr_factor(num_epochs_lr_schedule)
        else:
            return max(
                [
                    0.5
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (epoch - lr_warmup_epochs)
                            / (num_epochs_lr_schedule - lr_warmup_epochs)
                        )
                    ),
                    1e-9,
                ]
            )

    return get_lr_factor

# %%
class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        *,
        bias=True,
        k=8,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        self.k = k
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype,
            # bias=bias, #wtf is going on here
        )

    def forward(self, input):
        chunks = torch.chunk(input, self.k, dim=0)
        ret = torch.cat([super().forward(tensor) for tensor in chunks], dim=0)
        return ret


def replace_bns(model, k=8):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            bn = child
            setattr(
                model,
                name,
                CustomBatchNorm2d(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                    device=DEVICE,
                    dtype=None,
                    k=k,
                ),
            )
        else:
            replace_bns(child, k)


# %%
# from imagenet_utils import get_hf_augmented_loaders
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms as T
from torch.utils.data import DataLoader, RandomSampler
from datasets import load_dataset


from tqpmod.model_utils import rebalance_layers
from typing import Any, Callable


class HFTransform:
    """Module-level (picklable) wrapper around a torchvision transform pipeline."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, examples):
        examples["pixel_values"] = [
            self.transform(img.convert("RGB")) for img in examples["image"]
        ]
        return examples


def hf_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "label": labels}


def get_hf_augmented_loaders(
    ds,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers_train: int = 10,
    num_workers_val: int = 4,
    seed: int = 42,
):
    normalize_transform = T.Normalize(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(int(input_size / 0.875), interpolation=3),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    train_ds = ds["train"].with_transform(HFTransform(train_transform))
    val_ds = ds["validation"].with_transform(HFTransform(val_transform))

    generator = torch.Generator().manual_seed(seed)
    train_sampler = RandomSampler(train_ds, generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers_train,
        collate_fn=hf_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(1.5 * batch_size),
        shuffle=False,
        num_workers=num_workers_val,
        collate_fn=hf_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# %%


def main():
    ds = load_dataset("ILSVRC/imagenet-1k")

    def longrun_scale_layer(
        train_params: dict[str, Any], run_name: str = "ImageNet_scale_train_"
    ):
        logger, folder = init_loger_and_folder(run_name)
        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        SEED = torch.randint(0, 10000, (1,))
        num_epochs = 90  
        learning_rate = 0.1
        weight_decay = 1e-4
        lr_warmup_epochs = 5
        REG_wait_epochs = 0
        if train_params is not None:
            num_epochs = train_params.get("num_epochs", num_epochs)
            num_epochs_lr_schedule = num_epochs - 1
            num_epochs_lr_schedule = train_params.get(
                "num_epochs_lr_schedule", num_epochs_lr_schedule
            )
            learning_rate = train_params.get("learning_rate", learning_rate)
            weight_decay = train_params.get("weight_decay", weight_decay)
            lr_warmup_epochs = train_params.get(
                "lr_warmup_epochs", lr_warmup_epochs
            )

            REG_wait_epochs = train_params.get("REG_wait_epochs", REG_wait_epochs)
            SEED = train_params.get("SEED", SEED)
        i_accum = 4
        k_bn = 2
        batch_size = 256  # 32 * k (=8) = 256, optimal for lrmax = 0.1
        train_loader, val_loader = get_hf_augmented_loaders(
            ds,
            batch_size,
            num_workers_train=11,
            num_workers_val=4,
            seed=SEED.item(),
        )
        steps_per_epoch = len(train_loader)
        # ALL these hyperparams could get overwritten by train_params

        


        anneal_start = REG_wait_epochs
        anneal_end = num_epochs
        anneal_start_step = anneal_start * steps_per_epoch
        anneal_end_step = anneal_end * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch
        torch.manual_seed(SEED)

        # define loss function (criterion) and optimizer
        scaler = torch.amp.GradScaler("cuda")
        label_smoothing = 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(
            DEVICE
        )  #

        model = get_model("resnet50", num_classes=1000, weights=None)
        replace_bns_bool = True
        if replace_bns_bool == True:
            replace_bns(model, k=k_bn)
            print("!" * 30)
            print(
                "CustomBatchNorm in use, make sure correct train_epoch and batchsize are activated"
            )
            print(
                "Make sure batch size is 256 or that you know what you are doing!!"
            )
            print("!" * 30)

        param_groups = []

        model = model.to(DEVICE)
        main_group = {"params":[],"weight_decay":weight_decay,"lr":learning_rate,"quant_bits": 0,}

        for name,param in model.named_parameters():
            #if ("conv" in name or "fc" in name) and "scale" not in name and "bias" not in name:
            if ("weight" in name) and param.ndim > 1:
                main_group["params"].append(param)        
        param_groups.append(main_group)
        
        # make a rest group
        rest_group = []

        param_groups_list = []
        for param_group in param_groups:
            param_groups_list.extend([id(param) for param in param_group["params"]])
        for name, param in model.named_parameters():
            if id(param) not in param_groups_list:
                rest_group.append(param)
        
        param_groups.append(
            {"params": rest_group, "weight_decay": 0, "lr": learning_rate}
        )

        get_lr_factor: Callable = build_lr_factor(
            lr_warmup_epochs * steps_per_epoch,
            num_epochs_lr_schedule * steps_per_epoch,
        )

        base_optimizer = torch.optim.SGD(
            param_groups,
            learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            # fused=True,  # fused Adam: on some configs about 5 percent faster.
        )
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            base_optimizer, lr_lambda=get_lr_factor
        )

        # --- PARQ QAT setup -------------------------------------------------
        # quant_bits=0 was set on `main_group` above, which tells PARQ to
        # quantize those params to a ternary {-1, 0, +1} codebook (per the
        # PARQ "quant-bits" convention: 0 == ternary, 1-4 == that many bits).
        # `rest_group` has no "quant_bits" key, so those params (biases,
        # BN affine params, etc.) are left in full precision.
        #
        # UnifQuantizer computes the (per-tensor) quantized grid/scale, and
        # ProxPARQ is the PARQ proximal map: it gradually anneals weights
        # from full precision onto the quantized grid over
        # [anneal_start, anneal_end] optimizer steps, using a sigmoid
        # schedule controlled by `steepness`.


        quantizer = TernaryUnifQuantizer()
        prox_map = ProxPARQ(
            anneal_start=anneal_start_step,
            anneal_end=anneal_end_step,
            steepness=75,
        )
        optimizer = QuantOptimizer(base_optimizer, quantizer, prox_map)
        best_acc = 0.0

        optimizer.zero_grad(set_to_none=True)
        n_reg_params = sum([torch.numel(x) for x in main_group["params"]])
        n_params = sum([torch.numel(x) for x in model.parameters()])
        print(
            f"This model has {n_reg_params} ternary (PARQ, quant_bits=0) params out of {n_params} total params ({n_reg_params/n_params*100:.2f}%)"
        )
        model.compile(mode="max-autotune", fullgraph=True,)# dynamic=False)
        # model.compile()
        torch.save(
            {
                "epoch": 0,
                "num_epochs": num_epochs,
                "SEED": SEED,
                "label_smoothing": label_smoothing,
                "REG_wait_epochs": REG_wait_epochs,
                "lr_warmup_epochs": lr_warmup_epochs,
                "num_epochs_lr_schedule": num_epochs_lr_schedule,
                "model_state_dict": model.state_dict(),
                "base_optimizer_state_dict": base_optimizer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_acc": 0,
                "lr_schedule": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": json.dumps(history),
                "best_acc": best_acc,
                "batch_size": batch_size,
            },
            f"{folder}/init_pre_train.pth",
        )
        with wandb.init(
            project="ResNet50-TQP",
            name=f"{run_name}",
            config={
                "architecture": "ResNet50",
                "load_state": False,
                "Use Pretrained": f"{False}",
                "dataset": "ImageNet",
                "seed": SEED,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "optimizer": base_optimizer.__class__.__name__,
                "start_lr": learning_rate,
                "lr_warmup_epochs": lr_warmup_epochs,
                "weight_decay": weight_decay,
                "reg_wait_epochs": REG_wait_epochs,
                "label_smoothing": label_smoothing,
                "qat_method": "PARQ",
                "quant_bits": 0,  # ternary {-1, 0, +1}
                "quant_method": quantizer.__class__.__name__,
                "quant_proxmap": prox_map.__class__.__name__,
                "anneal_start": anneal_start,
                "anneal_end": anneal_end,
                "note": "",
            },
            # mode="disabled"
        ) as run:

            for epoch in range(0, num_epochs):
                # for epoch in range(start_epoch, 350):
                # for epoch in range(num_epochs, num_epochs+20):
                start_time = time.time()
                logger.info(
                    f"\nEpoch [{epoch+1}/{num_epochs}] | LR: {base_optimizer.param_groups[0]['lr']:.10f}"
                )


                # Train
                # NOTE: `optimizer` (not `base_optimizer`) is passed here --
                # PARQ's QuantOptimizer wraps base_optimizer and performs
                # the ternary quantization proximal-map update as part of
                # optimizer.step(), on top of the ordinary SGD step.
                train_loss, train_acc = train_epoch_imagenet_kaccum_chunked(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    DEVICE,
                    amp=True,
                    k=i_accum,
                )
                #break

                # Save history
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                logger.info(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
                )

                to_log = {
                    "lr": torch.tensor(lr_scheduler.get_last_lr()).mean().item(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }

                if False:
                    pass
                else:
                    test_loss, test_acc = evaluate(
                        model, val_loader, criterion, DEVICE, amp=True
                    )
                    history["test_acc"].append(test_acc)
                    history["test_loss"].append(test_loss)
                    logger.info(
                        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
                    )
                    to_log.update(
                        {
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                        }
                    )
                    best_acc = max([test_acc, best_acc])
                    # Save best model
                    torch.save(
                        {
                            "epoch": epoch,
                            "num_epochs": num_epochs,
                            "SEED": SEED,
                            "label_smoothing": label_smoothing,
                            "REG_wait_epochs": REG_wait_epochs,
                            "lr_warmup_epochs": lr_warmup_epochs,
                            "num_epochs_lr_schedule": num_epochs_lr_schedule,
                            "model_state_dict": model.state_dict(),
                            "base_optimizer_state_dict": base_optimizer.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "current_acc": test_acc,
                            "lr_schedule": lr_scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "history": json.dumps(history),
                            "best_acc": best_acc,
                            "batch_size": batch_size,
                        },
                        f"{folder}/newest_dict.pth",
                    )
                    logger.info(
                        f"saved model. accuracy: {test_acc:.2f}% (best: {best_acc:.2f}%)"
                    )
                    # health check:
                    if math.isnan(train_loss):
                        raise RuntimeError("train loss is nan")

                epoch_time = time.time() - start_time
                logger.info(f"Epoch Time: {epoch_time:.2f}s")
                to_log.update({"epoch time:": epoch_time})
                run.log(to_log, step=epoch,commit=True)

        return history, model, optimizer

    hist, model, optimizer = longrun_scale_layer(

        {"num_epochs": 90, "REG_wait_epochs": 0, "lr_warmup_epochs": 5},
        "parq_wcd_ls_example",
    )
    return hist, model, optimizer

if __name__ == "__main__":
    hist, model, optimizer = main()
