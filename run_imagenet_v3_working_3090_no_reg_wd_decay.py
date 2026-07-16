# %%
# Copyright (c) Meta Platforms, Inc. and affiliates.
# and David Edel
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple example of QAT using DDP (ResNet on CIFAR-10)
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10"""

from functools import partial

import torch

import json
import torch.nn as nn
from torchvision import transforms as T
from torch import Tensor, optim
from torchvision.models import get_model

import wandb
import time

from tqpmod.logging_utils import init_loger_and_folder

import math  # for nan check

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
torch.set_float32_matmul_precision("high")  # improves matmul speed


# %%
class SoftBinningScale:
    def __init__(self, bins=3):
        if bins < 3 or bins % 2 != 1:
            raise ValueError("bins must be odd >= 3")
        self.bins = bins // 2
        self.max = self.bins

    @torch.no_grad()
    @torch.compile()
    def call_test(self, x: torch.Tensor):
        x = torch.abs(x)
        y = 1 - self.max + x
        mask = x < self.max
        y[mask] = x[mask] - torch.floor(x[mask])
        y = y * torch.abs(1 - y)
        return y

    def compute_xright_quantile(self, x: torch.Tensor):
        x, _ = x.clone().detach().flatten().sort()
        third = x[x.shape[0] // 3]
        two_thirds = x[int(x.shape[0] * (2 / 3))]
        return third.abs() + two_thirds.abs()


soft_binning = SoftBinningScale(3)


# %%
@torch.compile()
@torch.no_grad()
def calc_reg_loss_scale_layer(
    regularized_params,
    soft_binning: SoftBinningScale,
    n_params,
    device=torch.device("cuda:0"),
    M=1,
    beta=5e-4,
):
    """
    returns tuple of:
    avg_reg_loss, a norm, %done
    """

    reg_loss = torch.zeros((), device=device, requires_grad=False)
    quantized_params = torch.zeros(
        (), device=device, requires_grad=False, dtype=torch.int64
    )
    norm = torch.zeros((), device=device, requires_grad=False)
    for param in regularized_params:

        reg_loss += soft_binning.call_test(param).sum().detach()

        calc_param = param.detach()
        quantized_params += (
            (abs(calc_param - calc_param.round().clip(-M, M)) < (beta / 2))
            .sum()
            .to(torch.int64)
        )

        norm += param.norm().detach()

    return (
        (reg_loss / n_params).item(),
        norm.sqrt().item(),
        (quantized_params / n_params).item(),
    )


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
    optimizer:torch.optim.Optimizer,
    scaler,
    lr_scheduler,
    wdd_setter,
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
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        wdd_setter()
        pbar.set_postfix(
            {"loss": (loss.detach() * k).item()}
        )  # slower becuase of .item(), but ok
        #break
    return running_loss.item() / (len(loader)), 100.0 * correct.item() / total


# Evaluation function
@torch.no_grad()
def evaluate(
    model, loader, criterion, device, amp=False
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
from typing import Any, Callable, cast, override


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

from torch.optim.lr_scheduler import _update_param_group_val

def update_wds(param_groups_wdd,base_wds,max_lrs):
    for param_group,base_wd,max_lr in zip(param_groups_wdd,base_wds,max_lrs):
        # _update_param_group_val(param_group,"weight_decay",param_group["lr"] * base_wd)

        _update_param_group_val(param_group,"weight_decay",param_group["lr"] / max_lr * base_wd)

def main():
    ds = load_dataset("ILSVRC/imagenet-1k")

    def longrun_scale_layer(
        beta, train_params: dict[str, Any], run_name: str = "ImageNet_scale_train_"
    ):
        try:
            logger, folder = init_loger_and_folder(run_name)
            history = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
                "reg_loss": [],
                "norm": [],
                "factor_avg": [],
                "beta": [],
            }

            # ALL these hyperparams could get overwritten by train_params
            num_epochs = 80  # 1000  # like 310
            num_epochs_lr_schedule = 79  # 999 # 300
            learning_rate = 0.1
            weight_decay = 1e-4
            lr_warmup_epochs = 10
            percentage_new_cos = 0.1
            new_cos_epoch = (
                num_epochs - lr_warmup_epochs
            ) / 2  # gets overwritten below

            REG_wait_epochs = 11  # 55

            SEED = torch.randint(0, 10000, (1,))

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
                percentage_new_cos = train_params.get(
                    "percentage_new_cos", percentage_new_cos
                )
                new_cos_epoch = (
                    num_epochs - lr_warmup_epochs
                ) / 2  # Overwrite default based on provided params
                new_cos_epoch = train_params.get("new_cos_epoch", new_cos_epoch)
                REG_wait_epochs = train_params.get("REG_wait_epochs", REG_wait_epochs)
                SEED = train_params.get("SEED", SEED)

            torch.manual_seed(SEED)
            batch_size = 256  # 32 * k (=8) = 256, optimal for lrmax = 0.1

            # train_loader = get_train_loader(batch_size)
            # val_loader = get_val_loader(batch_size)
            train_loader, val_loader = get_hf_augmented_loaders(
                ds,
                batch_size,
                num_workers_train=11,
                num_workers_val=4,
                seed=SEED.item(),
            )
            steps_per_epoch = len(train_loader)

            # define loss function (criterion) and optimizer
            scaler = torch.amp.GradScaler("cuda")
            label_smoothing = 0.1
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(
                DEVICE
            )  #

            model = get_model("resnet50", num_classes=1000, weights=None)
            replace_bns_bool = True
            if replace_bns_bool == True:
                replace_bns(model, k=4)
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
            conv_group = {"params":[],"weight_decay":weight_decay,"lr":learning_rate,"quant_bits": 0,}

            for name,param in model.named_parameters():
                #if ("conv" in name or "fc" in name) and "scale" not in name and "bias" not in name:
                if ("weight" in name) and param.ndim > 1 and not "fc" in name:
                    conv_group["params"].append(param)        
            param_groups.append(conv_group)
            
            linear_group = {"params":[],"weight_decay":weight_decay,"lr":learning_rate,"quant_bits": 0,}
            linear_group["params"].append(model.fc.weight)
            param_groups.append(linear_group)
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
            #print("DEBUG: ANOMALY DETECTOR")
            #torch.autograd.set_detect_anomaly(True)
            best_acc = 0.0
            best_reg_loss = float("inf")

            base_optimizer.zero_grad(set_to_none=True)
            reg_params = [x for x in conv_group["params"]]
            n_reg_params = sum([torch.numel(x) for x in conv_group["params"]])
            n_params = sum([torch.numel(x) for x in model.parameters()])
            print(
                f"This model has {n_reg_params} reg params out of {n_params} total params ({n_reg_params/n_params*100:.2f}%)"
            )
            print(f"{len(reg_params)=}")
            model.compile(mode="max-autotune", fullgraph=True,)# dynamic=False)
            #model.compile()
            
            #wdd setup
            param_groups_wdd = [conv_group]
            base_wds = [group["weight_decay"] for group in param_groups_wdd]
            max_lrs = [learning_rate] # effective lr is: alpha / (1-beta), but we dont need to account for that
            wdd_setter = partial(update_wds,param_groups_wdd=param_groups_wdd,base_wds=base_wds,max_lrs=max_lrs)
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
                    "current_acc": 0,
                    "reg_loss": torch.inf,
                    "lr_schedule": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": json.dumps(history),
                    "best_reg_loss": best_reg_loss,
                    "best_acc": best_acc,
                    "batch_size": batch_size,
                },
                f"{folder}/init_pre_train.pth",
            )
            with wandb.init(
                project="ResNet50-TQP",
                name=f"{run_name}-{beta}",
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
                    "beta": beta,
                    "reg_wait_epochs": REG_wait_epochs,
                    "label_smoothing": label_smoothing,
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
                    train_loss, train_acc = train_epoch_imagenet_kaccum_chunked(
                        model,
                        train_loader,
                        criterion,
                        base_optimizer,
                        scaler,
                        lr_scheduler,
                        wdd_setter,
                        DEVICE,
                        amp=True,
                        k=2,
                    )
                    #break
                    reg_loss, norm, share_done = calc_reg_loss_scale_layer(
                        reg_params, soft_binning, n_reg_params, beta=beta
                    )

                   
                    # Save history
                    history["beta"].append(beta)
                    history["train_loss"].append(train_loss)
                    history["train_acc"].append(train_acc)

                    logger.info(
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Reg Loss: {reg_loss:.4f}"
                    )

                    history["reg_loss"].append(reg_loss)
                    history["norm"].append(norm)

                    to_log = {
                        "lr": torch.tensor(lr_scheduler.get_last_lr()).mean().item(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "reg_loss": reg_loss,
                        "norm": norm,
                        "beta": beta,
                        "share_done": share_done,
                    }

                    if False:
                        pass
                    else:
                        test_loss, test_acc = evaluate(
                            model, val_loader, criterion, DEVICE, amp=False
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
                        if reg_loss < best_reg_loss:
                            best_reg_loss = reg_loss
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
                                "current_acc": test_acc,
                                "reg_loss": reg_loss,
                                "lr_schedule": lr_scheduler.state_dict(),
                                "scaler": scaler.state_dict(),
                                "history": json.dumps(history),
                                "best_reg_loss": best_reg_loss,
                                "best_acc": best_acc,
                                "batch_size": batch_size,
                            },
                            f"{folder}/newest_dict.pth",
                        )
                        logger.info(
                            f"saved model. reg_loss: {reg_loss:.3f} vs best_reg_loss: {best_reg_loss:.3f}, accuracy: {test_acc:.2f}%"
                        )
                        # health check:
                        
                    epoch_time = time.time() - start_time
                    logger.info(f"Epoch Time: {epoch_time:.2f}s")
                    to_log.update({"epoch time:": epoch_time})
                    run.log(to_log, step=epoch,commit=True)
                    reg_loss: float
                    if math.isnan(reg_loss):
                        raise RuntimeError("idk why but reg loss in nan")
        except KeyboardInterrupt as e:
            logger.error("error: ", e)
        return history, model, base_optimizer

    hist, model, optimizer = longrun_scale_layer(
        1e-4,
        {"num_epochs": 90, "REG_wait_epochs": 10, "lr_warmup_epochs": 5},
        "no_reg_wdd",
    )
    return hist, model, optimizer

if __name__ == "__main__":
    hist, model, optimizer = main()
