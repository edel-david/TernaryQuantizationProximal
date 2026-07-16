from typing import Any, Dict, List, Optional, Set, Tuple
from timm.layers import PatchEmbed
from torch import Tensor
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import logging
NORM_LAYERS = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm)


def get_param_groups(
    model: torch.nn.Module,
    params_quant: Dict[str, Tensor],
    params_no_wd: Dict[str, Tensor],
    params_wd: Dict[str, Tensor],
    skip_wd_names: Optional[Set[str]] = None,
    prefix: str = "",
    force_full_prec: bool = False,
) -> None:
    """Recurse over children of model to extract quantizable params_quant, as well as
    non-quantizable params (params_no_wd, params_wd).
    """
    # drop torch.compile and DDP wrapper prefixes, if they exist
    for mn, module in model.named_children():
        cur_prefix = f"{prefix}.{mn}" if prefix else mn

        # leave ViT embedding and final classification layer at full precision
        # TODO: generalize to other architectures
        use_full_prec = (
            force_full_prec
            or isinstance(module, PatchEmbed)
            or isinstance(module, NORM_LAYERS)
        )
        for pn, param in module.named_parameters(recurse=False):
            param_name = f"{cur_prefix}.{pn}"
            for attr in ("_orig_mod", "module"):
                param_name = param_name.rsplit(f"{attr}.", 1)[-1]

            use_full_prec |= param_name.startswith("head.")
            if not use_full_prec and pn == "weight":
                params_quant[param_name] = param
            elif pn == "bias" or skip_wd_names and param_name in skip_wd_names:
                params_no_wd[param_name] = param
            else:
                params_wd[param_name] = param
        get_param_groups(
            module,
            params_quant,
            params_no_wd,
            params_wd,
            skip_wd_names=skip_wd_names,
            prefix=cur_prefix,
            force_full_prec=use_full_prec,
        )


def split_param_groups(
    model: torch.nn.Module,
    skip_wd_names: Optional[Set[str]] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits model parameters into 3 groups, described below.

    Returns:
        params_quant: quantized, weight decay
        params_no_wd: unquantized, no weight decay
        params_wd: unquantized, weight decay
    """
    params_quant, params_no_wd, params_wd = {}, {}, {}
    get_param_groups(
        model, params_quant, params_no_wd, params_wd, skip_wd_names=skip_wd_names
    )
    n_found_params = len(params_quant) + len(params_no_wd) + len(params_wd)
    assert n_found_params == len(list(model.parameters()))

    for name, dct in zip(
        ("quant", "no_wd", "wd"), (params_quant, params_no_wd, params_wd)
    ):
        print(f"[params_{name}], {len(dct)}: {tuple(dct.keys())}")
    return (
        list(params_quant.values()),
        list(params_no_wd.values()),
        list(params_wd.values()),
    )


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    evaluate: bool,
    seed: int,
    logger = logging.getLogger(__name__),
    debug = False,
    reduced_set = False,
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

        if reduced_set ==True:
            logger.warning("DEBUG IS ENABLED IN Dataset creation!!!!!!")
            from torch.utils.data import Subset
            train_set = Subset(train_set, torch.arange(256))


        train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=batch_size,
            num_workers=11 if not debug else 0,
            pin_memory=True,
            persistent_workers=True if not debug else False,
            drop_last=True # otherwise, some instances are weighed higher
        )
    else:
        train_loader = None

    val_set = datasets.CIFAR10(
        data_dir,
        train=False,
        transform=T.Compose([T.ToTensor(), normalize_transform]),
    )

    if reduced_set ==True:
        logger.warning("DEBUG IS ENABLED IN Dataset creation!!!!!!")
        from torch.utils.data import Subset
        val_set = Subset(val_set, torch.arange(256))


    val_loader = DataLoader(
        val_set,
        batch_size=batch_size, # could be larger to accelerate
        num_workers=(num_workers // 2) if not debug else 0,
        pin_memory=True,
        persistent_workers=True if not debug else False,
    )
    return train_loader, val_loader
