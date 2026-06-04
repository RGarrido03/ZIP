import torch
from torch import nn, Tensor

from torch.optim import SGD, Adam, AdamW, RAdam
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from argparse import ArgumentParser

import os, sys, math
from typing import Union, Tuple, Dict, List, Optional
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import losses


def _check_lr(lr: float, eta_min: float) -> None:
    assert lr > eta_min > 0, f"lr and eta_min must satisfy 0 < eta_min < lr, got lr={lr} and eta_min={eta_min}."


def _check_warmup(warmup_epochs: int, warmup_lr: float) -> None:
    assert warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."


def _warmup_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
) -> float:
    """
    Linear Warmup
    """
    base_lr, warmup_lr = float(base_lr), float(warmup_lr)
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    _check_warmup(warmup_epochs, warmup_lr)

    if epoch < warmup_epochs:        
        # Compute the current learning rate in log-linear scale
        lr = math.exp(math.log(warmup_lr) + epoch * (math.log(base_lr) - math.log(warmup_lr)) / warmup_epochs)
    else:
        lr = base_lr

    return lr


def step_decay(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    step_size: int,
    gamma: float,
    eta_min: float,
) -> float:
    """
    Warmup + Step Decay
    """
    base_lr, warmup_lr, eta_min = float(base_lr), float(warmup_lr), float(eta_min)
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert step_size >= 1, f"step_size must be greater than or equal to 1, got {step_size}."
    assert 0 < gamma < 1, f"gamma must be in the range (0, 1), got {gamma}."
    _check_lr(base_lr, eta_min)
    _check_warmup(warmup_epochs, warmup_lr)

    if epoch < warmup_epochs:
        lr = _warmup_lr(epoch, base_lr, warmup_epochs, warmup_lr)
    else:
        epoch -= warmup_epochs
        lr = base_lr * (gamma ** (epoch // step_size))
        lr = max(lr, eta_min)

    return lr / base_lr


def cosine_annealing(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_max: int,
    eta_min: float,
) -> float:
    """
    Warmup + Cosine Annealing
    """
    base_lr, warmup_lr, eta_min = float(base_lr), float(warmup_lr), float(eta_min)
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert T_max >= 1, f"T_max must be greater than or equal to 1, got {T_max}."
    _check_lr(base_lr, eta_min)
    _check_warmup(warmup_epochs, warmup_lr)

    if epoch < warmup_epochs:
        lr = _warmup_lr(epoch, base_lr, warmup_epochs, warmup_lr)
    else:
        epoch -= warmup_epochs
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

    return lr / base_lr


def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Warmup + Cosine Annealing with Warm Restarts
    """
    base_lr, warmup_lr, eta_min = float(base_lr), float(warmup_lr), float(eta_min)
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    _check_lr(base_lr, eta_min)
    _check_warmup(warmup_epochs, warmup_lr)

    if epoch < warmup_epochs:
        lr = _warmup_lr(epoch, base_lr, warmup_epochs, warmup_lr)
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)
        
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr


def get_loss_fn(args: ArgumentParser) -> nn.Module:
    return losses.QuadLoss(
        input_size=args.input_size,
        block_size=args.block_size,
        bins=args.bins,
        reg_loss=args.reg_loss,
        aux_loss=args.aux_loss,
        weight_cls=args.weight_cls,
        weight_reg=args.weight_reg,
        weight_aux=args.weight_aux,
        numItermax=args.numItermax,
        regularization=args.regularization,
        scales=args.scales,
        min_scale_weight=args.min_scale_weight,
        max_scale_weight=args.max_scale_weight,
        alpha=args.alpha,
    )


def get_optimizer(
    args: ArgumentParser,
    model: nn.Module
) -> Tuple[Union[SGD, Adam, AdamW, RAdam], LambdaLR]:
    backbone_params = []
    new_params = []
    vpt_params = []
    adpater_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vpt" in name:
            vpt_params.append(param)
        elif "adapter" in name:
            adpater_params.append(param)
        elif "backbone" not in name or ("refiner" in name or "decoder" in name):
            new_params.append(param)
        else:
            backbone_params.append(param)
    

    param_groups = [
        {"params": new_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.backbone_weight_decay}
    ]
    if args.optimizer == "adam":
        optimizer = Adam(param_groups)
    elif args.optimizer == "adamw":
        optimizer = AdamW(param_groups)
    elif args.optimizer == "sgd":
        optimizer = SGD(param_groups, momentum=0.9)
    else:
        assert args.optimizer == "radam", f"Expected optimizer to be one of ['adam', 'adamw', 'sgd', 'radam'], got {args.optimizer}."
        optimizer = RAdam(param_groups, decoupled_weight_decay=True)

    if args.scheduler == "step":
        lr_lambda = partial(
            step_decay,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            step_size=args.step_size,
            eta_min=args.eta_min,
            gamma=args.gamma,
        )
    elif args.scheduler == "cos":
        lr_lambda = partial(
            cosine_annealing,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            T_max=args.T_max,
            eta_min=args.eta_min,
        )
    elif args.scheduler == "cos_restarts":
        lr_lambda = partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
            base_lr=args.lr
        )

    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=[lr_lambda for _ in range(len(param_groups))]
    )

    return optimizer, scheduler


def load_checkpoint(
    args: ArgumentParser,
    model: nn.Module,
    optimizer: Union[SGD, Adam, AdamW, RAdam],
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
    ckpt_dir: Optional[str] = None,
) -> Tuple[nn.Module, Union[SGD, Adam, AdamW, RAdam], Union[LambdaLR, None], GradScaler, int, Union[Dict[str, float], None], Dict[str, List[float]], Dict[str, float]]:
    ckpt_path = os.path.join(args.ckpt_dir if ckpt_dir is None else ckpt_dir, "ckpt.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        loss_info = ckpt["loss_info"]
        hist_scores = ckpt["hist_scores"]
        best_scores = ckpt["best_scores"]

        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if grad_scaler is not None:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

        print(f"Loaded checkpoint from {ckpt_path}.")

    else:
        start_epoch = 1
        loss_info, hist_scores = None, {"mae": [], "rmse": [], "nae": []}
        best_scores = {k: [torch.inf] * args.save_best_k for k in hist_scores.keys()}
        print(f"Checkpoint not found at {ckpt_path}.")

    return model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_scores, best_scores


def save_checkpoint(
    epoch: int,
    model_state_dict: OrderedDict[str, Tensor],
    optimizer_state_dict: OrderedDict[str, Tensor],
    scheduler_state_dict: OrderedDict[str, Tensor],
    grad_scaler_state_dict: OrderedDict[str, Tensor],
    loss_info: Dict[str, List[float]],
    hist_scores: Dict[str, List[float]],
    best_scores: Dict[str, float],
    ckpt_dir: str,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "loss_info": loss_info,
        "hist_scores": hist_scores,
        "best_scores": best_scores,
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "ckpt.pth"))
