import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Union
from copy import deepcopy

from utils import barrier, reduce_mean, update_loss_info
from evaluate import evaluate


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: Union[GradScaler, None],
    device: torch.device = torch.device("cuda"),
    rank: int = 0,
    nprocs: int = 1,
    **kwargs,
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:
    info = None
    wandb_run = kwargs.pop("wandb_run", None)
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    if "eval_data_loader" in kwargs:  # we are evaluting the model withing one training epoch
        assert "eval_freq" in kwargs and 0 < kwargs["eval_freq"] < 1, f"eval_freq should be a float between 0 and 1, but got {kwargs['eval_freq']}"
        assert "sliding_window" in kwargs, "sliding_window should be provided in kwargs"
        assert "max_input_size" in kwargs, "max_input_size should be provided in kwargs"
        assert "window_size" in kwargs, "window_size should be provided in kwargs"
        assert "stride" in kwargs, "stride should be provided in kwargs"
        assert "max_num_windows" in kwargs, "max_num_windows should be provided in kwargs"

        eval_within_epoch = True
        eval_data_loader = kwargs["eval_data_loader"]
        eval_freq = int(kwargs["eval_freq"] * len(data_loader))
        sliding_window = kwargs["sliding_window"]
        max_input_size = kwargs["max_input_size"]
        window_size = kwargs["window_size"]
        stride = kwargs["stride"]
        max_num_windows = kwargs["max_num_windows"]

        best_scores = {}
        best_weights = {}

    else:
        eval_within_epoch = False
        best_scores = None
        best_weights = None
        
    for batch_idx, (image, gt_points, gt_den_map) in enumerate(data_iter):
        image = image.to(device)
        gt_points = [p.to(device) for p in gt_points]
        gt_den_map = gt_den_map.to(device)
        model.train()
        with torch.set_grad_enabled(True):
            with autocast(device_type="cuda", enabled=grad_scaler is not None and grad_scaler.is_enabled()):
                if (model.module.zero_inflated if ddp else model.zero_inflated):
                    pred_logit_pi_map, pred_logit_map, pred_lambda_map, pred_den_map = model(image)
                    total_loss, total_loss_info = loss_fn(
                        pred_logit_pi_map=pred_logit_pi_map,
                        pred_logit_map=pred_logit_map,
                        pred_lambda_map=pred_lambda_map,
                        pred_den_map=pred_den_map,
                        gt_den_map=gt_den_map,
                        gt_points=gt_points,
                    )
                else:
                    pred_logit_map, pred_den_map = model(image)
                    total_loss, total_loss_info = loss_fn(
                        pred_logit_map=pred_logit_map,
                        pred_den_map=pred_den_map,
                        gt_den_map=gt_den_map,
                        gt_points=gt_points,
                    )

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        total_loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in total_loss_info.items()}
        info = update_loss_info(info, total_loss_info)
        barrier(ddp)

        if wandb_run is not None and rank == 0:
            wandb_run.log({"train/loss": total_loss_info.get("total_loss", total_loss.item() if ddp else total_loss.item()), **{f"train/{k}": v for k, v in total_loss_info.items()}})
        if eval_within_epoch and ((batch_idx + 1) % eval_freq == 0 or batch_idx == len(data_loader) - 1):
            batch_scores = evaluate(
                model=model,
                data_loader=eval_data_loader,
                sliding_window=sliding_window,
                max_input_size=max_input_size,
                window_size=window_size,
                stride=stride,
                max_num_windows=max_num_windows,
                device=device,
                amp=grad_scaler is not None and grad_scaler.is_enabled(),
                local_rank=rank,
                nprocs=nprocs,
                progress_bar=False,
            )
            for k, v in batch_scores.items():
                if k not in best_scores:
                    best_scores[k] = v
                    best_weights[k] = deepcopy(model.module.state_dict() if ddp else model.state_dict())
                elif v < best_scores[k]:  # smaller is better
                    best_scores[k] = v
                    best_weights[k] = deepcopy(model.module.state_dict() if ddp else model.state_dict())

            barrier(ddp)

    torch.cuda.empty_cache()
    return model, optimizer, grad_scaler, {k: np.mean(v) for k, v in info.items()}, best_scores, best_weights
