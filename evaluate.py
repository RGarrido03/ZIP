import torch
from torch.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import wandb
import random


_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def _density_to_image(arr: np.ndarray, caption: str, background: np.ndarray | None = None, alpha: float = 0.5) -> wandb.Image:
    """Convert a 2D density/lambda array to a jet-colormap RGB image for wandb.

    If *background* (HWC uint8) is provided, the heatmap is alpha-blended on top
    of it so the original scene is visible behind the model output.
    """
    vmax = arr.max()
    if vmax <= 0:
        vmax = 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    colored = cm.jet(norm(arr))
    heatmap = (colored[:, :, :3] * 255).astype(np.uint8)

    if background is not None:
        blended = (heatmap * alpha + background * (1 - alpha)).astype(np.uint8)
        return wandb.Image(blended, caption=caption)

    return wandb.Image(heatmap, caption=caption)

from utils import sliding_window_predict, barrier, calculate_errors


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    sliding_window: bool,
    max_input_size: int = 4096,
    window_size: int = 224,
    stride: int = 224,
    max_num_windows: int = 64,
    device: torch.device = torch.device("cuda"),
    amp: bool = False,
    local_rank: int = 0,
    nprocs: int = 1,
    progress_bar: bool = True,
    wandb_run = None,
    num_log_samples: int = 4,
) -> Tuple[Tensor, Tensor]:
    ddp = nprocs > 1
    model = model.to(device)
    model.eval()
    pred_counts, gt_counts = [], []
    data_iter = tqdm(data_loader) if (local_rank == 0 and progress_bar) else data_loader

    for image, gt_points, gt_density in data_iter:
        image = image.to(device)
        image_height, image_width = image.shape[-2:]
        gt_counts.extend([len(p) for p in gt_points])

        # Resize image if it's smaller than the window size
        aspect_ratio = image_width / image_height
        if image_height < window_size:
            new_height = window_size
            new_width = int(new_height * aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width
        if image_width < window_size:
            new_width = window_size
            new_height = int(new_width / aspect_ratio)
            image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
            image_height, image_width = new_height, new_width

        with torch.set_grad_enabled(False), autocast(device_type="cuda", enabled=amp):
            if sliding_window or (image_height * image_width) > max_input_size ** 2:
                pred_den_maps = sliding_window_predict(model, image, window_size, stride, max_num_windows)
            else:
                pred_den_maps = model(image)

            pred_counts.extend(pred_den_maps.sum(dim=(-1, -2, -3)).cpu().numpy().tolist())
    
    barrier(ddp)
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"

    if ddp:
        pred_counts, gt_counts = torch.tensor(pred_counts, device=device), torch.tensor(gt_counts, device=device)
        # Pad `pred_counts` and `gt_counts` to the same length across all processes.
        local_length = torch.tensor([len(pred_counts)], device=device)
        lengths = [torch.zeros_like(local_length) for _ in range(nprocs)]
        dist.all_gather(lengths, local_length)
        max_length = max([l.item() for l in lengths])
        padded_pred_counts, padded_gt_counts = torch.full((max_length,), float("nan"), device=device), torch.full((max_length,), float("nan"), device=device)
        padded_pred_counts[:len(pred_counts)], padded_gt_counts[:len(gt_counts)] = pred_counts, gt_counts
        gathered_pred_counts, gathered_gt_counts = [torch.zeros_like(padded_pred_counts) for _ in range(nprocs)], [torch.zeros_like(padded_gt_counts) for _ in range(nprocs)]
        dist.all_gather(gathered_pred_counts, padded_pred_counts)
        dist.all_gather(gathered_gt_counts, padded_gt_counts)
        # Concatenate predictions and ground truths from all processes and remove padding (nan values).
        pred_counts, gt_counts = torch.cat(gathered_pred_counts).cpu(), torch.cat(gathered_gt_counts).cpu()
        pred_counts, gt_counts = pred_counts[~torch.isnan(pred_counts)], gt_counts[~torch.isnan(gt_counts)]
        pred_counts, gt_counts = pred_counts.numpy(), gt_counts.numpy()

    else:
        pred_counts, gt_counts = np.array(pred_counts), np.array(gt_counts)

    # Log sample predictions to wandb
    if wandb_run is not None and local_rank == 0:
        model.train()  # temporarily switch to train mode to get intermediate outputs
        logged = 0
        num = random.random()
        i = 0
        for image, gt_points, gt_density in data_loader:
            if i < 80:
                i += 1
                continue
            num = random.random()
            if num < 0.5:
                continue
            
            if logged >= num_log_samples:
                break
            image = image.to(device)
            image_height, image_width = image.shape[-2:]

            # Resize if needed (same logic as main loop)
            aspect_ratio = image_width / image_height
            if image_height < window_size:
                new_height = window_size
                new_width = int(new_height * aspect_ratio)
                image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
                image_height, image_width = new_height, new_width
            if image_width < window_size:
                new_width = window_size
                new_height = int(new_width / aspect_ratio)
                image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
                image_height, image_width = new_height, new_width
            # Resize down if image exceeds max_input_size to avoid OOM
            if image_height * image_width > max_input_size ** 2:
                scale = max_input_size / max(image_height, image_width)
                new_height = int(image_height * scale)
                new_width = int(image_width * scale)
                image = F.interpolate(image, size=(new_height, new_width), mode="bicubic", align_corners=False)
                image_height, image_width = new_height, new_width
            # Interpolate GT density to match image dimensions if resized
            gt_density = gt_density.to(device)
            if gt_density.shape[-2:] != (image_height, image_width):
                gt_density = F.interpolate(gt_density, size=(image_height, image_width), mode="bilinear", align_corners=False)


            with torch.no_grad(), autocast(device_type="cuda", enabled=amp):
                outputs = model(image)

            if isinstance(outputs, tuple) and len(outputs) >= 4:
                # zero_inflated model returns (logit_pi, logit_maps, lambda_maps, den_maps)
                _, _, lambda_map, den_map = outputs[:4]
                lambda_map = F.interpolate(lambda_map, size=(image_height, image_width), mode="bilinear", align_corners=False)
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                # non-zero_inflated model returns (logit_maps, den_maps)
                _, den_map = outputs
                lambda_map = None
            else:
                # eval-mode fallback: model returned just den_map
                den_map = outputs if not isinstance(outputs, tuple) else outputs[0]
                lambda_map = None

            den_map = F.interpolate(den_map, size=(image_height, image_width), mode="bilinear", align_corners=False)

            for b in range(image.size(0)):
                if logged >= num_log_samples:
                    break
                # Denormalize the original image for visual overlay
                bg = image[b].cpu().numpy()
                bg = bg * _STD[:, None, None] + _MEAN[:, None, None]
                bg = np.clip(bg, 0, 1)
                bg = (bg.transpose(1, 2, 0) * 255).astype(np.uint8)

                den_img = den_map[b].squeeze().cpu().numpy()
                wandb_run.log({f"eval/pred_density_{logged}": _density_to_image(den_img, f"sample {logged} - Pred Density", background=bg)})
                if lambda_map is not None:
                    lam_img = lambda_map[b].squeeze().cpu().numpy()
                    wandb_run.log({f"eval/lambda_{logged}": _density_to_image(lam_img, f"sample {logged} - Lambda", background=bg)})
                gt_img = gt_density[b].squeeze().cpu().numpy()
                wandb_run.log({f"eval/gt_density_{logged}": _density_to_image(gt_img, f"sample {logged} - GT Density", background=bg)})
                logged += 1

        model.eval()  # restore eval mode


    torch.cuda.empty_cache()
    return calculate_errors(pred_counts, gt_counts)
