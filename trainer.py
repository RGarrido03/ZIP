import hashlib
import json
import os
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from torch import nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from evaluate import evaluate
from models import get_model
from train import train
from utils import (
    barrier,
    calc_bin_center,
    cleanup,
    get_config,
    get_dataloader,
    get_logger,
    get_loss_fn,
    get_optimizer,
    get_writer,
    init_seeds,
    load_checkpoint,
    log,
    save_checkpoint,
    setup,
    update_eval_result,
    update_train_result,
)

parser = ArgumentParser(description="Train an EBC model.")

# Parameters for model
parser.add_argument("--model_name", type=str, default="mamba3_micro", help="The name of the model to use.")
parser.add_argument("--block_size", type=int, default=None, choices=[7, 8, 14, 16, 28, 32], help="The block sizes for the model.")
parser.add_argument("--clip_weight_name", type=str, default=None, help="The weight name for CLIP models.")
parser.add_argument("--norm", type=str, default="none", choices=["none", "bn", "ln"], help="The normalization layer to use. 'none' means no normalization layer will be detected automatically, 'bn' means batch normalization, 'ln' means layer normalization.")
parser.add_argument("--act", type=str, default="none", choices=["none", "relu", "gelu"], help="The activation function to use. 'none' means no activation function will be detected automatically, 'relu' means ReLU, 'gelu' means GELU.")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=True, help="The dataset to train on.")
parser.add_argument("--in_memory_dataset", action="store_true", help="Load the dataset into memory. This will speed up training but requires more memory.")
parser.add_argument("--input_size", type=int, default=None, help="The size of the input image.")
parser.add_argument("--batch_size", type=int, default=None, help="The training batch size.")
parser.add_argument("--num_crops", type=int, default=None, help="The number of crops for multi-crop training.")
parser.add_argument("--aug_min_scale", type=float, default=None, help="The minimum scale for random scale augmentation.")
parser.add_argument("--aug_max_scale", type=float, default=None, help="The maximum scale for random scale augmentation.")
parser.add_argument("--aug_brightness", type=float, default=None, help="The brightness factor for random color jitter augmentation.")
parser.add_argument("--aug_contrast", type=float, default=None, help="The contrast factor for random color jitter augmentation.")
parser.add_argument("--aug_saturation", type=float, default=None, help="The saturation factor for random color jitter augmentation.")
parser.add_argument("--aug_hue", type=float, default=None, help="The hue factor for random color jitter augmentation.")
parser.add_argument("--aug_kernel_size", type=int, default=None, help="The kernel size for Gaussian blur augmentation.")
parser.add_argument("--aug_saltiness", type=float, default=None, help="The saltiness for pepper salt noise augmentation.")
parser.add_argument("--aug_spiciness", type=float, default=None, help="The spiciness for pepper salt noise augmentation.")
parser.add_argument("--aug_blur_prob", type=float, default=None, help="The probability for Gaussian blur augmentation.")
parser.add_argument("--sigma", type=float, default=None, help="Sigma for Gaussian density map generation. None means no smoothing (dot map).")

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--max_input_size", type=int, default=4096, help="The maximum size of the input image in evaluation. Images larger than this will be processed using sliding window by force to avoid OOM.")
parser.add_argument("--max_num_windows", type=int, default=64, help="The maximum number of windows to be simultaneously processed.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to a multiple of the input size.")

# Parameters for loss function
parser.add_argument("--reg_loss", type=str, default="zipnll", choices=["zipnll", "pnll", "dm", "msmae", "mae", "mse"], help="The regression loss function.")
parser.add_argument("--aux_loss", type=str, default="msmae", choices=["zipnll", "pnll", "dm", "msmae", "mae", "mse", "none"], help="The auxiliary loss function.")
parser.add_argument("--weight_cls", type=float, default=1.0, help="The weight for classification loss.")
parser.add_argument("--weight_reg", type=float, default=1.0, help="The weight for regression loss.")
parser.add_argument("--weight_aux", type=float, default=1.0, help="The weight for auxiliary loss.")
parser.add_argument("--numItermax", type=int, default=100, help="The maximum number of iterations for the OT/POT solver.")
parser.add_argument("--regularization", type=float, default=10.0, help="The regularization term for the OT/POT loss.")
parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4], help="The scales for multi-scale mae loss.")
parser.add_argument("--min_scale_weight", type=float, default=0.0, help="The minimum weight for multi-scale mae loss.")
parser.add_argument("--max_scale_weight", type=float, default=1.0, help="The maximum weight for multi-scale mae loss.")
parser.add_argument("--alpha", type=float, default=0.5, help="The alpha for multi-scale mae loss.")

# Parameters for optimizer
parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw", "radam"], help="The optimizer to use.")
parser.add_argument("--lr", type=float, default=None, help="The learning rate for untrained modules.")
parser.add_argument("--vpt_lr", type=float, default=None, help="The learning rate for the visual prompt tokens.")
parser.add_argument("--adapter_lr", type=float, default=None, help="The learning rate for the adapter layers. If None, it will be set to the same as lr.")
parser.add_argument("--lora_lr", type=float, default=None, help="The learning rate for the LoRA layers. If None, it will be set to the same as lr.")
parser.add_argument("--backbone_lr", type=float, default=None, help="The learning rate for the pretrained backbone.")
parser.add_argument("--weight_decay", type=float, default=None, help="The weight decay for untrained modules.")
parser.add_argument("--vpt_weight_decay", type=float, default=None, help="The weight decay for the visual prompt tokens.")
parser.add_argument("--adapter_weight_decay", type=float, default=None, help="The weight decay for the adapter layers. If None, it will be set to the same as weight_decay.")
parser.add_argument("--lora_weight_decay", type=float, default=None, help="The weight decay for the LoRA layers. If None, it will be set to the same as weight_decay.")
parser.add_argument("--backbone_weight_decay", type=float, default=None, help="The weight decay for the pretrained backbone.")

# Parameters for learning rate scheduler
parser.add_argument("--scheduler", type=str, default="cos_restarts", choices=["step", "cos", "cos_restarts"], help="The learning rate scheduler.")
parser.add_argument("--warmup_epochs", type=int, default=25, help="Number of epochs for warmup. The learning rate will linearly change from warmup_lr to lr.")
parser.add_argument("--warmup_lr", type=float, default=1e-5, help="Learning rate for warmup.")
parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate.")
# Step Decay parameters
parser.add_argument("--gamma", type=float, default=0.925, help="The decay factor for step scheduler.")
parser.add_argument("--step_size", type=int, default=20, help="The step size for step scheduler.")
# Cosine Annealing with Warm Restarts parameters
parser.add_argument("--T_0", type=int, default=5, help="Number of epochs for the first restart.")
parser.add_argument("--T_mult", type=int, default=2, help="A factor increases T_0 after a restart.")
# Cosine Annealing parameters
parser.add_argument("--T_max", type=int, default=20, help="The maximum number of epochs for the cosine annealing scheduler.")

# Parameters for training
parser.add_argument("--ckpt_dir_name", type=str, default=None, help="The name of the checkpoint folder.")
parser.add_argument("--ckpt_backbone", type=str, default=None, help="The name of the checkpoint for the backbone (.pth).")
parser.add_argument("--total_epochs", type=int, default=1300, help="Number of epochs to train.")
parser.add_argument("--eval_start", type=int, default=None, help="Start to evaluate after this number of epochs.")
parser.add_argument("--eval_freq", type=float, default=None, help="Evaluate every this number of epochs. If < 1, evaluate every this fraction of an epoch.")
parser.add_argument("--save_freq", type=int, default=50, help="Save checkpoint every this number of epochs. Could help reduce I/O.")
parser.add_argument("--save_best_k", type=int, default=5, help="Save the best k checkpoints.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for data loading.")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

# Parameters for wandb
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
parser.add_argument("--wandb_project", type=str, default="zip-ebc", help="Weights & Biases project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team/user).")
parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name. Defaults to the checkpoint dir name.")


def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    args.local_rank = local_rank
    print(f"Initialized successfully. Training with {nprocs} GPUs.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    print(f"Using device: {device}.")

    ddp = nprocs > 1

    # Define the bins and bin centers
    with open(os.path.join(current_dir, "configs", "bin_config.json"), "r") as f:
        bins = json.load(f)[args.dataset][str(args.block_size)]
    bins = [(float(b[0]), float(b[1])) for b in bins]

    with open(os.path.join(current_dir, "counts", f"{args.dataset}.json"), "r") as f:
        count_stats = json.load(f)[str(args.block_size)]
        count_stats = {int(k): int(v) for k, v in count_stats.items()}
        bin_centers, bin_counts = calc_bin_center(bins, count_stats)

    args.bins = bins
    args.bin_centers = bin_centers
    args.bin_counts = bin_counts

    model = get_model(
        model_name=args.model_name,
        model_info_path=os.path.join(args.ckpt_dir, "model_info.pth"),
        block_size=args.block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=args.reg_loss == "zipnll" or args.aux_loss == "zipnll",
        input_size=args.input_size,
        norm=args.norm,
        act=args.act,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = total_params - total_trainable_params

    grad_scaler = GradScaler(device=device) if args.amp else None

    loss_fn = get_loss_fn(args)
    optimizer, scheduler = get_optimizer(args, model)

    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(args, model, optimizer, scheduler, grad_scaler)

    if args.ckpt_backbone is not None:
        if not os.path.isfile(args.ckpt_backbone):
            raise FileNotFoundError(f"Backbone checkpoint file not found at {args.ckpt_backbone}")
        print(f"Loading backbone checkpoint from {args.ckpt_backbone}...")
        model.restore_backbone_checkpoint(args.ckpt_backbone)
        print("Backbone checkpoint loaded successfully.")

    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model

    if local_rank == 0:
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train.log"))
        logger.info(get_config(vars(args), mute=False))
        logger.info(f"Total parameters: {total_params:,}\nTrainable parameters: {total_trainable_params:,}\nNon-trainable parameters: {total_nontrainable_params:,}\n")

        wandb_run = None
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or os.path.basename(args.ckpt_dir),
                config=vars(args),
                dir=args.ckpt_dir,
            )
    else:
        writer = None
        logger = None
        wandb_run = None
    train_loader, sampler = get_dataloader(args, split="train")
    val_loader = get_dataloader(args, split="val")

    for epoch in range(start_epoch, args.total_epochs + 1):  # start from 1
        if local_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            message = f"\tlr: {lr:.3e}"
            log(logger, epoch, args.total_epochs, message=message)
            if wandb_run is not None:
                wandb_run.log({"train/lr": lr})

        if sampler is not None:
            sampler.set_epoch(epoch)

        if args.eval_freq < 1:
            eval_model = epoch >= args.eval_start

            if eval_model:
                model, optimizer, grad_scaler, loss_info, curr_val_scores, curr_weights = train(
                    model=model,
                    data_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    device=device,
                    rank=local_rank,
                    nprocs=nprocs,
                    eval_data_loader=val_loader,
                    eval_freq=args.eval_freq,
                    sliding_window=args.sliding_window,
                    max_input_size=args.max_input_size,
                    window_size=args.input_size,
                    stride=args.stride,
                    max_num_windows=args.max_num_windows,
                    wandb_run=wandb_run,
                )
                scheduler.step()
                barrier(ddp)
            
            else:
                model, optimizer, grad_scaler, loss_info, _, _ = train(
                    model=model,
                    data_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    device=device,
                    rank=local_rank,
                    nprocs=nprocs,
                    wandb_run=wandb_run,
                )
                scheduler.step()
                barrier(ddp)

        else:
            model, optimizer, grad_scaler, loss_info, _, _ = train(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                device=device,
                rank=local_rank,
                nprocs=nprocs,
                wandb_run=wandb_run,
            )

            scheduler.step()
            barrier(ddp)

            print(f"Evaluating at epoch {epoch} with args eval_freq {args.eval_freq} and eval_start {args.eval_start}. Calculated eval_model: {(epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)}.")
            eval_model = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
            if eval_model:
                curr_val_scores = evaluate(
                    model=model,
                    data_loader=val_loader,
                    sliding_window=args.sliding_window,
                    max_input_size=args.max_input_size,
                    window_size=args.input_size,
                    stride=args.stride,
                    max_num_windows=args.max_num_windows,
                    device=device,
                    amp=args.amp,
                    local_rank=local_rank,
                    nprocs=nprocs,
                    wandb_run=wandb_run,
                )

                state_dict = deepcopy(model.module.state_dict() if ddp else model.state_dict())
                curr_weights = {k: state_dict for k in curr_val_scores.keys()}  # copy the state_dict                    

        if local_rank == 0:
            update_train_result(epoch, loss_info, writer, wandb_run=wandb_run)
            log(logger, None, None, loss_info=loss_info, message="\n" * 2 if not eval_model else None)

            if eval_model:
                hist_val_scores, best_val_scores = update_eval_result(
                    epoch=epoch,
                    curr_scores=curr_val_scores,
                    hist_scores=hist_val_scores,
                    best_scores=best_val_scores,
                    model_info={"config": model.module.config if ddp else model.config, "weights": curr_weights},
                    writer=writer,
                    ckpt_dir=args.ckpt_dir,
                    wandb_run=wandb_run,
                )
        
                log(logger, None, None, None, curr_val_scores, best_val_scores, message="\n" * 3)

        if local_rank == 0 and (epoch % args.save_freq == 0):
            save_checkpoint(
                epoch + 1,
                model.module.state_dict() if ddp else model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict() if scheduler is not None else None,
                grad_scaler.state_dict() if grad_scaler is not None else None,
                loss_info,
                hist_val_scores,
                best_val_scores,
                args.ckpt_dir,
            )

        barrier(ddp)

    if local_rank == 0:
        if wandb_run is not None:
            wandb_run.finish()
        writer.close()
        print("Training completed. Best scores:")
        for k in best_val_scores.keys():
            scores = " ".join([f"{best_val_scores[k][i]:.4f};" for i in range(len(best_val_scores[k]))])
            print(f"    {k}: {scores}. \t Mean: {np.mean(best_val_scores[k]):.4f}")

    cleanup(ddp)


def main():
    args = parser.parse_args()
    args.dataset = standardize_dataset_name(args.dataset)

    dataset_config_path = os.path.join(current_dir, "configs", f"{args.dataset}.yaml")
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)
    for k, v in dataset_config.items():
        if k in vars(args) and vars(args)[k] is None:
            vars(args)[k] = v
    
    # Sliding window prediction will be used if args.sliding_window is True, or when the image size is larger than args.max_input_size
    args.stride = args.stride or args.input_size

    if args.reg_loss != "dm" and args.aux_loss != "dm":
        args.numItermax = None
        args.regularization = None
    
    if args.reg_loss != "msmae" and args.aux_loss != "msmae":
        args.scales = None
        args.min_scale_weight = None
        args.max_scale_weight = None
        args.alpha = None
    else:
        assert args.max_scale_weight >= args.min_scale_weight >= 0, f"Expected max_scale_weight to be greater than or equal to min_scale_weight, got {args.min_scale_weight} and {args.max_scale_weight}"
        assert 1 >= args.alpha > 0, f"Expected alpha to be between 0 and 1, got {args.alpha}"
        
    if args.scheduler == "step":
        args.T_0 = None
        args.T_mult = None
        args.T_max = None
    elif args.scheduler == "cos":
        args.step_size = None
        args.gamma = None
        args.T_0 = None
        args.T_mult = None
    else:
        args.step_size = None
        args.gamma = None
        args.T_max = None
    
    args.nprocs = torch.cuda.device_count()
    args.batch_size = int(args.batch_size / args.nprocs)
    args.num_workers = int(args.num_workers / args.nprocs)
    
    if args.ckpt_dir_name is None:
        hyperparams_dict = (vars(args)).copy()
        hyperparams_dict.pop("save_freq")
        hyperparams_dict.pop("save_best_k")
        hyperparams_dict.pop("local_rank")
        hyperparams_dict.pop("num_workers")
        hyperparams_dict.pop("nprocs")
        hyperparams_dict.pop("ckpt_dir_name")
        hyperparams_dict = json.dumps(hyperparams_dict, sort_keys=True)
        args.hash = hashlib.sha256(hyperparams_dict.encode("utf-8")).hexdigest()

        ckpt_dir_name = f"zip_{args.model_name}_{args.block_size}_"
        ckpt_dir_name += f"{args.weight_cls}+{args.weight_reg}x{(args.reg_loss)}+{args.weight_aux}{(args.aux_loss)}_"
        ckpt_dir_name += f"{args.optimizer}_{args.scheduler}_{args.hash[:8]}"
    else:
        ckpt_dir_name = args.ckpt_dir_name

    args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, ckpt_dir_name)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        if args.in_memory_dataset:
            print("In-memory dataset is not supported for distributed training. Using disk-based dataset instead.")
            args.in_memory_dataset = False
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()