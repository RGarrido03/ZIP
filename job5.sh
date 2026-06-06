#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodes=1
#SBATCH --nodelist=sega         
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=vssd_nc_cls_imagenet1k_ddp
#SBATCH --output=runs/zip_%j.out
#SBATCH --error=runs/zip_%j.err

# Removemos as exportações de MASTER_ADDR, MASTER_PORT e NCCL_SOCKET_IFNAME
# pois não haverá comunicação distribuída.

export PYTHONPATH="/slurm_shared/home/andrepedroribeiro@av.it.pt/mamba3-caa:$PYTHONPATH"

source /slurm_shared/home/andrepedroribeiro@av.it.pt/mamba3-caa/.venv/bin/activate


export WANDB_ENTITY='andreribeiro87-universidade-de-aveiro'
export WANDB_API_KEY='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'
export WANDB_PROJECT=cls_vssd_nc

# export CUDA_VISIBLE_DEVICES=0

uv run trainer.py \
    --dataset sha --input_size 224 --block_size 16 --sliding_window --warmup_lr 1e-3 \
    --num_workers 4 --wandb --eval_freq 1 --eval_start 0  2>&1
