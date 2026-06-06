#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodes=1
#SBATCH --nodelist=xbox         
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=zip
#SBATCH --output=runs/zip_%j.out
#SBATCH --error=runs/zip_%j.err

# Removemos as exportações de MASTER_ADDR, MASTER_PORT e NCCL_SOCKET_IFNAME
# pois não haverá comunicação distribuída.


source .venv/bin/activate


export WANDB_ENTITY='andreribeiro87-universidade-de-aveiro'
export WANDB_API_KEY='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'
export WANDB_PROJECT=cls_vssd_nc

# export CUDA_VISIBLE_DEVICES=0

python trainer.py \
    --dataset sha --input_size 224 --block_size 16 --sliding_window --warmup_lr 1e-3 \
    --num_workers 4 --wandb --eval_freq 50 --eval_start 0 \
    --ckpt_backbone "/slurm_shared/home/andrepedroribeiro@av.it.pt/mamba3-caa/work_dirs/cls_vssd_nc_micro_ddp/last.pth" 2>&1
