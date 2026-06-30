#!/bin/bash
#SBATCH --partition=gpuPartition
#SBATCH --nodes=1
#SBATCH --nodelist=xbox
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15
#SBATCH --threads-per-core=2
#SBATCH --job-name=run_queue
#SBATCH --output=runs/run_queue_%j.out
#SBATCH --error=runs/run_queue_%j.err
#SBATCH --mem=60G

# Removemos as exportações de MASTER_ADDR, MASTER_PORT e NCCL_SOCKET_IFNAME
# pois não haverá comunicação distribuída.


source .venv/bin/activate


# export ARCH=${ARCH:-mamba3_micro}
export ARCH=${ARCH:-mamba3_pico}
export ARCH_MIN=$(echo $ARCH | sed 's/mamba3_//')
export WANDB_ENTITY='andreribeiro87-universidade-de-aveiro'
export WANDB_API_KEY='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'
export WANDB_PROJECT=zip-ebc

# export CUDA_VISIBLE_DEVICES=0
uv run python run_queue_wait.py \
    --video output_42_full.mp4 \
    --model-info checkpoints/sha/zip_mamba3_micro_16_1.0+1.0xzipnll+1.0msmae_adam_cos_restarts_fece5697/ckpt.pth \
    --model-name mamba3_micro --strategy morphology \
    --output annotated_output_42.mp4 --progress-every 100 2>&1