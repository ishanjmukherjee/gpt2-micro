#!/bin/bash
#SBATCH --job-name=gpt2-micro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00

# initialize conda in batch
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gpt2-micro

# make sure slurm sets CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# launch distributed training via Accelerate
srun accelerate launch \
    --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    train.py
