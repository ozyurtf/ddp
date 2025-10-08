#!/bin/bash
#SBATCH --job-name=ddp_test       # Job name
#SBATCH --output=ddp_%j.out       # Standard output (%j = job ID)
#SBATCH --error=ddp_%j.err        # Standard error
#SBATCH --ntasks=2                # Number of processes (match number of GPUs)
#SBATCH --gres=gpu:2              # Number of GPUs
#SBATCH --time=00:05:00           # Max run time (HH:MM:SS)
#SBATCH --cpus-per-task=2         # CPUs per process 
#SBATCH --mem=8GB                 # Memory per node 

# Activate your environment
source /gpfs/u/home/AICD/AICDnhns/.local/share/enroot/colosseum/opt/conda/etc/profile.d/conda.sh
conda activate ddp

# Run DDP using torchrun
torchrun --nproc_per_node=2 elastic_ddp.py