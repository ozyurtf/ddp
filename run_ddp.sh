#!/bin/bash
#SBATCH --job-name=ddp_test       # Job name
#SBATCH --output=ddp_%j.out       # Standard output (%j = job ID)
#SBATCH --error=ddp_%j.err        # Standard error
#SBATCH --ntasks=2                # Number of processes (match number of GPUs)
#SBATCH --gres=gpu:2              # Number of GPUs
#SBATCH --time=01:00:00           # Max run time (HH:MM:SS)
#SBATCH --cpus-per-task=4         # CPUs per process 
#SBATCH --mem=16G                 # Memory per node 

module load cuda/12.2             # Load CUDA module 
module load python/3.10           # Load Python module 

# Activate your environment
source /gpfs/u/home/AICD/AICDnhns/.local/share/enroot/colosseum/opt/conda/etc/profile.d/conda.sh
conda activate ddp

# Run DDP using torchrun
torchrun --nproc_per_node=2 elastic_ddp.py