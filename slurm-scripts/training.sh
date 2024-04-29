#!/bin/bash

#SBATCH --cpus-per-task=24  # Adjust to leverage parallel computation
#SBATCH -p gpu
#SBATCH --mem=128g
#SBATCH -t 12:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=aryonna@email.unc.edu

# Activate your virtual environment
source /nas/longleaf/home/aryonna/488FinalProject/final_proj_env/bin/activate

sar -u -r 60 > cpu_mem_usage_$SLURM_JOB_ID.txt &
# Run your Python script
python /nas/longleaf/home/aryonna/488FinalProject/training/brand_perception_layer.py

wait

echo "Job ended at `date`"