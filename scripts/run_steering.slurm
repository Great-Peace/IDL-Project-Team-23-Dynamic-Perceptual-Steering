#!/bin/bash
#SBATCH --job-name=dps_steering
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100-32g:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/steering_%j.out
#SBATCH --error=logs/steering_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@andrew.cmu.edu

set -euo pipefail

echo "Job started: $(date)"
module load cuda/11.8
module load anaconda3
conda activate dps_env

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

cd "$PROJECT_DIR"
mkdir -p logs "$HF_HOME"

echo "Running Phase 3A - Manual Steering..."
python experiments/phase3a_manual_steering.py --config configs/config.yaml

echo "Job finished: $(date)"
