#!/bin/bash
#SBATCH --job-name=dps_adversarial
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100-32g:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/adversarial_%j.out
#SBATCH --error=logs/adversarial_%j.err
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

echo "Running Phase 4 - Adversarial Steering..."
python experiments/phase4_adversarial.py --config configs/config.yaml

echo "Job finished: $(date)"
