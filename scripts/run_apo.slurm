#!/bin/bash
#SBATCH --job-name=dps_apo
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100-32g:2
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/apo_%j.out
#SBATCH --error=logs/apo_%j.err
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

echo "Running Phase 3B - APO..."
python experiments/phase3b_apo.py --config configs/config.yaml

echo "Job finished: $(date)"
