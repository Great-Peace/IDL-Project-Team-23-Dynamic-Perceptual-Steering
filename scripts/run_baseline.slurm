#!/bin/bash
#SBATCH --job-name=dps_baseline
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100-32g:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@andrew.cmu.edu

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

module load cuda/11.8
module load anaconda3
conda activate dps_env

export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

cd "$PROJECT_DIR"
mkdir -p logs results "$HF_HOME"

echo "Running Phase 2 - Baseline Evaluation..."
python experiments/phase2_baseline.py --config configs/config.yaml

echo "Job finished: $(date)"
