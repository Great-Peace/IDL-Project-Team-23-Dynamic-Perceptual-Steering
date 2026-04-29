#!/bin/bash
# ============================================================
# Environment Setup Script for PSC Bridges-2
# Run this once before starting experiments
# Usage: bash scripts/setup_environment.sh
# ============================================================

set -euo pipefail

echo "Setting up Dynamic Perceptual Steering environment..."
echo "This installs dependencies for PSC Bridges-2."

module load cuda/11.8
module load anaconda3

conda create -n dps_env python=3.10 -y
conda activate dps_env

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Set or override HF_HOME before running this script if you want a custom cache path.
HF_CACHE="${HF_HOME:-/ocean/projects/$USER/hf_cache}"
mkdir -p "$HF_CACHE"
echo "export HF_HOME=$HF_CACHE" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$HF_CACHE" >> ~/.bashrc

echo ""
echo "Environment setup complete."
echo ""
echo "Next steps:"
echo "1. Confirm HF_HOME points to the storage location you want on PSC"
echo "2. Build or copy the datasets described in README.md and DATASET_CARD.md"
echo "3. Run: sbatch scripts/run_baseline.slurm"
