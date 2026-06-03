#!/bin/bash
#SBATCH --job-name=hg_iso_baseline
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=bdjd-delta-gpu
#
# Isomorphic pruning baseline — arxiv 2407.04616 approach.
# Uniform Taylor-importance pruning ratio across all attention + MLP blocks.
# No S_min, no theta, no alpha.
# Compare its finetuned_acc against run_ablation.sh "iso_baseline" run
# to verify our hypergraph S_min=0/theta=1/alpha=0 reproduces it exactly.
#
# Usage: sbatch run_iso_baseline.sh

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=5
OUTDIR="/work/hdd/bdjd/hypergraph_pruning/results/iso_paper_baseline"

mkdir -p "${OUTDIR}"

export PATH=/work/hdd/bdjd/miniconda3/bin:$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh

cd /work/hdd/bdjd/hypergraph_pruning

PYTHONUNBUFFERED=1 python run_iso_baseline.py \
  --model         ${MODEL} \
  --data_path     ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --epochs        ${EPOCHS} \
  --output_dir    ${OUTDIR}
