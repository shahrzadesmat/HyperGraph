#!/bin/bash
# Sweep 3: alpha × edge_threshold (2D grid)
# Fix best S_min and theta from sweeps 1 & 2 before running.
#
# Rationale: alpha's effect depends on graph sparsity (edge_threshold).
#   Dense graph (low threshold) → many edges → boost diluted across all blocks
#   Sparse graph (high threshold) → few edges → each edge carries stronger signal
#
# Usage: bash run_sweep_alpha.sh

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=20
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

# ---- edit these after sweeps 1 & 2 ----
BEST_SMIN=0.15
BEST_THETA=0.3

sbatch_run() {
    local name=$1 ALPHA=$2 ETHR=$3
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=6:00:00 \
           --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
cd /work/hdd/bdjd/hypergraph_pruning
PYTHONUNBUFFERED=1 python run.py \
  --model ${MODEL} --data_path ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --S_min ${BEST_SMIN} --theta ${BEST_THETA} \
  --alpha ${ALPHA} --edge_threshold ${ETHR} \
  --epochs ${EPOCHS} --output_dir ${outdir}
"
    echo "[Submitted] ${name}  alpha=${ALPHA}  edge_threshold=${ETHR}"
}

echo "=== Sweep 3: alpha × edge_threshold (S_min=${BEST_SMIN}, theta=${BEST_THETA}) ==="
echo ""
echo "Hypothesis: alpha helps MORE with sparse graphs (high threshold)."
echo "alpha=0.0 row is the theta-only baseline (same for all thresholds)."
echo ""

# alpha=0 baseline (one per threshold, but result is identical since edges don't matter)
sbatch_run "sweep_alpha_a0.0_t0.3"   0.0  0.3

# Sparse graph (high threshold): only very similar-importance block pairs couple
# This is where alpha is expected to have the most discriminative effect
for ALPHA in 0.2 0.5 1.0 2.0; do
    sbatch_run "sweep_alpha_a${ALPHA}_t0.3"   ${ALPHA}  0.3
    sbatch_run "sweep_alpha_a${ALPHA}_t0.5"   ${ALPHA}  0.5
    sbatch_run "sweep_alpha_a${ALPHA}_t0.7"   ${ALPHA}  0.7
done

echo ""
echo "Total: 13 jobs"
echo ""
echo "After completion, look for the region where:"
echo "  - High threshold (0.5-0.7) + some alpha > alpha=0 baseline"
echo "  - That gap is alpha's true contribution with a well-calibrated graph"
echo ""
echo "Collect: python collect_results.py ${BASE_DIR}/sweep_alpha_*"
