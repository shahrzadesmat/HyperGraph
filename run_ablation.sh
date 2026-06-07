#!/bin/bash
# Submit all ablation runs as SLURM jobs.
# Usage: bash run_ablation.sh
# Edit DATA_PATH and TARGET_MACS before running.

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5                    # GigaOps target (DeiT-Small baseline ≈ 4.6G, ~46% reduction)
MODEL="deit_small_patch16_224"
EPOCHS=5
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

HEAD_SCALE=0.2   # r_attn_base = r_mlp_base * HEAD_SCALE (matches VainF DeiT-Small)

sbatch_run() {
    local name=$1
    local S_MIN=$2
    local THETA=$3
    local ALPHA=$4
    local LAM=${5:-1.0}
    local outdir="${BASE_DIR}/${name}"

    mkdir -p "$outdir"
    rm -f "${outdir}/results.json"   # clear stale results

    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=03:00:00 \
           --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" \
           --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
cd /work/hdd/bdjd/hypergraph_pruning
PYTHONUNBUFFERED=1 python -u run.py \
  --model         ${MODEL} \
  --data_path     ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --S_min         ${S_MIN} \
  --theta         ${THETA} \
  --alpha         ${ALPHA} \
  --lam           ${LAM} \
  --head_scale    ${HEAD_SCALE} \
  --epochs        ${EPOCHS} \
  --output_dir    ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN}  theta=${THETA}  alpha=${ALPHA}  lam=${LAM}"
}

# Ablation ladder — add one novel component at a time.
# S_min calibrated from sensitivity data: min block sensitivity=0.361,
# so S_min=0.40 removes the 3 least sensitive blocks (3,4,5).
# theta calibrated from Taylor score spread: scores nearly uniform,
# so theta=0.05 is needed to create meaningful groups.
sbatch_run "iso_baseline"  0.00  1.0   0.0   # isomorphic pruning (no novel params)
sbatch_run "plus_smin"     0.40  1.0   0.0   # + depth pruning (removes blocks 3,4,5)
sbatch_run "plus_theta"    0.40  0.05  0.0   # + per-group width ratios
sbatch_run "plus_alpha"    0.40  0.05  0.3   # + functional-coupling boost (full method)

# ---------------------------------------------------------------------------
# Phase 2 — Idea #1: Entropy hybrid lambda sweep
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: Entropy-hybrid lambda sweep ==="

# Sweep lam at the +S_min level to isolate Idea #1 contribution.
# lam=1.0 is identical to plus_smin (reused, no new job needed).
sbatch_run "smin_lam07"    0.40  1.0   0.0   0.7   # 70% bypass + 30% entropy
sbatch_run "smin_lam05"    0.40  1.0   0.0   0.5   # 50/50 blend
sbatch_run "smin_lam03"    0.40  1.0   0.0   0.3   # 30% bypass + 70% entropy
sbatch_run "smin_lam00"    0.40  1.0   0.0   0.0   # pure entropy (data-free baseline)

# Best lambda carried through to the full method row
sbatch_run "full_lam07"    0.40  0.05  0.3   0.7   # full method + best entropy blend

echo ""
echo "Check progress: squeue -u \$USER"
echo "Results:        ls ${BASE_DIR}/*/results.json"
echo ""
echo "Ablation ladder:"
echo "  iso_baseline   → S_min=0,    theta=1.0,  alpha=0,   lam=1.0  (VainF baseline)"
echo "  plus_smin      → S_min=0.40, theta=1.0,  alpha=0,   lam=1.0"
echo "  plus_theta     → S_min=0.40, theta=0.05, alpha=0,   lam=1.0"
echo "  plus_alpha     → S_min=0.40, theta=0.05, alpha=0.3, lam=1.0  (original full)"
echo "  --- Idea #1: Entropy hybrid ---"
echo "  smin_lam07     → S_min=0.40, theta=1.0,  alpha=0,   lam=0.7"
echo "  smin_lam05     → S_min=0.40, theta=1.0,  alpha=0,   lam=0.5"
echo "  smin_lam03     → S_min=0.40, theta=1.0,  alpha=0,   lam=0.3"
echo "  smin_lam00     → S_min=0.40, theta=1.0,  alpha=0,   lam=0.0  (pure entropy)"
echo "  full_lam07     → S_min=0.40, theta=0.05, alpha=0.3, lam=0.7  (best result)"
