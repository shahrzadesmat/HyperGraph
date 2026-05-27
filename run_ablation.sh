#!/bin/bash
# Submit all four ablation runs as SLURM jobs.
# Usage: bash run_ablation.sh
# Edit DATA_PATH and TARGET_MACS before running.

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5                    # GigaOps target (DeiT-Small baseline ≈ 4.6G, ~46% reduction)
MODEL="deit_small_patch16_224"
EPOCHS=20
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

sbatch_run() {
    local name=$1
    local S_MIN=$2
    local THETA=$3
    local ALPHA=$4
    local outdir="${BASE_DIR}/${name}"

    mkdir -p "$outdir"

    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=6:00:00 \
           --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" \
           --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
cd /work/hdd/bdjd/hypergraph_pruning
PYTHONUNBUFFERED=1 python run.py \
  --model         ${MODEL} \
  --data_path     ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --S_min         ${S_MIN} \
  --theta         ${THETA} \
  --alpha         ${ALPHA} \
  --epochs        ${EPOCHS} \
  --output_dir    ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN}  theta=${THETA}  alpha=${ALPHA}"
}

# Ablation ladder — add one novel component at a time
sbatch_run "iso_baseline"  0.00  1.0  0.0   # isomorphic pruning (no novel params)
sbatch_run "plus_smin"     0.15  1.0  0.0   # + depth pruning via sensitivity
sbatch_run "plus_theta"    0.15  0.3  0.0   # + per-group width ratios
sbatch_run "plus_alpha"    0.15  0.3  0.3   # + functional-coupling boost (full method)

echo ""
echo "Check progress: squeue -u \$USER"
echo "Results:        ls ${BASE_DIR}/*/results.json"
