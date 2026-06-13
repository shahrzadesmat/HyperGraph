#!/bin/bash
# Re-run the entropy-hybrid lambda sweep WITH normalized bypass sensitivity.
# Writes to results/norm_* so the old (un-normalized) runs stay intact for
# side-by-side comparison.
# Usage: bash run_ablation_norm.sh

set -e

DATA_PATH="/work/hdd/bfxa/dshah13/data/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=20
BASE_DIR="/work/hdd/bfxa/dshah13/HyperGraph/results"
CODE_DIR="/work/hdd/bfxa/dshah13/HyperGraph"
VENV="/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv/bin/python"
HEAD_SCALE=0.2

# Args: name S_MIN THETA ALPHA LAM
sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4 LAM=${5:-1.0}
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    rm -f "${outdir}/results.json"

    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=12:00:00 \
           --account=bfxa-delta-gpu \
           --output="${outdir}/slurm_%j.out" \
           --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
cd ${CODE_DIR}
PYTHONUNBUFFERED=1 ${VENV} run.py \
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

echo "=== Normalized-bypass entropy-hybrid lambda sweep ==="
# +S_min level: isolate Idea #1 with normalized blend.
sbatch_run "norm_lam07"  0.40  1.0   0.0   0.7
sbatch_run "norm_lam05"  0.40  1.0   0.0   0.5
sbatch_run "norm_lam03"  0.40  1.0   0.0   0.3
sbatch_run "norm_lam00"  0.40  1.0   0.0   0.0
# full method + best-looking blend
sbatch_run "norm_full07" 0.40  0.05  0.3   0.7

echo ""
echo "Compare to un-normalized: results/smin_lam0{7,5,3,0}, results/full_lam07"
echo "Check progress: squeue -u \$USER"
