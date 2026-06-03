#!/bin/bash
# Confirm the positive diagnostic result with the combined-best config and
# multi-seed runs.
#
# Diagnostic findings (seed 42, fair 2.5G / 5ep pipeline):
#   VainF                     0.6880
#   iso_baseline (params off) 0.6914
#   theta+alpha, no depth     0.7010   ← width allocation wins
#   S_min sweet spot          1 block (0.6950 > 0 blocks)
#
# Here:
#  (A) combined_best  = 1-block depth (S_min=0.365 -> remove B4) + theta + alpha
#      3 seeds — does combining the 1-block depth win with the width win beat 0.701?
#  (B) theta_alpha_nodepth seeds 99,200 — confirm 0.701 is robust (seed42 done).

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=5
HEAD_SCALE=0.2
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4 SEED=$5
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    rm -f "${outdir}/results.json"
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
  --head_scale    ${HEAD_SCALE} \
  --epochs        ${EPOCHS} \
  --seed          ${SEED} \
  --output_dir    ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN} theta=${THETA} alpha=${ALPHA} seed=${SEED}"
}

# (A) combined best: 1-block depth + theta + alpha, 3 seeds
sbatch_run "combined_best_s42"   0.365  0.025  0.3   42
sbatch_run "combined_best_s99"   0.365  0.025  0.3   99
sbatch_run "combined_best_s200"  0.365  0.025  0.3   200

# (B) theta+alpha no-depth, extra seeds (seed42 already = 0.7010)
sbatch_run "theta_alpha_nodepth_s99"   0.00  0.025  0.3   99
sbatch_run "theta_alpha_nodepth_s200"  0.00  0.025  0.3   200

echo ""
echo "Anchors (seed42): VainF=0.6880  iso_baseline=0.6914  theta+alpha_nodepth=0.7010"
