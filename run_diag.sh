#!/bin/bash
# Diagnostics after the fair-pipeline ablation showed S_min hurts.
#  (A) theta+alpha WITHOUT depth pruning — do grouping/coupling help on a
#      healthy baseline (S_min=0, no blocks removed)?
#  (B) S_min sweep — does removing FEWER blocks hurt less?  Pure depth
#      (theta=1, alpha=0) so the depth cost is isolated.
#
# Known anchors (already run, fair 2.5G / 5ep pipeline):
#   iso_baseline  S_min=0   (0 blocks)  ft=0.6914
#   plus_smin     S_min=0.40 (3 blocks) ft=0.6766
#
# Sensitivities: B4=.361 B5=.372 B3=.376 B6=.410 B7=.427 (then .47+)
#   S_min 0.365 -> remove {4}            (1 block)
#   S_min 0.374 -> remove {4,5}          (2 blocks)
#   S_min 0.380 -> remove {3,4,5}        (3 blocks, == 0.40)
#   S_min 0.415 -> remove {3,4,5,6}      (4 blocks)
#   S_min 0.430 -> remove {3,4,5,6,7}    (5 blocks)

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=5
HEAD_SCALE=0.2
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4
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
  --output_dir    ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN} theta=${THETA} alpha=${ALPHA}"
}

# (A) theta + alpha, no depth pruning
sbatch_run "theta_alpha_nodepth"  0.00  0.025  0.3

# (B) S_min sweep, pure depth (theta=1, alpha=0)
sbatch_run "smin_1blk"  0.365  1.0  0.0   # remove {4}
sbatch_run "smin_2blk"  0.374  1.0  0.0   # remove {4,5}
sbatch_run "smin_4blk"  0.415  1.0  0.0   # remove {3,4,5,6}
sbatch_run "smin_5blk"  0.430  1.0  0.0   # remove {3,4,5,6,7}

echo ""
echo "Anchors: iso_baseline(0 blk)=0.6914  plus_smin(3 blk)=0.6766"
echo "Check: squeue -u \$USER ; ls ${BASE_DIR}/{theta_alpha_nodepth,smin_*}/results.json"
