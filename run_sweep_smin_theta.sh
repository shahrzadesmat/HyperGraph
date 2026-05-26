#!/bin/bash
# Sweep 1: S_min sensitivity   (alpha=0, theta=1.0 — isolates depth pruning alone)
# Sweep 2: theta sensitivity   (alpha=0, best S_min from sweep 1)
#
# Run sweep 1 first, pick the best S_min, then edit BEST_SMIN and run sweep 2.
# Usage: bash run_sweep_smin_theta.sh [smin|theta]

set -e

DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
TARGET_MACS=2.5
MODEL="deit_small_patch16_224"
EPOCHS=20
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

# ---- edit after sweep 1 completes ----
BEST_SMIN=0.15    # placeholder; update from sweep_smin results

sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4 ETHR=$5
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=24:00:00 \
           --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
cd /work/hdd/bdjd/hypergraph_pruning
PYTHONUNBUFFERED=1 python run.py \
  --model ${MODEL} --data_path ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --S_min ${S_MIN} --theta ${THETA} --alpha ${ALPHA} \
  --edge_threshold ${ETHR} --epochs ${EPOCHS} --output_dir ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN} theta=${THETA} alpha=${ALPHA}"
}

MODE=${1:-smin}

if [ "$MODE" = "smin" ]; then
    echo "=== Sweep 1: S_min (theta=1.0, alpha=0.0) ==="
    # theta=1.0 = isomorphic grouping, so this isolates S_min alone
    sbatch_run "sweep_smin_0.00"  0.00  1.0  0.0  0.3
    sbatch_run "sweep_smin_0.05"  0.05  1.0  0.0  0.3
    sbatch_run "sweep_smin_0.10"  0.10  1.0  0.0  0.3
    sbatch_run "sweep_smin_0.15"  0.15  1.0  0.0  0.3
    sbatch_run "sweep_smin_0.20"  0.20  1.0  0.0  0.3
    sbatch_run "sweep_smin_0.25"  0.25  1.0  0.0  0.3
    echo ""
    echo "After these finish, pick best S_min then run: bash run_sweep_smin_theta.sh theta"

elif [ "$MODE" = "theta" ]; then
    echo "=== Sweep 2: theta (S_min=${BEST_SMIN}, alpha=0.0) ==="
    # theta=1.0 is the isomorphic baseline; theta=0.0 gives each block its own ratio
    sbatch_run "sweep_theta_0.0"  ${BEST_SMIN}  0.0  0.0  0.3   # max heterogeneity
    sbatch_run "sweep_theta_0.1"  ${BEST_SMIN}  0.1  0.0  0.3
    sbatch_run "sweep_theta_0.2"  ${BEST_SMIN}  0.2  0.0  0.3
    sbatch_run "sweep_theta_0.3"  ${BEST_SMIN}  0.3  0.0  0.3
    sbatch_run "sweep_theta_0.5"  ${BEST_SMIN}  0.5  0.0  0.3
    sbatch_run "sweep_theta_1.0"  ${BEST_SMIN}  1.0  0.0  0.3   # isomorphic (no grouping)

else
    echo "Usage: bash run_sweep_smin_theta.sh [smin|theta]"
    exit 1
fi

echo ""
echo "Collect results: python collect_results.py ${BASE_DIR}/sweep_${MODE}_*"
