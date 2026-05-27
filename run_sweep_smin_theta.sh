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
BEST_SMIN=0.40    # placeholder; update from sweep_smin results

sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4 ETHR=$5
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=12:00:00 \
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
  --S_min ${S_MIN} --theta ${THETA} --alpha ${ALPHA} \
  --edge_threshold ${ETHR} --epochs ${EPOCHS} --output_dir ${outdir}
"
    echo "[Submitted] ${name}  S_min=${S_MIN} theta=${THETA} alpha=${ALPHA}"
}

MODE=${1:-smin}

if [ "$MODE" = "smin" ]; then
    echo "=== Sweep 1: S_min (theta=1.0, alpha=0.0) ==="
    # Sensitivity range from calibration: min=0.361 (block4), so S_min must
    # exceed 0.361 to trigger depth pruning at all.
    # 0.37 → removes block 4 only (1 block)
    # 0.40 → removes blocks 3,4,5 (3 blocks)
    # 0.43 → removes blocks 3,4,5,6 (4 blocks)
    # 0.46 → removes blocks 3,4,5,6,7 (5 blocks)
    # 0.50 → removes blocks 2,3,4,5,6,7,8,11 (8 blocks — aggressive)
    sbatch_run "sweep_smin_0.00"  0.00  1.0  0.0  0.3   # iso baseline (no depth pruning)
    sbatch_run "sweep_smin_0.37"  0.37  1.0  0.0  0.3   # removes 1 block
    sbatch_run "sweep_smin_0.40"  0.40  1.0  0.0  0.3   # removes 3 blocks
    sbatch_run "sweep_smin_0.43"  0.43  1.0  0.0  0.3   # removes 4 blocks
    sbatch_run "sweep_smin_0.46"  0.46  1.0  0.0  0.3   # removes 5 blocks
    sbatch_run "sweep_smin_0.50"  0.50  1.0  0.0  0.3   # removes 8 blocks
    echo ""
    echo "After these finish, pick best S_min then run: bash run_sweep_smin_theta.sh theta"

elif [ "$MODE" = "theta" ]; then
    echo "=== Sweep 2: theta (S_min=${BEST_SMIN}, alpha=0.0) ==="
    # Taylor scores are nearly uniform across blocks (edge weights >0.84),
    # so theta must be very small to create multiple groups.
    # theta=1.0 → all blocks in one group (isomorphic baseline)
    # theta=0.10 → may split into 2-3 groups
    # theta=0.05 → finer split
    # theta=0.02 → each block nearly its own group
    sbatch_run "sweep_theta_0.02" ${BEST_SMIN}  0.02  0.0  0.3
    sbatch_run "sweep_theta_0.05" ${BEST_SMIN}  0.05  0.0  0.3
    sbatch_run "sweep_theta_0.10" ${BEST_SMIN}  0.10  0.0  0.3
    sbatch_run "sweep_theta_0.15" ${BEST_SMIN}  0.15  0.0  0.3
    sbatch_run "sweep_theta_0.20" ${BEST_SMIN}  0.20  0.0  0.3
    sbatch_run "sweep_theta_1.0"  ${BEST_SMIN}  1.0   0.0  0.3   # isomorphic (no grouping)

else
    echo "Usage: bash run_sweep_smin_theta.sh [smin|theta]"
    exit 1
fi

echo ""
echo "Collect results: python collect_results.py ${BASE_DIR}/sweep_${MODE}_*"
