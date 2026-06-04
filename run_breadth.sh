#!/bin/bash
# Breadth sweep: does theta+alpha beat isomorphic across MODELS and MAC TARGETS?
# Each cell: iso_baseline (theta=1,alpha=0) vs ours (theta=0.025,alpha=0.3),
# residual-preserving + Taylor pipeline, same budget -> clean internal comparison.

set -e
DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
EPOCHS=5
HEAD_SCALE=0.2
BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"

sbatch_run() {
    local name=$1 MODEL=$2 TARGET=$3 THETA=$4 ALPHA=$5
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"; rm -f "${outdir}/results.json"
    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=32 --mem=64G \
           --time=03:00:00 --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" --error="${outdir}/slurm_%j.err" --export=ALL \
           --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
cd /work/hdd/bdjd/hypergraph_pruning
PYTHONUNBUFFERED=1 python -u run.py \
  --model ${MODEL} --data_path ${DATA_PATH} \
  --target_macs_g ${TARGET} --S_min 0.0 --theta ${THETA} --alpha ${ALPHA} \
  --head_scale ${HEAD_SCALE} --epochs ${EPOCHS} --seed 42 --output_dir ${outdir}
"
    echo "[Submitted] ${name}  ${MODEL} @ ${TARGET}G  theta=${THETA} alpha=${ALPHA}"
}

# ---- DeiT-Small, two new MAC targets (2.5G already done) ----
sbatch_run "br_small_1.8_iso"  deit_small_patch16_224 1.8 1.0   0.0
sbatch_run "br_small_1.8_ours" deit_small_patch16_224 1.8 0.025 0.3
sbatch_run "br_small_3.3_iso"  deit_small_patch16_224 3.3 1.0   0.0
sbatch_run "br_small_3.3_ours" deit_small_patch16_224 3.3 0.025 0.3

# ---- DeiT-Tiny (baseline ~1.26G), two targets ----
sbatch_run "br_tiny_0.7_iso"   deit_tiny_patch16_224  0.7 1.0   0.0
sbatch_run "br_tiny_0.7_ours"  deit_tiny_patch16_224  0.7 0.025 0.3
sbatch_run "br_tiny_0.9_iso"   deit_tiny_patch16_224  0.9 1.0   0.0
sbatch_run "br_tiny_0.9_ours"  deit_tiny_patch16_224  0.9 0.025 0.3

echo ""
echo "Anchor (DeiT-Small @2.5G): iso=0.6914  ours(theta+alpha)=0.6997  (+0.83pp)"
