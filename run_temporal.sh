#!/bin/bash
# LEVER B headroom: is key_t predictable from recent keys? (temporal axis MLA ignores)
# Separate job; does NOT touch the running mla_7b_aggr / mla_13b jobs.
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_temporal" \
       --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
       --time=02:00:00 --account=bdjd-delta-gpu \
       --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_temporal_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_temporal_%j.err" \
       --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMP_MODEL='meta-llama/Llama-2-7b-hf' TMP_CALIB=80000 TMP_WINDOWS='1,4' TMP_LAYERS='0,8,16,24,31'
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase4_temporal.py
"
echo "[submitted] kv_temporal  (Lever B headroom, 7b)"
