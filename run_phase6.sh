#!/bin/bash
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_outaware" \
       --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
       --time=01:30:00 --account=bdjd-delta-gpu --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_outaware_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_outaware_%j.err" --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export P6_CALIB=60000 P6_DS='32,64,128,256' P6_PPL_CHUNKS=50
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase6_output_aware_v.py
"
