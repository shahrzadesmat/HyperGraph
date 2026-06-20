#!/bin/bash
# Gate #1: is the dimensional x temporal coupling EXPLOITABLE? (joint vs separable-surrogate KV transform)
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_gate1" \
       --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
       --time=01:30:00 --account=bdjd-delta-gpu \
       --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_gate1_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_gate1_%j.err" \
       --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export G1_MODEL='meta-llama/Llama-2-7b-hf' G1_CALIB=50000 G1_TB=16 G1_RS='64,128,256,512' G1_PPL_CHUNKS=50
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u gate1_joint_st.py
"
echo "[submitted] kv_gate1  (coupling exploitability gate)"
