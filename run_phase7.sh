#!/bin/bash
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_prefix" \
       --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
       --time=01:00:00 --account=bdjd-delta-gpu --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_prefix_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_prefix_%j.err" --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export P7_L=1024 P7_CT=256 P7_NSCEN=4 P7_NPROBE=6 P7_DS='16,32,64,128,256'
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase7_prefix_kv.py
"
