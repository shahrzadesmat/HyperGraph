#!/bin/bash
# Does temporal redundancy SURVIVE MLA compression? (is the MLA latent still temporally predictable)
# Separate job; does NOT touch the running mla_7b_aggr / mla_13b / kv_temporal jobs.
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_survive" \
       --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
       --time=02:00:00 --account=bdjd-delta-gpu \
       --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_survive_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_survive_%j.err" \
       --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export SM_MODEL='meta-llama/Llama-2-7b-hf' SM_CALIB=60000 SM_LAYERS='0,8,16,24,31' SM_DCS='2048,1024,512,256'
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase5_survive_mla.py
"
echo "[submitted] kv_survive  (does temporal survive MLA?)"
