#!/bin/bash
#SBATCH --job-name=hg_phase8
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=bfxa-delta-gpu
#SBATCH --output=phase8_%j.log
#
# Phase-8: does TEMPORAL INNOVATION coding (DeltaKV) stack on top of the nonlinear-MLA latent?
# Caches e_t = z_t - predict(z_{t-1}) vs the per-token latent z_t at matched bits/token.
# Output table -> results/phase8_deltakv_stack${KV_TAG}.txt
#
# Usage: sbatch run_phase8.sh [hf_model]
#   default model is the gated Llama-2-7b (needs HF_TOKEN in ./.env)
set -e
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
VENV=/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv
cd /work/hdd/bfxa/dshah13/HyperGraph
set -a; [ -f ./.env ] && . ./.env; set +a          # HF_TOKEN for the gated model
export HF_HOME=/work/hdd/bfxa/dshah13/data/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# phase8 knobs (override on the command line by exporting before sbatch)
export KV_MODEL="${MODEL}"
export KV_TAU="${KV_TAU:-0.9}"
export KV_CALIB="${KV_CALIB:-60000}"
export KV_PPL_CHUNKS="${KV_PPL_CHUNKS:-40}"
export KV_BITS="${KV_BITS:-2,3,4,6,16}"
export KV_CLIP="${KV_CLIP:-4.0}"

PYTHONUNBUFFERED=1 "${VENV}/bin/python" -u phase8_deltakv_stack.py
