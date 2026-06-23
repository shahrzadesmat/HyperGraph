#!/bin/bash
#SBATCH --job-name=hg_phase9
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --account=bfxa-delta-gpu
#SBATCH --output=phase9_%j.log
#
# Tier-1: ATTENTION-AWARE key compression (whitened generalized eigenproblem) vs key-PCA(=MLA)
# and vs phase3's buggy query-aware. Score-error diagnostic + WikiText-2 perplexity surgery.
# WIN: attnAware ppl < keyPCA ppl at matched rank -> the lever is the OBJECTIVE, not nonlinearity.
# Output -> results/phase9_attn_aware${KV_TAG}.txt
#
# Usage: sbatch run_phase9.sh [hf_model]   (default gated Llama-2-7b; needs HF_TOKEN in ./.env)
set -e
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
VENV=/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv
cd /work/hdd/bfxa/dshah13/HyperGraph
mkdir -p results
set -a; [ -f ./.env ] && . ./.env; set +a          # HF_TOKEN for the gated model
export HF_HOME=/work/hdd/bfxa/dshah13/data/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# phase9 knobs (override by exporting before sbatch)
export KV_MODEL="${MODEL}"
export QA_CALIB="${QA_CALIB:-60000}"
export QA_PPL_CHUNKS="${QA_PPL_CHUNKS:-60}"
export QA_RANKS="${QA_RANKS:-8,16,32,48,64,96}"
export QA_PPL_RANKS="${QA_PPL_RANKS:-32,64}"
export KV_RCOND="${KV_RCOND:-1e-3}"
export KV_OUT="${KV_OUT:-results/phase9_attn_aware${KV_TAG:-}.txt}"

PYTHONUNBUFFERED=1 "${VENV}/bin/python" -u phase9_attn_aware.py
