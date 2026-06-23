#!/bin/bash
#SBATCH --job-name=hg_phase10
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --account=bfxa-delta-gpu
#SBATCH --output=phase10_%j.log
#
# Tier-2: the EMPTY CELL — nonlinear residual on the attn-aware (Σq-whitened) base, trained under the
# Σq metric vs Euclidean, with a Gaussian null + downstream perplexity. Decisive either way:
#   +corrW < attnAware  -> nonlinearity pays in the right metric (novel positive)
#   +corrW ≈ attnAware  -> objective is the whole lever (clean negative, closes the 2x2)
# Output -> results/phase10_nonlinear_attn${KV_TAG}.txt
#
# Usage: sbatch run_phase10.sh [hf_model]   (default gated Llama-2-7b; needs HF_TOKEN in ./.env)
set -e
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
VENV=/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv
cd /work/hdd/bfxa/dshah13/HyperGraph
mkdir -p results
set -a; [ -f ./.env ] && . ./.env; set +a          # HF_TOKEN for the gated model
export HF_HOME=/work/hdd/bfxa/dshah13/data/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# phase10 knobs (override by exporting before sbatch)
export KV_MODEL="${MODEL}"
export P10_CALIB="${P10_CALIB:-60000}"
export P10_PPL_CHUNKS="${P10_PPL_CHUNKS:-60}"
export P10_RANKS="${P10_RANKS:-16,32}"
export P10_NULL="${P10_NULL:-1}"
export KV_RCOND="${KV_RCOND:-1e-3}"
export KV_OUT="${KV_OUT:-results/phase10_nonlinear_attn${KV_TAG:-}.txt}"

PYTHONUNBUFFERED=1 "${VENV}/bin/python" -u phase10_nonlinear_attn.py
