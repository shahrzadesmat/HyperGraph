#!/bin/bash
#SBATCH --job-name=hg_phase11
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --account=bfxa-delta-gpu
#SBATCH --output=phase11_%j.log
#
# Second-order objective: the output/Fisher metric M̄ = Σ_m c_m q q^T (c_m = attention-mass² ×
# value-distinctiveness through W_O). Compares keyPCA vs attnAware(Σq=KQ-SVD) vs fisher(M̄) on K.
# WIN: fisher ppl < attnAware ppl -> the 2nd-order output metric beats the 1st-order logit metric.
# Output -> results/phase11_fisher_metric${KV_TAG}.txt
#
# Usage: sbatch run_phase11.sh [hf_model]   (default gated Llama-2-7b; needs HF_TOKEN in ./.env)
set -e
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
VENV=/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv
cd /work/hdd/bfxa/dshah13/HyperGraph
mkdir -p results
set -a; [ -f ./.env ] && . ./.env; set +a          # HF_TOKEN for the gated model
export HF_HOME=/work/hdd/bfxa/dshah13/data/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# phase11 knobs (override by exporting before sbatch)
export KV_MODEL="${MODEL}"
export P11_CALIB_SEQS="${P11_CALIB_SEQS:-80}"
export P11_SEQLEN="${P11_SEQLEN:-512}"
export P11_BATCH="${P11_BATCH:-4}"
export P11_PPL_CHUNKS="${P11_PPL_CHUNKS:-60}"
export P11_RANKS="${P11_RANKS:-8,16,32,64}"
export KV_RCOND="${KV_RCOND:-1e-3}"
export KV_OUT="${KV_OUT:-results/phase11_fisher_metric${KV_TAG:-}.txt}"

PYTHONUNBUFFERED=1 "${VENV}/bin/python" -u phase11_fisher_metric.py
