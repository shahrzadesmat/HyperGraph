#!/bin/bash
#SBATCH --job-name=hg_phase12
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --account=bfxa-delta-gpu
#SBATCH --output=phase12_%j.log
#
# Joint K+V at matched TOTAL budget: mla vs kqsvd(Σq-K + G-V) vs fisher(M̄-K + G-V).
# WIN: fisher stack stays usable at higher total cacheX -> 2nd-order objective extends the frontier.
# Output -> results/phase12_joint_kv${KV_TAG}.txt
#
# Usage: sbatch run_phase12.sh [hf_model]   (default gated Llama-2-7b; needs HF_TOKEN in ./.env)
set -e
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
VENV=/work/hdd/bfxa/dshah13/mac_pruning_fv/.venv
cd /work/hdd/bfxa/dshah13/HyperGraph
mkdir -p results
set -a; [ -f ./.env ] && . ./.env; set +a
export HF_HOME=/work/hdd/bfxa/dshah13/data/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export KV_MODEL="${MODEL}"
export P12_CALIB_SEQS="${P12_CALIB_SEQS:-80}"
export P12_SEQLEN="${P12_SEQLEN:-512}"
export P12_BATCH="${P12_BATCH:-4}"
export P12_PPL_CHUNKS="${P12_PPL_CHUNKS:-60}"
export P12_RANKS="${P12_RANKS:-8,16,32,64}"
export KV_RCOND="${KV_RCOND:-1e-3}"
export KV_OUT="${KV_OUT:-results/phase12_joint_kv${KV_TAG:-}.txt}"

PYTHONUNBUFFERED=1 "${VENV}/bin/python" -u phase12_joint_kv.py
