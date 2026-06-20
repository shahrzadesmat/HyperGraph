#!/bin/bash
# nonlinear-MLA vs linear-MLA across SIZE (7b vs 13b) and BUDGET (mild -> aggressive),
# to find a regime where MLA is NOT already lossless and see if nonlinear holds up better.
set -e
cd /work/hdd/bdjd/hypergraph_pruning
submit() {
  local model=$1 taus=$2 calib=$3 tag=$4 walltime=$5
  sbatch --job-name="mla${tag}" \
         --partition=gpuA100x4 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
         --time=${walltime} --account=bdjd-delta-gpu \
         --exclude=gpua001,gpub066,gpub088 \
         --output="/work/hdd/bdjd/hypergraph_pruning/mla${tag}_%j.out" \
         --error="/work/hdd/bdjd/hypergraph_pruning/mla${tag}_%j.err" \
         --export=ALL \
         --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export KV_MODEL='${model}' KV_TAUS='${taus}' KV_CALIB='${calib}' KV_TAG='${tag}'
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase2b_mla_joint.py
"
  echo "[submitted] mla${tag}  model=${model}  taus=${taus}  calib=${calib}  time=${walltime}"
}
# A: 7b, AGGRESSIVE budgets (push past the 2x near-lossless point into where MLA breaks)
submit "meta-llama/Llama-2-7b-hf"  "0.95,0.90,0.85,0.80,0.75" 60000 "_7b_aggr" "05:00:00"
# B: 13b, mild->aggressive (the bigger-model test)
submit "meta-llama/Llama-2-13b-hf" "0.95,0.88,0.80"           40000 "_13b"     "06:00:00"
