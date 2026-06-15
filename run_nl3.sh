#!/bin/bash
# Confound-free nonlinear-vs-linear redundancy probe (probe_nonlinear3.py).
# Synthesized from the 3 surviving design-lens agents. DeiT validates the whole
# pipeline fast; Llama is the decisive case (the one that exposed the overfit).
set -e
cd /work/hdd/bdjd/hypergraph_pruning

submit() {
  local model=$1 sites=$2 walltime=$3 name=$4
  sbatch --job-name="nl3_${name}" \
         --partition=gpuA100x4 \
         --gres=gpu:1 \
         --cpus-per-task=16 \
         --mem=128G \
         --time=${walltime} \
         --account=bdjd-delta-gpu \
         --exclude=gpua001,gpub066,gpub088 \
         --output="/work/hdd/bdjd/hypergraph_pruning/nl3_${name}_%j.out" \
         --error="/work/hdd/bdjd/hypergraph_pruning/nl3_${name}_%j.err" \
         --export=ALL \
         --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /work/hdd/bdjd/hypergraph_pruning
python -u probe_nonlinear3.py ${model} ${sites}
"
  echo "[submitted] nl3_${name}  model=${model} sites=${sites} time=${walltime}"
}

submit "deit_small_patch16_224"   "mlp,heads" "01:00:00" "deit"
submit "meta-llama/Llama-2-7b-hf" "mlp,heads" "06:00:00" "llama"
