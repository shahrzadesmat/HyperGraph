#!/bin/bash
# Test #2: output-aware (query-subspace) vs reconstruction-aware (key-PCA = MLA) KEY compression.
# Does compressing keys onto the QUERY subspace beat compressing onto KEY-variance, at matched rank,
# on (1) the q.k score-error and (2) downstream WikiText-2 perplexity? -> is the output-aware lever real.
set -e
cd /work/hdd/bdjd/hypergraph_pruning
sbatch --job-name="kv_qaware" \
       --partition=gpuA100x4 \
       --gres=gpu:1 \
       --cpus-per-task=16 \
       --mem=128G \
       --time=01:00:00 \
       --account=bdjd-delta-gpu \
       --exclude=gpua001,gpub066,gpub088 \
       --output="/work/hdd/bdjd/hypergraph_pruning/kv_qaware_%j.out" \
       --error="/work/hdd/bdjd/hypergraph_pruning/kv_qaware_%j.err" \
       --export=ALL \
       --wrap="
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /work/hdd/bdjd/hypergraph_pruning
/work/hdd/bdjd/miniconda3/envs/pytorch_fresh/bin/python -u phase3_query_aware.py
"
