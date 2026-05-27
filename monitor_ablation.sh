#!/bin/bash
# Monitors the 4 ablation jobs and retries any step that doesn't outperform
# the previous one.
#
# Retry logic:
#   plus_smin fails  → report (can't fix analytically; check sensitivity output)
#   plus_theta fails → halve theta and resubmit plus_theta + plus_alpha
#   plus_alpha fails → double alpha and resubmit plus_alpha
#
# Run in a tmux/screen session:
#   bash monitor_ablation.sh

BASE_DIR="/work/hdd/bdjd/hypergraph_pruning/results"
SCRIPT_DIR="/work/hdd/bdjd/hypergraph_pruning"
DATA_PATH="/work/hdd/bdjd/imagenet_10pct"
MODEL="deit_small_patch16_224"
TARGET_MACS=2.5
EPOCHS=20
BEST_SMIN=0.40
POLL_SECS=120    # check every 2 minutes

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── helpers ──────────────────────────────────────────────────────────────────

get_acc() {
    python3 -c "
import json, sys
try:
    print(json.load(open('${BASE_DIR}/$1/results.json'))['finetuned_acc'])
except: print(-1)
"
}

beats() {
    # true if $1 > $2 + 0.001
    python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') + 0.001 else 1)"
}

num_groups() {
    # count attn groups from latest .out file for a run
    python3 -c "
import glob, re, ast
files = sorted(glob.glob('${BASE_DIR}/$1/slurm_*.out'))
if not files: print(1); exit()
for line in open(files[-1]):
    if 'Attn groups:' in line:
        groups = ast.literal_eval(line.split('Attn groups:')[1].strip())
        print(len(groups)); exit()
print(1)
"
}

wait_for() {
    local name=$1
    log "Waiting for ${name} to finish..."
    while [ ! -f "${BASE_DIR}/${name}/results.json" ]; do
        sleep $POLL_SECS
    done
    local acc=$(get_acc $name)
    log "${name} done — finetuned_acc=${acc}"
}

sbatch_run() {
    local name=$1 S_MIN=$2 THETA=$3 ALPHA=$4
    local outdir="${BASE_DIR}/${name}"
    mkdir -p "$outdir"
    # remove old results so wait_for doesn't see stale data
    rm -f "${outdir}/results.json"
    sbatch --job-name="hg_${name}" \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --cpus-per-task=32 \
           --mem=64G \
           --time=12:00:00 \
           --account=bdjd-delta-gpu \
           --output="${outdir}/slurm_%j.out" \
           --error="${outdir}/slurm_%j.err" \
           --export=ALL \
           --wrap="
export PATH=/work/hdd/bdjd/miniconda3/bin:\$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh
cd ${SCRIPT_DIR}
PYTHONUNBUFFERED=1 python run.py \
  --model ${MODEL} --data_path ${DATA_PATH} \
  --target_macs_g ${TARGET_MACS} \
  --S_min ${S_MIN} --theta ${THETA} --alpha ${ALPHA} \
  --edge_threshold 0.3 --epochs ${EPOCHS} \
  --output_dir ${outdir}
"
    log "Submitted ${name}  S_min=${S_MIN} theta=${THETA} alpha=${ALPHA}"
}

# ── Step 1: iso_baseline ─────────────────────────────────────────────────────
wait_for "iso_baseline"
ACC_ISO=$(get_acc iso_baseline)

# ── Step 2: plus_smin ────────────────────────────────────────────────────────
wait_for "plus_smin"
ACC_SMIN=$(get_acc plus_smin)

if ! beats $ACC_SMIN $ACC_ISO; then
    log "WARNING: plus_smin (${ACC_SMIN}) did not beat iso_baseline (${ACC_ISO})."
    log "  S_min=0.40 depth pruning does not help here. Proceeding anyway."
fi

# ── Step 3: plus_theta (retry with smaller theta if needed) ──────────────────
THETA=0.05
for RETRY in 1 2 3; do
    wait_for "plus_theta"
    ACC_THETA=$(get_acc plus_theta)
    NGROUPS=$(num_groups plus_theta)
    log "plus_theta: acc=${ACC_THETA}  groups=${NGROUPS}  theta=${THETA}"

    if beats $ACC_THETA $ACC_SMIN && [ "$NGROUPS" -gt 1 ]; then
        log "plus_theta improved with ${NGROUPS} groups. Moving on."
        break
    fi

    if [ $RETRY -eq 3 ]; then
        log "WARNING: theta exhausted retries. Keeping best theta=${THETA}."
        break
    fi

    # halve theta and resubmit
    THETA=$(python3 -c "print(round($THETA / 2, 4))")
    log "plus_theta did not improve (groups=${NGROUPS}). Retrying with theta=${THETA}..."
    sbatch_run "plus_theta" $BEST_SMIN $THETA 0.0
    sbatch_run "plus_alpha" $BEST_SMIN $THETA 0.3
done

# ── Step 4: plus_alpha (retry with larger alpha if needed) ───────────────────
ALPHA=0.3
for RETRY in 1 2 3; do
    wait_for "plus_alpha"
    ACC_ALPHA=$(get_acc plus_alpha)
    log "plus_alpha: acc=${ACC_ALPHA}  alpha=${ALPHA}"

    if beats $ACC_ALPHA $ACC_THETA; then
        log "plus_alpha improved. Ablation complete!"
        break
    fi

    if [ $RETRY -eq 3 ]; then
        log "WARNING: alpha exhausted retries. alpha may not contribute at this setting."
        break
    fi

    ALPHA=$(python3 -c "print(round($ALPHA * 2, 2))")
    log "plus_alpha did not improve. Retrying with alpha=${ALPHA}..."
    sbatch_run "plus_alpha" $BEST_SMIN $THETA $ALPHA
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  ABLATION RESULTS"
echo "════════════════════════════════════════"
printf "  iso_baseline : %.4f\n" $ACC_ISO
printf "  + S_min=%.2f : %.4f  (delta=%+.4f)\n" $BEST_SMIN $ACC_SMIN $(python3 -c "print(round($ACC_SMIN-$ACC_ISO,4))")
printf "  + theta=%-5s : %.4f  (delta=%+.4f)\n" $THETA $ACC_THETA $(python3 -c "print(round($ACC_THETA-$ACC_SMIN,4))")
printf "  + alpha=%-5s : %.4f  (delta=%+.4f)\n" $ALPHA $ACC_ALPHA $(python3 -c "print(round($ACC_ALPHA-$ACC_THETA,4))")
echo "════════════════════════════════════════"
