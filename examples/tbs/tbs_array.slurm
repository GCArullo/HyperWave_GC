#!/bin/bash
# TBS background-search campaign: one 4-second segment per array task, pocoMC.
# Even task index -> noise+injection, odd -> noise-only (Smith & Thrane / Kou+
# 2506.14179 convention). Each task writes its log Bayes factor; aggregate the
# 4000 of them into the duty-cycle xi posterior with tbs_duty_cycle.py.
#
#   sbatch examples/tbs/tbs_array.slurm
#   # then, once the array finishes:
#   python examples/tbs/tbs_duty_cycle.py --dir results/tbs_campaign
#
#SBATCH --job-name=hw_tbs
#SBATCH --array=0-3999
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --output=logs/tbs_%A_%a.out

set -euo pipefail

# --- environment (MSI: hyperwave-dev conda env) ----------------------------
REPO_DIR="${REPO_DIR:-/users/4/asasli/GIT/HyperWave}"
cd "$REPO_DIR"; mkdir -p logs
source /users/4/asasli/miniconda3/etc/profile.d/conda.sh
conda activate hyperwave-dev
PY=/users/4/asasli/miniconda3/envs/hyperwave-dev/bin/python
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# --- configure -------------------------------------------------------------
OUT="results/tbs_campaign_${SAMPLER:-pocomc}"
# PI's GW170814 PSD pickle + full prior if available; else aLIGO design stand-in
# (the speedup + xi-recovery are insensitive to the exact PSD).
PSD_ARG=""; [ -n "${PSD:-}" ] && PSD_ARG="--psd-pickle $PSD"
PRIOR_ARG=""; [ -n "${PRIOR:-}" ] && PRIOR_ARG="--prior-file $PRIOR"
IDX=${SLURM_ARRAY_TASK_ID}
# --------------------------------------------------------------------------

SAMPLER="${SAMPLER:-pocomc}"   # pocomc (fast) | dynesty (accurate evidence, needed
                               # for the faint-signal duty-cycle recovery)
$PY examples/tbs/tbs_segment.py --segment "$IDX" --sampler "$SAMPLER" \
    --n-total 4096 --n-effective 512 --n-active 256 \
    --nlive "${NLIVE:-500}" --walks "${WALKS:-100}" \
    $PSD_ARG $PRIOR_ARG --outdir "$OUT"
