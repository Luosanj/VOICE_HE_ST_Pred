#!/bin/bash
# Phase-1 LoRA-CLIP (HE <-> scFoundation InfoNCE) with validation. Auto-resumes from clip_lora_<TAG>_latest.pt,
# so re-running after a crash/preemption continues where it stopped.
#
# Every hyper-parameter below is the one the shipped 23M run actually used (read back from
# the released Phase-1 checkpoint's args + world=2 from its train log) — only the --val_* block is new.
#
#   bash run_phase1.sh                                  # 2-GPU, 3 epochs, 10% spatial val band
#   TAG=v3b VAL_FRAC=0.05 bash run_phase1.sh
#   SCF_STATS=$VOICE_V2_ROOT/scf_stats.npz bash run_phase1.sh      # reuse the all-cell scF stats, skip the ~130GB stream
#
# 🔴 LAUNCH UNDER tmux/screen. Closing an OnDemand session SIGTERMs the job tree; `setsid nohup` is NOT enough.
set -eu
SF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # repo root
PY=${PY:-python}                                               # override to pin an interpreter
# Paths come from configs/default.yaml, or from these variables. See voice/paths.py.
: "${VOICE_HF_HOME:?set VOICE_HF_HOME to the HuggingFace cache holding MahmoodLab/UNI2-h}"
: "${VOICE_V2_ROOT:?set VOICE_V2_ROOT to the corpus root (sample_meta/, crops_raw/)}"
: "${VOICE_CKPT_DIR:?set VOICE_CKPT_DIR to where checkpoints should be written}"
export HF_HOME="$VOICE_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false

TAG=${TAG:-v3}                   # ckpts -> $VOICE_CKPT_DIR/clip_lora_${TAG}_{latest,best,epoch<N>,final}.pt
NPROC=${NPROC:-2}                # world size of the shipped run; MUST stay constant across resumes
# ---- exactly the shipped 23M Phase-1 recipe ----
EPOCHS=${EPOCHS:-3}
EFF_BATCH=${EFF_BATCH:-2048}     # InfoNCE negatives PER RANK
CHUNK=${CHUNK:-128}              # GradCache memory knob (the shipped run used 128, not the 256 default); OOM auto-halves
LR_LORA=${LR_LORA:-1.5e-4}
LR_TOWER=${LR_TOWER:-1e-3}
WARMUP=${WARMUP:-200}
NBLOCKS=${NBLOCKS:-12}; RANK_R=${RANK_R:-16}; ALPHA=${ALPHA:-32}; DROPOUT=${DROPOUT:-0.05}
SAVE_EVERY=${SAVE_EVERY:-100}
WORKERS=${WORKERS:-2}
MAX_CELLS=${MAX_CELLS:-0}        # >0 = deterministic subsample of TRAIN cells (scale ablations)
# ---- validation (new) ----
VAL_FRAC=${VAL_FRAC:-0.1}        # per-slide spatial band; 0 disables validation entirely
VAL_MARGIN=${VAL_MARGIN:-256}    # MUST be >= the 224px crop width or train/val crops overlap (asserted)
VAL_EVERY=${VAL_EVERY:-500}
VAL_BATCH=${VAL_BATCH:-0}        # 0 = eff_batch. Keep FIXED: InfoNCE depends on the negative count
VAL_BATCHES=${VAL_BATCHES:-8}
SCF_STATS=${SCF_STATS:-}         # empty = recompute mu/sd on the TRAIN split (one-time ~130GB stream, then cached)
EXTRA=${EXTRA:-}

LOG=$SF/logs/phase1_${TAG}.log; mkdir -p "$SF/logs"
[ -n "$SCF_STATS" ] && EXTRA="$EXTRA --scf_stats $SCF_STATS"

echo "===== $(date) | TAG=$TAG NPROC=$NPROC EPOCHS=$EPOCHS CHUNK=$CHUNK VAL_FRAC=$VAL_FRAC =====" | tee -a "$LOG"
cd "$SF"
torchrun --nproc_per_node="$NPROC" train_phase1.py \
    --tag "$TAG" --epochs "$EPOCHS" --resume \
    --eff_batch "$EFF_BATCH" --chunk "$CHUNK" \
    --lr_lora "$LR_LORA" --lr_tower "$LR_TOWER" --warmup "$WARMUP" \
    --nblocks "$NBLOCKS" --r "$RANK_R" --alpha "$ALPHA" --dropout "$DROPOUT" \
    --save_every "$SAVE_EVERY" --workers "$WORKERS" --max_cells "$MAX_CELLS" \
    --val_frac "$VAL_FRAC" --val_margin "$VAL_MARGIN" --val_every "$VAL_EVERY" \
    --val_batch "$VAL_BATCH" --val_batches "$VAL_BATCHES" \
    $EXTRA 2>&1 | tee -a "$LOG"
echo "===== $(date) exited rc=$? =====" | tee -a "$LOG"
