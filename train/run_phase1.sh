#!/bin/bash
# Train Stage 1. Input: VOICE_* paths and cached crops/masks/scFoundation features. Output: alignment checkpoints.
set -euo pipefail
SF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
: "${VOICE_HF_HOME:?set VOICE_HF_HOME to the HuggingFace cache holding MahmoodLab/UNI2-h}"
: "${VOICE_V2_ROOT:?set VOICE_V2_ROOT to the corpus root (sample_meta/, crops_raw/)}"
: "${VOICE_CKPT_DIR:?set VOICE_CKPT_DIR to where checkpoints should be written}"
export HF_HOME="$VOICE_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false

TAG=${TAG:-v3}
NPROC=${NPROC:-2}
EPOCHS=${EPOCHS:-3}
EFF_BATCH=${EFF_BATCH:-2048}
CHUNK=${CHUNK:-128}
LR_LORA=${LR_LORA:-1.5e-4}
LR_TOWER=${LR_TOWER:-1e-3}
WARMUP=${WARMUP:-200}
NBLOCKS=${NBLOCKS:-12}; RANK_R=${RANK_R:-16}; ALPHA=${ALPHA:-32}; DROPOUT=${DROPOUT:-0.05}
SAVE_EVERY=${SAVE_EVERY:-100}
WORKERS=${WORKERS:-2}
MAX_CELLS=${MAX_CELLS:-0}
VAL_FRAC=${VAL_FRAC:-0.1}
VAL_MARGIN=${VAL_MARGIN:-256}
VAL_EVERY=${VAL_EVERY:-500}
VAL_BATCH=${VAL_BATCH:-0}
VAL_BATCHES=${VAL_BATCHES:-8}
SCF_STATS=${SCF_STATS:-}
EXTRA=${EXTRA:-}

LOG=$SF/logs/phase1_${TAG}.log; mkdir -p "$SF/logs"
[ -n "$SCF_STATS" ] && EXTRA="$EXTRA --scf_stats $SCF_STATS"

echo "===== $(date) | TAG=$TAG NPROC=$NPROC EPOCHS=$EPOCHS CHUNK=$CHUNK VAL_FRAC=$VAL_FRAC =====" | tee -a "$LOG"
cd "$SF"
torchrun --nproc_per_node="$NPROC" train/train_phase1.py \
    --tag "$TAG" --epochs "$EPOCHS" --resume \
    --eff_batch "$EFF_BATCH" --chunk "$CHUNK" \
    --lr_lora "$LR_LORA" --lr_tower "$LR_TOWER" --warmup "$WARMUP" \
    --nblocks "$NBLOCKS" --r "$RANK_R" --alpha "$ALPHA" --dropout "$DROPOUT" \
    --save_every "$SAVE_EVERY" --workers "$WORKERS" --max_cells "$MAX_CELLS" \
    --val_frac "$VAL_FRAC" --val_margin "$VAL_MARGIN" --val_every "$VAL_EVERY" \
    --val_batch "$VAL_BATCH" --val_batches "$VAL_BATCHES" \
    $EXTRA 2>&1 | tee -a "$LOG"
echo "===== $(date) exited rc=$? =====" | tee -a "$LOG"
