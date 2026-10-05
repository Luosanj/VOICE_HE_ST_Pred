#!/bin/bash
# Train Stage 2. Input: VOICE_* paths, Stage-1 checkpoint, crops/masks/counts. Output: expression-model checkpoints.
set -euo pipefail
SF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
: "${VOICE_HF_HOME:?set VOICE_HF_HOME to the HuggingFace cache holding MahmoodLab/UNI2-h}"
: "${VOICE_V2_ROOT:?set VOICE_V2_ROOT to the corpus root (sample_meta/, crops_raw/)}"
: "${VOICE_CKPT_DIR:?set VOICE_CKPT_DIR to where checkpoints should be written}"
export HF_HOME="$VOICE_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false

TAG=${TAG:-p2v3}
NPROC=${NPROC:-3}
EPOCHS=${EPOCHS:-2}
LR_LORA=${LR_LORA:-3e-5}
LR_SE2=${LR_SE2:-1e-4}
WARMUP_FRAC=${WARMUP_FRAC:-0.05}
CONST_LR=${CONST_LR:-0}
NBLOCKS=${NBLOCKS:-12}; RANK_R=${RANK_R:-16}; ALPHA=${ALPHA:-32}; DROPOUT=${DROPOUT:-0.05}
FREEZE_FRAC=${FREEZE_FRAC:-0.2}
D_MODEL=${D_MODEL:-512}; N_LAYERS=${N_LAYERS:-6}
MAX_CELLS=${MAX_CELLS:-256}
PATCH=${PATCH:-256}; OVERLAP=${OVERLAP:-30}
GRAD_CLIP=${GRAD_CLIP:-1.0}
SAVE_EVERY=${SAVE_EVERY:-200}
WORKERS=${WORKERS:-6}
LORA_CKPT=${LORA_CKPT:-$VOICE_V2_ROOT/ckpts/clip_lora_v2_final.pt}
INIT_FROM=${INIT_FROM:-}
EXCL_INSLIDE=${EXCL_INSLIDE:-1}
VAL_FRAC=${VAL_FRAC:-0.1}
VAL_MARGIN=${VAL_MARGIN:-256}
VAL_EVERY=${VAL_EVERY:-2000}
VAL_PATCHES=${VAL_PATCHES:-400}
EXTRA=${EXTRA:-}

LOG=$SF/logs/phase2_${TAG}.log; mkdir -p "$SF/logs"
[ "$EXCL_INSLIDE" = "1" ] && EXTRA="$EXTRA --exclude_inslide"
[ -n "$INIT_FROM" ] && EXTRA="$EXTRA --init_from $INIT_FROM"

echo "===== $(date) | TAG=$TAG NPROC=$NPROC EPOCHS=$EPOCHS CONST_LR=$CONST_LR VAL_FRAC=$VAL_FRAC =====" | tee -a "$LOG"
cd "$SF"
torchrun --nproc_per_node="$NPROC" train/train_phase2.py \
    --tag "$TAG" --epochs "$EPOCHS" --resume \
    --lr_lora "$LR_LORA" --lr_se2 "$LR_SE2" --warmup_frac "$WARMUP_FRAC" --const_lr "$CONST_LR" \
    --nblocks "$NBLOCKS" --r "$RANK_R" --alpha "$ALPHA" --dropout "$DROPOUT" --freeze_frac "$FREEZE_FRAC" \
    --d_model "$D_MODEL" --n_layers "$N_LAYERS" \
    --max_cells "$MAX_CELLS" --patch_size "$PATCH" --overlap "$OVERLAP" \
    --grad_clip "$GRAD_CLIP" --save_every "$SAVE_EVERY" --workers "$WORKERS" --lora_ckpt "$LORA_CKPT" \
    --val_frac "$VAL_FRAC" --val_margin "$VAL_MARGIN" --val_every "$VAL_EVERY" --val_max_patches "$VAL_PATCHES" \
    $EXTRA 2>&1 | tee -a "$LOG"
echo "===== $(date) exited rc=$? =====" | tee -a "$LOG"
