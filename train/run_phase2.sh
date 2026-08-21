#!/bin/bash
# Phase-2 SE2-LoRA (gene-supervised) with validation. Auto-resumes from se2_lora_<TAG>_latest.pt, so re-running
# after a crash/preemption continues where it stopped.
#
# Every hyper-parameter below is the one the shipped 23M run actually used (read back from
# he2gene/ckpts/se2_lora_p2v2_epoch0.pt args + world=3 from its train log) — only the --val_* block is new.
#
#   bash run_phase2.sh                       # 3-GPU, 75 slides, 2 epochs, 10% spatial val band
#   TAG=p2v3full EXCL_INSLIDE=0 bash ...      # all 80 slides
#   # the noinslide-style continuation (warm-start trained weights, flat LR, 1 more epoch):
#   TAG=p2v3_cont EPOCHS=1 CONST_LR=0.05 \
#     INIT_FROM=$VOICE_CKPT_DIR/se2_lora_p2v3_epoch1.pt bash run_phase2.sh
#
# 🔴 LAUNCH UNDER tmux/screen. Closing an OnDemand session SIGTERMs the job tree; `setsid nohup` is NOT enough
#    (this is how the ext_p2v2 3rd epoch died at 20%).
set -eu
SF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # repo root
PY=${PY:-python}                                               # override to pin an interpreter
# Paths come from configs/default.yaml, or from these variables. See voice/paths.py.
: "${VOICE_HF_HOME:?set VOICE_HF_HOME to the HuggingFace cache holding MahmoodLab/UNI2-h}"
: "${VOICE_V2_ROOT:?set VOICE_V2_ROOT to the corpus root (sample_meta/, crops_raw/)}"
: "${VOICE_CKPT_DIR:?set VOICE_CKPT_DIR to where checkpoints should be written}"
export HF_HOME="$VOICE_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false

TAG=${TAG:-p2v3}                 # ckpts -> he2gene/ckpts/se2_lora_${TAG}_{latest,best,epoch<N>,final}.pt
NPROC=${NPROC:-3}                # world size of the shipped run; MUST stay constant across resumes
# ---- exactly the shipped 23M Phase-2 recipe ----
EPOCHS=${EPOCHS:-2}
LR_LORA=${LR_LORA:-3e-5}
LR_SE2=${LR_SE2:-1e-4}
WARMUP_FRAC=${WARMUP_FRAC:-0.05}
CONST_LR=${CONST_LR:-0}          # >0 = flat lr = CONST_LR*base (no-spike continuation; the noinslide run used 0.05)
NBLOCKS=${NBLOCKS:-12}; RANK_R=${RANK_R:-16}; ALPHA=${ALPHA:-32}; DROPOUT=${DROPOUT:-0.05}
FREEZE_FRAC=${FREEZE_FRAC:-0.2}
D_MODEL=${D_MODEL:-512}; N_LAYERS=${N_LAYERS:-6}
MAX_CELLS=${MAX_CELLS:-256}      # cells per 256px patch
PATCH=${PATCH:-256}; OVERLAP=${OVERLAP:-30}
GRAD_CLIP=${GRAD_CLIP:-1.0}
SAVE_EVERY=${SAVE_EVERY:-200}
WORKERS=${WORKERS:-6}
LORA_CKPT=${LORA_CKPT:-$VOICE_V2_ROOT/ckpts/clip_lora_v2_final.pt}   # Phase-1 backbone to warm-start from
INIT_FROM=${INIT_FROM:-}         # warm-start TRAINED lora+se2 (fresh counter/opt) — use when the dataset changed
EXCL_INSLIDE=${EXCL_INSLIDE:-1}  # 1 = also hold out the 5 in-slide benchmark slices (75 slides)
# ---- validation (new) ----
VAL_FRAC=${VAL_FRAC:-0.1}        # per-slide spatial band; 0 disables validation entirely
VAL_MARGIN=${VAL_MARGIN:-256}    # px buffer; >= patch_size is the safe default
VAL_EVERY=${VAL_EVERY:-2000}
VAL_PATCHES=${VAL_PATCHES:-400}
EXTRA=${EXTRA:-}

LOG=$SF/logs/phase2_${TAG}.log; mkdir -p "$SF/logs"
[ "$EXCL_INSLIDE" = "1" ] && EXTRA="$EXTRA --exclude_inslide"
[ -n "$INIT_FROM" ] && EXTRA="$EXTRA --init_from $INIT_FROM"

echo "===== $(date) | TAG=$TAG NPROC=$NPROC EPOCHS=$EPOCHS CONST_LR=$CONST_LR VAL_FRAC=$VAL_FRAC =====" | tee -a "$LOG"
cd "$SF"
torchrun --nproc_per_node="$NPROC" train_phase2.py \
    --tag "$TAG" --epochs "$EPOCHS" --resume \
    --lr_lora "$LR_LORA" --lr_se2 "$LR_SE2" --warmup_frac "$WARMUP_FRAC" --const_lr "$CONST_LR" \
    --nblocks "$NBLOCKS" --r "$RANK_R" --alpha "$ALPHA" --dropout "$DROPOUT" --freeze_frac "$FREEZE_FRAC" \
    --d_model "$D_MODEL" --n_layers "$N_LAYERS" \
    --max_cells "$MAX_CELLS" --patch_size "$PATCH" --overlap "$OVERLAP" \
    --grad_clip "$GRAD_CLIP" --save_every "$SAVE_EVERY" --workers "$WORKERS" --lora_ckpt "$LORA_CKPT" \
    --val_frac "$VAL_FRAC" --val_margin "$VAL_MARGIN" --val_every "$VAL_EVERY" --val_max_patches "$VAL_PATCHES" \
    $EXTRA 2>&1 | tee -a "$LOG"
echo "===== $(date) exited rc=$? =====" | tee -a "$LOG"
