#!/bin/bash
# No-Stage-1 ablation: reuse prepared inputs and the standard Stage-2 trainer.
set -euo pipefail
if [ -n "${INIT_FROM:-}" ]; then
    echo "No-Stage-1 training requires INIT_FROM to be empty." >&2
    exit 2
fi
export TAG=${TAG:-abl_nos1} NPROC=${NPROC:-1}
export LORA_CKPT=none INIT_FROM=
export EPOCHS=${EPOCHS:-2} VAL_FRAC=${VAL_FRAC:-0} EXCL_INSLIDE=${EXCL_INSLIDE:-1}
# Keep this entry no-Stage-1 even if EXTRA contains another --lora_ckpt.
export EXTRA="${EXTRA:-} --lora_ckpt none"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_phase2.sh"
