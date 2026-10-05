"""Resolve data and model paths. Input: VOICE_* environment variables or YAML configuration. Output: paths and HuggingFace cache settings."""
from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_YAML = None


def _yaml():
    global _YAML
    if _YAML is None:
        p = Path(os.environ.get("VOICE_CONFIG", REPO / "configs" / "default.yaml"))
        if p.exists():
            try:
                import yaml
                _YAML = (yaml.safe_load(p.read_text()) or {}).get("paths", {}) or {}
            except Exception:
                _YAML = {}
        else:
            _YAML = {}
    return _YAML


_WHY = {
    "HF_HOME":      "HuggingFace cache holding MahmoodLab/UNI2-h (a GATED model: request access, then "
                    "`huggingface-cli download MahmoodLab/UNI2-h`)",
    "CKPT_DIR":     "directory with the released VOICE weights. For a release that is stage1/stage2.safetensors; "
                    "for internal checkpoints, clip_lora_v2_final.pt and se2_lora_p2v2_noinslide_epoch0.pt",
    "DATA_ROOT":    "training corpus root: <dataset>/<slide>/{manifest.csv.gz, patch_cell_boundaries.npz, patches/}",
    "SCF_DIR":      "precomputed scFoundation cell embeddings, Phase-1 contrastive targets only",
    "V2_ROOT":      "per-slide metadata + expression used by training (sample_meta/<slide>/)",
    "TESTSET_ROOT": "preprocessed benchmark slides used by experiments/benchmark/ (test_preprocessed/<group>/<slide>/)",
}


def get(key: str, default: str | None = None, required: bool = True) -> str:
    """Resolve one path setting. `key` is one of the names in _WHY."""
    v = os.environ.get(f"VOICE_{key}") or _yaml().get(key.lower()) or default
    if v is None and required:
        raise RuntimeError(
            f"path '{key}' is not set. Export VOICE_{key}=... or set paths.{key.lower()} in "
            f"configs/default.yaml.\n  It is the {_WHY.get(key, 'see voice/paths.py')}."
        )
    return str(v) if v is not None else None


def hf_home() -> str:
    """Exported into the environment before timm/transformers import, so UNI2-h resolves offline."""
    h = get("HF_HOME")
    os.environ.setdefault("HF_HOME", h)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return h


def ckpt_dir() -> str: return get("CKPT_DIR")
def data_root() -> str: return get("DATA_ROOT")
def scf_dir() -> str: return get("SCF_DIR")
def v2_root() -> str: return get("V2_ROOT")
def testset_root() -> str: return get("TESTSET_ROOT")
def index_csv() -> str: return get("INDEX_CSV", str(Path(data_root()) / "index.csv"), required=False)
