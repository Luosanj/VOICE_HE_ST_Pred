"""Load configurations and initialize random seeds. Input: YAML configuration and seed. Output: resolved configuration and seeded generators."""
from __future__ import annotations
import os, random
import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path, smoke=False):
    cfg = OmegaConf.load(path)
    if smoke:
        s = cfg.smoke
        cfg.retrieval.K = s.K
        cfg.retrieval.overfetch = s.overfetch
        cfg.encoder.backend = s.encoder_backend
        cfg.train.max_steps = s.max_steps
        cfg.train.batch_size = s.batch_size
        cfg.train.eval_every = s.eval_every
        cfg.splits.held_out_tissues = s.held_out_tissues
        cfg.splits.held_out_genes = s.held_out_genes
        cfg.paths.precompute_dir = os.path.join(cfg.paths.out_root, "precompute_smoke")
        cfg.paths.ckpt_dir = os.path.join(cfg.paths.out_root, "ckpts_smoke")
        cfg._smoke = True
    OmegaConf.resolve(cfg)
    return cfg


def apply_reference_mode(cfg, reference_mode=None):
    """Set precompute_dir for the selected reference mode."""
    if reference_mode:
        cfg.retrieval.reference_mode = reference_mode
    if cfg.retrieval.reference_mode == "self_slide":
        cfg.paths.precompute_dir = cfg.paths.precompute_dir + "_selfslide"
    return cfg


def slides_to_process(cfg, source):
    """List of (tissue, slide). Smoke -> the configured pair; full -> every slide with both modalities."""
    if cfg.get("_smoke", False):
        return [(s.tissue, s.slide) for s in cfg.smoke.slides]
    return [(t, s) for (t, s, _pid, _n) in source.list_slides()]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    return torch.device(cfg.encoder.device if torch.cuda.is_available() else "cpu")
