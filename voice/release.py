"""Loading a released VOICE model from safetensors.

The released weights are two flat safetensors files plus a `config.json`. Flat is what safetensors stores, so
the nested structure the model code expects (`lora`, `se2`, `he_tower`, ...) is encoded in the tensor names as
`group/param` and rebuilt here.

Both stages are needed and their order matters. Stage 1 carries a LoRA adapter for every block it adapted;
Stage 2 ships only the tensors it further trained, and is applied on top. Loading Stage 2 alone leaves the
remaining blocks at the frozen encoder, which does not fail loudly -- it just predicts worse.

    from voice.release import load_release
    model, se2, cfg = load_release("release/voice-23m", device="cuda")

`load_release` also accepts the two `.safetensors` paths directly, and falls back to the original `.pt`
checkpoints so that internal runs and released runs go through the same code path.
"""
from __future__ import annotations
import os, json
import torch


def _unflatten(flat):
    """{'lora/x': t, 'scf_mu': t} -> {'lora': {'x': t}, 'scf_mu': t}"""
    out = {}
    for k, v in flat.items():
        if "/" in k:
            g, name = k.split("/", 1)
            out.setdefault(g, {})[name] = v
        else:
            out[k] = v
    return out


def read_weights(path, device="cpu"):
    """Read a release .safetensors or a training .pt into the nested dict the model code expects."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return _unflatten(load_file(str(path), device=str(device)))
    return torch.load(str(path), map_location=device, weights_only=False)


def load_release(where, device="cuda", stage1=None, stage2=None):
    """Build the LoRA-adapted encoder and the SE(2)+NB head from a release directory.

    Returns (encoder, se2_head, config). `config` is the release config.json when there is one, otherwise a
    dict reconstructed from the checkpoints, so callers can treat both the same way.
    """
    from voice.encoder import build_uni2, inject_lora, load_lora
    from voice.scale_train import ScaleHE2Cell

    cfg = {}
    if where and os.path.isdir(where):
        p = os.path.join(where, "config.json")
        if os.path.exists(p):
            cfg = json.load(open(p))
        stage1 = stage1 or os.path.join(where, "stage1.safetensors")
        stage2 = stage2 or os.path.join(where, "stage2.safetensors")
    if not (stage1 and stage2):
        raise ValueError("give a release directory, or both --stage1 and --stage2")
    for f in (stage1, stage2):
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    dev = torch.device(device)
    w1 = read_weights(stage1, "cpu")
    w2 = read_weights(stage2, "cpu")

    # config.json uses release names (lora_rank, ...); a raw .pt carries the training names (r, ...).
    # Reading both means an internal checkpoint and a release load through the same path.
    ALIAS = {"lora_blocks": "nblocks", "lora_rank": "r", "lora_alpha": "alpha", "lora_dropout": "dropout"}
    def A(key, default=None):
        if key in cfg:
            return cfg[key]
        return w2.get(ALIAS.get(key, key), default)
    nblocks = int(A("lora_blocks", 12)); r = int(A("lora_rank", 16))
    alpha = int(A("lora_alpha", 32)); dropout = float(A("lora_dropout", 0.0))
    d_model, n_layers = int(A("d_model")), int(A("n_layers"))

    model = build_uni2(dev)
    inject_lora(model, nblocks, r, alpha, dropout)
    model.to(dev)
    load_lora(model, w1["lora"], w2.get("lora"))
    model.eval()

    se2_sd = w2["se2"]
    n_genes = cfg.get("n_genes") or int(se2_sd["head.mu_lin.weight"].shape[0])
    se2 = ScaleHE2Cell(n_genes, feat_dim=1536, d_model=d_model, n_layers=n_layers).to(dev)
    se2.load_state_dict({k: v.to(dev) for k, v in se2_sd.items()})
    se2.eval()

    if not cfg:
        cfg = dict(_class_name="VoiceModel", lora_blocks=nblocks, lora_rank=r, lora_alpha=alpha,
                   lora_dropout=dropout, d_model=d_model, n_layers=n_layers, n_genes=n_genes)
    return model, se2, cfg


def gene_names(where, n_genes=None):
    """The head's gene symbols in head order, if the release shipped genes.tsv."""
    if not where or not os.path.isdir(where):
        return None
    p = os.path.join(where, "genes.tsv")
    if not os.path.exists(p):
        return None
    import pandas as pd
    g = pd.read_csv(p, sep="\t")
    col = "gene_symbol" if "gene_symbol" in g.columns else g.columns[-1]
    idx = "global_gene_index" if "global_gene_index" in g.columns else None
    if idx:
        g = g.sort_values(idx)
    names = g[col].astype(str).tolist()
    return names[:n_genes] if n_genes else names
