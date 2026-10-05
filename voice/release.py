"""Load model weights; stage1='none' uses UNI2-h plus a no-Stage-1 Stage-2 checkpoint."""
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
        raise ValueError("give a release directory, or --stage1 (path or 'none') and --stage2")
    no_stage1 = str(stage1).strip().lower() == "none"
    for f in ([stage2] if no_stage1 else [stage1, stage2]):
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    dev = torch.device(device)
    w1 = {"lora": {}} if no_stage1 else read_weights(stage1, "cpu")
    w2 = read_weights(stage2, "cpu")


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
