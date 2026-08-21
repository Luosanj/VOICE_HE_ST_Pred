#!/usr/bin/env python
"""Convert a training checkpoint into a release artefact: safetensors weights plus a JSON config.

    python tools/export_release.py --stage1 clip_lora_v2_final.pt --stage2 se2_lora_p2v2_noinslide_epoch0.pt \
                                   --out release/voice-23m --name VOICE-23M

Training checkpoints are `torch.save` pickles carrying far more than a user needs: optimiser state, RNG state,
the full argparse namespace, and local filesystem paths. Three reasons that is the wrong thing to publish.

  * **Size.** Optimiser state triples the file (98 MB -> 294 MB) and is useless to anyone who is not resuming
    that exact run.
  * **Safety.** Loading a pickle executes whatever is in it, so `torch.load` on a downloaded `.pt` is arbitrary
    code execution. safetensors is a flat, inert tensor container; nothing runs when you read it.
  * **Leakage.** The Stage-2 args carry an absolute path to the machine it was trained on.

What comes out:

    <out>/stage1.safetensors    Stage-1 LoRA adapter + the two projection towers + scF normalisation
    <out>/stage2.safetensors    Stage-2 LoRA overlay + SE(2) decoder + NB head
    <out>/config.json           the architecture, flat -- every key is a constructor argument
    <out>/README.md            what the files are, how to load them, what the licence is

Nested dicts are flattened with a `/` separator (`lora/blocks.12.attn.qkv.A`), because safetensors stores a flat
name -> tensor map. `voice/release.py` reads them back into the same nested structure the model code expects.

Every export is verified: the written tensors are read back and compared to the source **element by element**,
and the export aborts if a single value differs.
"""
from __future__ import annotations
import os, sys, json, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

# Only these tensor groups are published. Everything else in the checkpoint -- optimiser state, RNG state, the
# argparse namespace, step/epoch counters -- is training bookkeeping and is dropped.
TENSOR_GROUPS = {
    "stage1": ["lora", "he_tower", "scf_tower"],
    "stage2": ["lora", "se2"],
}
BARE_TENSORS = {"stage1": ["scf_mu", "scf_sd"], "stage2": []}


def flatten(ck, stage):
    """Nested checkpoint -> flat {name: tensor} for safetensors, with `group/param` names."""
    out = {}
    for g in TENSOR_GROUPS[stage]:
        if g not in ck:
            continue
        if not isinstance(ck[g], dict):
            raise TypeError(f"{stage}: '{g}' is {type(ck[g]).__name__}, expected a state dict")
        for k, v in ck[g].items():
            out[f"{g}/{k}"] = v.detach().cpu().contiguous()
    for b in BARE_TENSORS[stage]:
        if b in ck and torch.is_tensor(ck[b]):
            out[b] = ck[b].detach().cpu().contiguous()
    return out


def config_of(ck1, ck2, n_genes):
    """The architecture, and nothing else.

    A release config exists so the modules can be instantiated -- that is the whole job. Training bookkeeping
    (step, epoch, learning rates, which run it came from) describes how the weights were MADE, not what they
    ARE; it belongs in the lab notebook, not in a file every user has to read past. Flat, one level, every key
    a constructor argument.
    """
    def g(ck, k, default=None):
        return ck[k] if k in ck else default
    return {
        "_class_name": "VoiceModel",
        "_voice_version": "1.0.0",
        "image_encoder": "MahmoodLab/UNI2-h",
        "image_size": 224,
        "feat_dim": 1536,
        "lora_blocks": int(g(ck2, "nblocks", 12)),
        "lora_rank": int(g(ck2, "r", 16)),
        "lora_alpha": int(g(ck2, "alpha", 32)),
        "lora_dropout": float(g(ck2, "dropout", 0.0)),
        "d_model": int(g(ck2, "d_model")),
        "n_layers": int(g(ck2, "n_layers")),
        "n_genes": int(n_genes),
        "crop_px": 201,
        "mpp": 0.2125,
        "token_grid": 16,
    }


def export_one(src, stage, out_dir):
    from safetensors.torch import save_file, load_file
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "opt" in ck:
        print(f"  [{stage}] dropping optimiser state, RNG state and resume bookkeeping", flush=True)
    tensors = flatten(ck, stage)
    if not tensors:
        raise SystemExit(f"{src}: none of {TENSOR_GROUPS[stage]} found -- is this the right stage?")
    dst = os.path.join(out_dir, f"{stage}.safetensors")
    save_file(tensors, dst)

    back = load_file(dst)
    if set(back) != set(tensors):
        raise SystemExit(f"{stage}: tensor names changed on round-trip")
    worst = 0.0
    for k, v in tensors.items():
        if back[k].shape != v.shape or back[k].dtype != v.dtype:
            raise SystemExit(f"{stage}/{k}: shape or dtype changed on round-trip")
        if not torch.equal(back[k], v):
            worst = max(worst, float((back[k].float() - v.float()).abs().max()))
    if worst > 0:
        raise SystemExit(f"{stage}: round-trip is NOT bit-exact (max |delta| = {worst:g}) -- refusing to publish")

    n = sum(v.numel() for v in tensors.values())
    print(f"  [{stage}] {len(tensors)} tensors, {n/1e6:.2f}M params -> {os.path.basename(dst)} "
          f"({os.path.getsize(dst)/1e6:.0f} MB, was {os.path.getsize(src)/1e6:.0f} MB) | round-trip bit-exact",
          flush=True)
    return ck


README = """# {name}

Released weights for VOICE: predicting single-cell gene expression from H&E.

| file | contents |
|---|---|
| `stage1.safetensors` | Stage-1 LoRA adapter for UNI2-h, the two projection towers, scFoundation normalisation |
| `stage2.safetensors` | Stage-2 LoRA overlay, SE(2) decoder, negative-binomial head |
| `config.json` | the architecture |

## Use

```python
from voice.release import load_release
model, se2, cfg = load_release("path/to/this/directory", device="cuda")
```

or run the entry points directly:

```bash
python predict/predict.py --image slide.svs --cells cells.npz --out pred.h5ad \\
    --stage1 {name_lower}/stage1.safetensors --stage2 {name_lower}/stage2.safetensors
```

The image encoder itself is **not** included: VOICE adapts **UNI2-h** (MahmoodLab), which is gated and must be
obtained under its own licence.

```bash
huggingface-cli download MahmoodLab/UNI2-h
```

## Licence

See the repository LICENSE. UNI2-h is licensed separately by its authors.
"""


def main():
    ap = argparse.ArgumentParser(description="Export a training checkpoint as a release artefact.")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="VOICE")
    ap.add_argument("--genes", default=None,
                    help="TSV of the head's gene table (gene_symbol, global_gene_index); copied in, since a "
                         "prediction is unreadable without it")
    ap.add_argument("--force", action="store_true", help="overwrite an existing export")
    a = ap.parse_args()

    if os.path.exists(a.out) and os.listdir(a.out) and not a.force:
        raise SystemExit(f"{a.out} is not empty. Pass --force to overwrite.")
    os.makedirs(a.out, exist_ok=True)
    print(f"[export] {a.name} -> {a.out}", flush=True)

    ck1 = export_one(a.stage1, "stage1", a.out)
    ck2 = export_one(a.stage2, "stage2", a.out)
    n_genes = int(ck2["se2"]["head.mu_lin.weight"].shape[0])
    cfg = config_of(ck1, ck2, n_genes)
    if a.genes and os.path.exists(a.genes):
        import shutil, pandas as pd
        g = pd.read_csv(a.genes, sep="\t")
        # A gene table that does not match the head is worse than none: every prediction would be mislabelled,
        # and silently -- the shapes never disagree because the table is only ever used for naming.
        if len(g) != cfg["n_genes"]:
            raise SystemExit(
                f"--genes has {len(g)} genes but the head emits {cfg['n_genes']}.\n"
                f"  These are different generations of the model: the 6029-gene head (d_model 512) uses "
                f"global_genes_v2.tsv, the 5742-gene head (d_model 384) uses the earlier table.\n"
                f"  Publishing the wrong one mislabels every gene in every prediction.")
        shutil.copy(a.genes, os.path.join(a.out, "genes.tsv"))
        print(f"  [genes] genes.tsv ({len(g)} genes, matches the head)", flush=True)
    else:
        print("  [genes] no --genes given: predictions will have positional gene names. Strongly recommended.",
              flush=True)

    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    lower = a.name.lower().replace(" ", "-")
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(README.format(name=a.name, name_lower=lower))

    leaked = []
    for root, _d, files in os.walk(a.out):
        for fn in files:
            if fn.endswith((".json", ".md")):
                t = open(os.path.join(root, fn)).read()
                for tok in ("/scratch", "/nfs", "/home/", "drjieliu", "zchx", "suyuanw", "yctao"):
                    if tok in t:
                        leaked.append(f"{fn}: {tok}")
    print(f"\n[audit] local paths in the exported metadata: {leaked if leaked else 'none'}", flush=True)
    if leaked:
        raise SystemExit("refusing to finish: the export leaks local paths")
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"[done] {a.out}  ({total/1e6:.0f} MB total, {len(os.listdir(a.out))} files)", flush=True)
    print(f"       {json.dumps(cfg)}", flush=True)


if __name__ == "__main__":
    main()
