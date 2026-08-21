# local/ — this cluster's filled-in paths

Everything in here except this file is git-ignored. It exists so a collaborator on this cluster can run the
code without filling anything in.

```bash
cd spatial_fm
source local/env.sh          # paths, interpreter, HuggingFace cache
```

`env.sh` sets `VOICE_CONFIG=local/config.yaml`, so every entry point resolves its paths from there.

## Predict on a slide

Two input layouts; both write an AnnData.

```bash
# ① a slide already prepared in the corpus layout
#    (patch_cell_boundaries.npz + manifest.csv.gz + patches/)
$PY predict/predict.py --prepared $VOICE_TESTSET_ROOT/Xenium_Lung/hest_TENX141 \
    --release $VOICE_RELEASE --save_embeddings --out lung.h5ad --workers 8

# ② a whole-slide image of your own
$PY predict/segment.py --image slide.svs --out cells.npz --mpp 0.25
$PY predict/predict.py --image slide.svs --cells cells.npz --mpp 0.25 \
    --release $VOICE_RELEASE --save_embeddings --out slide.h5ad
```

`--save_embeddings` adds `obsm["X_voice"]`, the 1536-d per-cell feature. **Use that for downstream tasks**
(cell typing, clustering, integration) rather than the predicted genes:

```python
import scanpy as sc
a = sc.read_h5ad("lung.h5ad")
sc.pp.neighbors(a, use_rep="X_voice")
sc.tl.leiden(a)
```

`--embeddings_only` skips the gene head if you only want the features.

## Prepared slides available here

```
$VOICE_TESTSET_ROOT/
    Xenium_Lung/hest_TENX141                              160,444 cells
    Xenium_Pancreas/hest_TENX140                          234,856
    Xenium_kidney/kidney_protein                          465,534
    Xenium_ovary/benchmark_ovary_io_...                   247,636
    Xenium_breast_cancer/benchmark_breast_prime5k_...      699,078
```

## Throughput

The prepared path decodes one PNG per cell, so it is CPU-bound. On a 4-core node that is ~22 cells/s
(≈2 h for a 160k-cell slide). Raise `--workers` on a machine with more cores.

## Weights

`$VOICE_RELEASE` points at `stage1.safetensors` + `stage2.safetensors` + `config.json` + `genes.tsv`. A copy
also sits in `weights/voice-23m/` inside the repo (git-ignored). The image encoder is **not** included: VOICE
adapts MahmoodLab/UNI2-h, which is gated — `env.sh` points `HF_HOME` at a cache that already has it.

## Benchmark

`benchmark/*.py` scores a model against measured expression and needs slides in the benchmark layout
(`image + cells.npz + expression.npz + genes.tsv`, see `benchmark/dataset.py`) described in
`local/slides.yaml`. That is a different layout from the prepared slides above; fill it in when you have
paired slides to score.
