# VOICE

Predicting single-cell gene expression from H&E.

VOICE is a three-stage model. **Stage 1** adapts a pathology foundation encoder (UNI2-h) with LoRA and aligns
each cell's image to a single-cell expression embedding by contrastive learning. **Stage 2** decodes that
representation with an SE(2)-equivariant transformer over the cell's spatial neighbourhood and a
negative-binomial head, giving expression for a 6,029-gene panel. **Stage 3** fuses this direct prediction with a
retrieval prediction using a per-gene weight fitted on reference slides.

This repository contains the training code, an inference entry point for your own H&E, and the benchmark code.

## Install

```bash
git clone <this repo> && cd voice
pip install -r requirements.txt
```

UNI2-h is a **gated** model: request access at `huggingface.co/MahmoodLab/UNI2-h`, then

```bash
huggingface-cli download MahmoodLab/UNI2-h
```

Fill in `configs/default.yaml` (or export `VOICE_HF_HOME`, `VOICE_CKPT_DIR`, ...). Only the entries your entry
point needs have to be set; `voice/paths.py` lists which those are and errors with the variable name if one is
missing.

## Predict

Point `--release` at a model directory (see `weights/README.md`). There are two ways to give it a slide.

**A whole-slide image.** Segment the nuclei, then predict:

```bash
python predict/segment.py --image slide.svs --out cells.npz --mpp 0.25
python predict/predict.py --image slide.svs --cells cells.npz --mpp 0.25 \
    --release weights/voice-23m --out pred.h5ad --save_embeddings
```

**A slide already prepared in the corpus layout** — `patch_cell_boundaries.npz` + `manifest.csv.gz` +
`patches/`, which is what the training and benchmark data look like:

```bash
python predict/predict.py --prepared /data/slides/my_slide \
    --release weights/voice-23m --out pred.h5ad --save_embeddings
```

The prepared path needs no `--mpp`: the crops were cut at the right physical size when the slide was built and
the polygons are already in crop coordinates. It is also exactly reproducible — the same PNG bytes reach the
encoder every run. `predict/inputs.py` documents both layouts.

Output is an AnnData: `X = log1p(mu)` over the model's gene head, `obsm["spatial"]` = cell centroids.

### Embeddings, for downstream tasks

`--save_embeddings` adds `obsm["X_voice"]`, the 1536-d per-cell feature the gene head reads from. **That is the
representation to use for cell-type classification, clustering or integration** — not the 6029 predicted genes,
which are a lossy view of the same vector. `--embeddings_only` skips the gene head altogether.

```python
import scanpy as sc
a = sc.read_h5ad("pred.h5ad")
sc.pp.neighbors(a, use_rep="X_voice")
sc.tl.leiden(a)                     # or train a classifier on a.obsm["X_voice"]
```

### Two things decide whether the numbers are meaningful

*Resolution* (whole-slide path only). The model sees a fixed **physical** field of view of about 42.7 µm per
cell — 201 px at the Xenium morphology resolution of 0.2125 µm/px. Pass `--mpp` and the crop is rescaled to
match. Getting this wrong does not raise an error; it shows the model a different amount of tissue and the
predictions degrade quietly.

*Cell boundaries.* The per-cell feature is a **mask-weighted** average over the encoder's 256 patch tokens, so
the boundary decides which image evidence is attributed to this cell rather than its neighbours. Without
polygons the mask falls back to the centre token, which works but is measurably worse. If you have
segmentations from your own pipeline, write them into `cells.npz` and skip `segment.py`.

### What `predict.py` does not do

It runs the direct branch (Stages 1–2) only. The retrieval branch and the Stage-3 gate need a reference bank of
millions of embedded cells from the same tissue, which cannot ship with the code; `benchmark/` shows how to
build one from your own reference slides.

## Train

```bash
bash train/run_phase1.sh     # contrastive, 2 GPU
bash train/run_phase2.sh     # gene supervision, 3 GPU
```

Both scripts hard-code the hyper-parameters the released models were actually trained with. Both hold out a
spatial band per slide with a margin, so a Phase-2 run inherits a Phase-1 backbone that never saw the validation
region either. See `docs/training.md` for the split rule, the resume behaviour and the smoke tests.

Phase 1 needs precomputed scFoundation cell embeddings as its contrastive target; Phase 2 and inference do not.

## Benchmark

`benchmark/` runs the paper's evaluations **on slides you supply**. A benchmark slide is an H&E image paired
with a spatial assay on the same tissue; `benchmark/dataset.py` documents the four files that make one, and
describes them in a small YAML:

```yaml
slides:
  - name: my_breast_slide
    dir:  /data/benchmark/my_breast_slide
    mpp:  0.25
```

```bash
python benchmark/gene_lists.py     --slides slides.yaml --out gene_lists/
python benchmark/eval_crossslide.py --slides slides.yaml --global_genes genes.tsv \
                                    --gene_lists gene_lists/ --save_preds preds/ --out cross.csv
python benchmark/eval_inslide.py   --slides slides.yaml --global_genes genes.tsv --out inslide.csv

python benchmark/figures/spatial_heatmap.py --preds preds/my_breast_slide.npz --gene ACTA2 --out heat.png
python benchmark/figures/pcc_boxplot.py     --preds VOICE=preds/my_breast_slide.npz --out box.png
```

- `gene_lists.py` builds the canonical, model-independent HVG/SVG lists a slide is scored on — same genes for
  every method being compared. It reads the vendor's `cell_feature_matrix.h5` feature ids when the slide
  provides one, so that **antibody channels are excluded**: on a protein add-on panel they otherwise take over
  the variance ranking and are scored as if they were genes, and they cannot be spotted by name (an anti-CD3E
  channel is called "CD3E", exactly like the RNA).
- `eval_crossslide.py` — zero-shot: nothing on the test slide is fitted.
- `eval_inslide.py` — five contiguous bands, decoder refit per fold, encoder frozen. **Not zero-shot**; do not
  pool these numbers with the cross-slide ones.
- `figures/` — spatial heat maps (measured vs predicted, within-panel z-scores) and per-gene PCC box plots.

A gene the model cannot emit scores 0 and stays in the denominator, so methods with different output spaces
stay comparable.

## Repository layout

```
voice/           the package: encoder + LoRA, SE(2) decoder, NB head, gene space, IO
  paths.py       every environment-dependent path, resolved from env vars or configs/default.yaml
  encoder.py     UNI2-h + LoRA + cell-mask pooling  (the only thing inference needs)
predict/         segment -> crop -> predict; inputs.py holds the two slide layouts
weights/         released weights go here (git-ignored; see weights/README.md)
train/           Phase 1 and Phase 2, with the shipped hyper-parameters
benchmark/       paper evaluations, canonical gene lists, figures
tests/           parity test: the packaged forward reproduces the published predictions
```

## Citation

To appear.
