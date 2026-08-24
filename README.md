# VOICE

Predicting single-cell gene expression from H&E.

VOICE is a three-stage model. **Stage 1** adapts a pathology foundation encoder (UNI2-h) with LoRA and aligns
each cell's image to a single-cell expression embedding by contrastive learning. **Stage 2** decodes that
representation with an SE(2)-equivariant transformer over the cell's spatial neighbourhood and a
negative-binomial head, giving expression for a 6,029-gene panel. **Stage 3** fuses this direct prediction with a
retrieval prediction using a per-gene weight fitted on reference slides and transferred to the target.

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

### Stage 3: retrieval and the per-gene gate

The direct branch predicts from the image alone. The retrieval branch asks a different question — which cells
in a reference cohort look like this one, and what were they expressing — and averages their measured profiles.
The gate fuses them per gene, `pred_g = beta_g * A_g + (1 - beta_g) * R_g`.

Stage 3 needs reference slides **of the same tissue that have measured expression**, so it is optional: a slide
with no reference cohort still gets the direct branch.

```bash
# 1. embed the reference slides into a bank
python predict/build_bank.py --release weights/voice-23m --bank_dir bank/lung \
    --global_genes weights/voice-23m/genes.tsv --prepared /data/ref/lung_a /data/ref/lung_b

# 2. fit the gate on those references (each is retrieved from a bank that EXCLUDES itself)
python benchmark/fit_gate.py --release weights/voice-23m --bank bank/lung \
    --prepared /data/ref/lung_a /data/ref/lung_b --out gate_lung.json

# 3. predict with both
python predict/predict.py --prepared /data/slides/target --release weights/voice-23m \
    --bank bank/lung --gate gate_lung.json --out pred.h5ad
```

The output then carries `layers["A"]` (direct), `layers["R"]` (retrieval, NaN where no reference measures the
gene), `var["beta"]`, and `X` = the fusion.

Two rules the code enforces rather than trusts you to remember. The bank must **not contain the target** —
retrieving a slide from a bank that includes it finds the cell itself and reports its own label. And the bank
must be embedded with the **same weights** as the query, or query and bank sit in different spaces and the
neighbours mean nothing; the encoder identity is stored in each bank file and checked.

Where beta is fitted is the whole point: fitting it on the target needs the labels being predicted, which is an
oracle, not a method. `benchmark/fit_gate.py` fits on references and transfers.

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
voice/           the package: encoder + LoRA, SE(2) decoder, NB head, retrieval, gate, gene space, IO
  paths.py       every environment-dependent path, resolved from env vars or configs/default.yaml
  encoder.py     UNI2-h + LoRA + cell-mask pooling  (the only thing the direct branch needs)
  retrieval.py   the bank, exact top-K, per-gene-signature cross-slide retrieval
  gate.py        the per-gene fusion weight: fitting, transferring, and the oracle upper bound
predict/         segment -> crop -> predict; inputs.py holds the two slide layouts
weights/         released weights go here (git-ignored; see weights/README.md)
train/           Phase 1 and Phase 2, with the shipped hyper-parameters
benchmark/       paper evaluations, canonical gene lists, figures
tests/           parity test: the packaged forward reproduces the published predictions
```

## Citation

If you find VOICE useful in your research, please cite:

```
@article{luo2026voice,
  title={VOICE: A Vision-Omics Foundation Model Integrating Direct and Retrieval-Based Prediction of In-situ Single-Cell Gene Expression},
  author={Luo, Xin and Tao, Yicheng and Zeng, Haoxuan and Wang, Suyuan and Ouyang, Chenzi and Zhu, Meiqi and Liu, Kai and Chen, Shuibing and Liu, Jie},
  journal={arXiv preprint arXiv:2608.08366},
  year={2026}
}
```

