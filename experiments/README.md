# Experiments

Run entries from the repository root with `python -m`. Supply your own data, model, bank, and output paths.

| Entry | Input | Output |
| --- | --- | --- |
| `benchmark.gene_lists` | Measured expression and cell coordinates | Canonical evaluation gene lists |
| `benchmark.eval_crossslide` | Paired target slides and trained model weights | Cross-slide PCC and predictions |
| `benchmark.eval_inslide` | Paired slides and trained model weights | Spatial five-fold PCC and predictions |
| `benchmark.fit_gate` | Reference slides, model weights, and bank | Per-gene fusion weights |
| `ablation.encoder_features` | Prepared slides, frozen UNI2-h or aligned encoder weights | Query features or per-slide reference banks |
| `ablation.retrieval_alignment` | Matched query features and banks for each encoder variant | Covered-gene retrieval PCC |
| `ablation.gate_methods` | Spatial out-of-fold branch predictions, band IDs, canonical gene lists | Grid/MLP PCC tables |
| `ablation.cache_references` | Model weights, same-tissue prepared slides, excluding banks | Reference A/R/Y NPZ caches |
| `ablation.beta_stability` | Reference caches and original target branch caches | Beta support, SD, pair correlations, pooled/global comparisons |
| `ablation.donor_split` | Training slides with donor labels, target slides, head gene table | Donor-grouped train/held-out split and Stage-2 slide lists |
| `ablation.donor_calibration` | Split, reference caches and target predictions for an all-slide and a held-out Stage 2 | Gate PCC tables (in-sample vs held-out references), gate and reference-slide statistics |
| `ablation.mask_sensitivity` | Prepared target, fixed weights/bank/gate, canonical genes | 13 mask conditions, A/R/fused predictions, PCC and mask statistics |
| `downstream.pancreas` | Paired expression/predictions, frozen domains, Reactome GMT | Domain agreement, shared DEGs, gene/pathway block correlations |
| `downstream.proliferation` | Paired breast data and official subtype labels | Six-gene G2/M AUROC and score PCC |
| `downstream.subtype_variation` | Paired breast data and official subtype labels | Standardized subtype differences and equal-type PCC |

## Expression prediction evaluation

```bash
python -m experiments.benchmark.gene_lists --slides slides.yaml --out gene_lists/
python -m experiments.benchmark.eval_crossslide --slides slides.yaml --global_genes genes.tsv \
  --gene_lists gene_lists/ --save_preds predictions/ --out crossslide.csv
python -m experiments.benchmark.eval_inslide --slides slides.yaml --global_genes genes.tsv --out inslide.csv
```

Input YAML:

```yaml
slides:
  - name: target
    dir: /data/target
    mpp: 0.25
```

Each directory contains an image, `cells.npz`, `expression.npz`, and `genes.tsv`. Prepared-input formats are described in the [README](../README.md#prediction). Output: all-gene and HVG/SVG PCC tables, with optional prediction NPZs. `benchmark/figures` plots these outputs.

## Retrieval alignment

Extract the same reference slides and query cells for each encoder variant. The query is excluded from each bank. `--mode frozen` uses UNI2-h; `--mode stage1 --stage1 checkpoint.pt` uses its contrastive adapter; `--mode stage2` accepts the release or the two checkpoints.

```bash
python -m experiments.ablation.encoder_features --prepared /data/reference --mode frozen \
  --global_genes genes.tsv --bank_dir bank/frozen
python -m experiments.ablation.encoder_features --prepared /data/target --mode frozen \
  --global_genes genes.tsv --out query/frozen.npz
python -m experiments.ablation.retrieval_alignment --slides retrieval_slides.yaml --out results/retrieval
```

Repeat extraction for Stage 1 using separate bank/query paths. Evaluation YAML: `slides` list with `name`, optional `exclude`, and `variants` mapping labels to `query` and `bank` paths. It verifies matching reference-slide names, query cells, expression, and genes, then scores genes covered by every variant. Defaults: K=200, tau=0.03.

## No-Stage-1 training

Use your own prepared training cache and offline UNI2-h cache. No bank construction, scFoundation embeddings, or Stage-1 checkpoint is required. Set `VOICE_HF_HOME`, `VOICE_V2_ROOT`, and a separate `VOICE_CKPT_DIR` for this run:

```bash
export VOICE_HF_HOME=/models/hf_cache
export VOICE_V2_ROOT=/data/training_cache
export VOICE_CKPT_DIR=/models/nostage1
NPROC=1 bash train/run_nostage1.sh
```

This calls the existing Stage-2 trainer with `--lora_ckpt none`, empty `init_from`, two epochs, no validation split, and the five in-slide benchmark slides excluded. UNI2-h base weights stay frozen; freshly initialized LoRA and SE(2) are trained together (not a frozen-encoder ablation). Defaults match the ablation: last 12 encoder blocks, r=16, alpha=32, dropout=0.05, freeze_frac=0.2, d_model=512, six spatial layers, lr_lora=3e-5, lr_se2=1e-4. The first three adapted blocks remain frozen with zero-B adapters. `NPROC` is the number of GPUs; one is the default. Use the same `NPROC` as the comparison run to match optimizer-step counts.

Required cache files: `manifest_v2.csv` (`sample`, boolean `in_training`), `global_genes_v2.tsv` (`gene_symbol`, `global_gene_index`), `crops_raw/<slide>/crops.u8`, `crops_raw/<slide>/maskW.f16`, and `sample_meta/<slide>/{expression.npz,genes.tsv,patch_cell_boundaries.npz}`. Crop/mask files are NumPy-format arrays without a `.npy` suffix, aligned with expression and boundary rows; shapes and mask normalization follow the main training README. Stage-1-only `cell_emb_scf` inputs are not needed. For your own held-out set, use `EXCL_INSLIDE=0 EXCLUDE_SLIDES=heldout.txt`; `SLIDES=train.txt` selects exact training slides.

Output: `VOICE_CKPT_DIR/se2_lora_abl_nos1_{latest,epoch*,final}.pt`. Change `TAG` to separate runs. Resume only from the same no-Stage-1 run; checkpoints retain training arguments and decoder/LoRA states (latest also stores optimizer and RNG states).

The existing prediction entry accepts these weights via `--stage1 none --stage2 /models/nostage1/se2_lora_abl_nos1_final.pt`. Supply your already prepared bank and beta/gate paths as usual; this entry does not build or modify them:

```bash
python predict/predict.py --prepared /data/target --stage1 none \
  --stage2 /models/nostage1/se2_lora_abl_nos1_final.pt --genes /data/training_cache/global_genes_v2.tsv \
  --bank /data/nostage1_bank --gate /data/nostage1_gate.json --out /results/nostage1.h5ad --save_embeddings
```

Query features and reference-bank features must use the same encoder. Omit `--bank` and `--gate` for direct-only prediction.

## Fusion methods

A gate-comparison YAML maps slide keys to `predictions` and `gene_list` paths. NPZ arrays: `Apred`, `Rr`, `Ylog`, `band`, and `genes`. Both branches must already be out of fold. Optional `transfers` entries contain `source: [slide_keys]` and `target: slide_key`.

```bash
python -m experiments.ablation.gate_methods --slides gate_slides.yaml --out results/gate_methods
python -m experiments.ablation.cache_references --slides references.yaml \
  --release /models/voice-23m --global_genes genes.tsv --out reference_cache
python -m experiments.ablation.beta_stability --references reference_cache \
  --predictions /data/target_branch_caches --targets targets.json --out results/beta
```

Reference YAML: `slides` list with `name`, `tissue`, `dir`, `bank`, and optional `exclude` names. Beta analysis selects up to four reference caches per tissue by cell count. `--targets` supplies a JSON mapping cache stems to `[tissue, per-gene TSV]`. Optional `--exclude`, `--duplicates`, and `--specimen_pairs` supply reference exclusions and paired-specimen groups.

Target cache layout: `Across_<stem>.npz` (`Apred`, `Ylog`, `gid`), `Rcross_<stem>__full.npz` (`R`, `panel`, `covered`), and `pergene_pcc/<stem>.tsv` with `gene`, `global_id`, `hvg_rank`, `svg_rank`, `pcc_A`, `pcc_STACK`, and `beta`.

## Donor-held-out gate calibration

Compares fusion weights fitted on Stage-2 training slides with weights fitted on held-out donors. Arms: (a) Stage 2 on all slides, weights from its training slides; (b) Stage 2 without the held-out donors, weights from its remaining same-tissue slides; (c) the same model, weights from the held-out donors.

```bash
python -m experiments.ablation.donor_split --slides train_slides.csv --targets targets.csv \
  --global_genes genes.tsv --val_frac 0.4 --out split
SLIDES=split/stage2_slides.txt EPOCHS=1 VAL_FRAC=0 TAG=heldout bash train/run_phase2.sh
python -m experiments.ablation.donor_calibration --config calibration.yaml --out results/calibration
```

`train_slides.csv`: `slide`, `dir`, `donor`, `tissue` for every Stage-2 training slide (`tissue` empty outside the evaluation tissues; slides of one specimen share a `donor`). `targets.csv`: `tissue`, `dir`. The split keeps the union of trained genes unchanged, never splits a donor, then maximizes same-tissue target-gene coverage of both parts and matches `--val_frac` by slides, then cells. Outputs: `split.json`, `stage2_slides.txt`, `heldout_slides.txt`. `train/run_phase2.sh` also accepts `EXCLUDE_SLIDES=split/heldout_slides.txt`.

For each Stage-2 model, supply reference caches (`ablation.cache_references`, one per reference bank exclusion) and target predictions from that model's weights and banks. Calibration YAML:

```yaml
split: split/split.json
targets:
  lung: {dir: /data/lung_target, gene_list: gene_lists/lung_target.tsv}
models:
  all_slides:
    predictions: {lung: pred/all/lung.h5ad}
    references: {slide_out: cache/all_slide_out, donor_out: cache/all_donor_out}
  held_out:
    predictions: {lung: pred/heldout/lung.h5ad}
    references: {slide_out: cache/heldout_slide_out, donor_out: cache/heldout_donor_out}
```

`slide_out` caches exclude only the reference slide from its bank; `donor_out` caches also exclude the other slides of its donor. Predictions are `predict.py --bank` outputs without `--gate` (`X` direct, `layers['R']` retrieval), or NPZs with `A`, `R`, `gid`, `Ylog` over the gene list. Each reference gets 41-point grid weights on covered target-panel genes; beta is their mean over a reference set (`largest4`, `all`, `train`, `val`); genes without beta keep the direct branch. Outputs: `results.tsv` (7 PCC metrics, gated genes, mean beta, references per gene, SD across references), `references.tsv` (direct, retrieval and best-mix PCC per reference slide), `pergene.tsv.gz`, and `summary.md`, with direct, retrieval, fixed 0.5 and target-fitted oracle rows.

## Mask sensitivity

```bash
python -m experiments.ablation.mask_sensitivity --prepared /data/target \
  --stage1 /models/clip_lora_v2_final.pt --stage2 /models/stage2_bundle.pt \
  --global_genes genes.tsv --gene_list target_genes.tsv \
  --bank bank/target_tissue --gate target_gate.json --out results/masks
```

`dataset.json` supplies `um_per_pixel_output`; masks are rasterized at 224×224. Defaults retain complete 256-pixel tiles until at least 30,000 cells are sampled, seed 20260929. Conditions are original plus 1/2-µm erosion, dilation, and four translations. Bank expression and beta stay fixed. Query and bank use the same encoder checkpoints.

## Paired biological inputs

Supply `--data paired.npz` with `Ylog`, `pred` (or `Spred`), `genes`, and `pos` in source pixels. Prediction columns must match genes. `--data paired.npz --predictions fused.npy` overrides the cached predictions with the final fused matrix in the same row/column order. Alternatively supply `--predictions pred.npy`, `--expression expression.npz` (raw CSR counts), `--gene_alignment genes.csv` (`gene`, `prediction_column`, `truth_column`), and `--cells cells.csv.gz` (`expr_row`, `cell_type`, `x_pixel`, `y_pixel`) in prediction-row order. Fused predictions are used as stored.

```bash
python -m experiments.downstream.pancreas --data pancreas.npz --mpp 0.2736318407960199 \
  --gene_list pancreas_genes.tsv --frozen_model frozen_model.npz \
  --reactome c2.cp.reactome.v2026.1.Hs.symbols.gmt --out results/pancreas
python -m experiments.downstream.proliferation --data breast.npz --cells cells.csv.gz \
  --mpp 0.363788 --out results/proliferation
python -m experiments.downstream.subtype_variation --data breast.npz --cells cells.csv.gz \
  --mpp 0.363788 --out results/subtypes
```

Pancreas uses measured-data PCA (30 components), 14 spatial neighbors plus self, six clusters, and fixed projection of predictions. Pass `--frozen_model` to reuse a fitted domain model; omit it to fit one on measured expression. Paired DEG tests compare region versus rest within 500-µm blocks with at least five cells per group; BH q<0.05 defines shared DEGs. Reactome scores use measured whole-slide means/SDs, SD floor 0.1, and at least five covered genes per pathway.

Breast proliferation uses CCND1, CENPF, HMGA1, MKI67, SQLE, and TOP2A, standardized separately in measured and predicted expression across tumor epithelial cells. AUROC compares `Prolif_Invasive_Tumor` with `Invasive_Tumor`.

Subtype differences divide each subtype mean difference by pooled within-subtype SD. Genes require ≥20 detections, ≥1% detection within the pair, and nonzero measured SD. All subtype pairs are averaged within each broad type, then broad types are averaged equally. `--groups` overrides the default five-type annotation mapping.
