# VOICE

Predict single-cell gene expression from H&E with UNI2-h, a spatial decoder, and reference-cell retrieval.

## Setup

```bash
pip install -r requirements.txt
```

For `predict/segment.py`, also install `cellpose>=4.0` and `opencv-python-headless`. SVS/NDPI/MRXS inputs require OpenSlide and `openslide-python`.

UNI2-h access and weights: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).
Set paths in `configs/default.yaml` or `VOICE_*` environment variables. See [training inputs](#training-and-evaluation) and [model weights](weights/README.md).

## Prediction

Input: a prepared slide with cell crop images, `manifest.csv.gz` (`expr_row`, `patch_path`), and `patch_cell_boundaries.npz` (`expr_rows`, `indptr`, `vertex_x_patch`, `vertex_y_patch`, `x_pixel`, `y_pixel`, `output_size`). Coordinates are `(y,x)` in source pixels; polygons are in crop pixels.

Measured expression, when required, is `expression.npz` (scipy CSR raw counts) with columns listed in `genes.tsv`. RNA/protein panels use vendor `features.h5` for channel filtering.

```bash
python predict/predict.py --prepared /data/target --release /models/voice-23m \
  --out pred.h5ad --save_embeddings
```

For a whole-slide image, supply cell polygons in the same pixel coordinate system and image resolution:

```bash
python predict/segment.py --image slide.svs --mpp 0.25 --out cells.npz
python predict/predict.py --image slide.svs --cells cells.npz --mpp 0.25 \
  --release /models/voice-23m --out pred.h5ad
```

Whole-slide crops cover 55 µm and are resized to 224×224. Prepared crops must already use this field of view.

Output: AnnData `X` with direct log1p predictions or fused gene scores, `obsm["spatial"]` with coordinates, and optional `obsm["X_voice"]` with 1536-dimensional cell features.

## Retrieval and fusion

Input: same-tissue reference slides with measured `expression.npz` and `genes.tsv`; model weights must match the query. Exclude the target and its duplicate sections from references. Genes without retrieval or fitted weights use the direct branch.

```bash
python predict/build_bank.py --release /models/voice-23m --bank_dir bank/lung \
  --global_genes /models/voice-23m/genes.tsv --prepared /data/lung_a /data/lung_b
python experiments/benchmark/fit_gate.py --release /models/voice-23m --bank bank/lung \
  --prepared /data/lung_a /data/lung_b --out gate_lung.json
python predict/predict.py --prepared /data/target --release /models/voice-23m \
  --bank bank/lung --gate gate_lung.json --out pred.h5ad
```

Output: reference bank files, per-gene beta JSON, and AnnData with direct/retrieval layers, beta, and fused `X`. Retrieval uses K=200, tau=0.03, and per-slide standardized log1p reference expression. Fused values are gene-wise scores.

## Training and evaluation

| Path setting | Input/output |
| --- | --- |
| `VOICE_HF_HOME` | UNI2-h HuggingFace cache |
| `VOICE_CKPT_DIR` | Stage-2 checkpoint output |
| `VOICE_DATA_ROOT` | Prepared-slide corpus |
| `VOICE_SCF_DIR` | scFoundation cell embeddings |
| `VOICE_V2_ROOT` | Training cache |

Training cache: `manifest_v2.csv`, `global_genes_v2.tsv`, `crops_raw/<slide>/{crops.u8.npy,maskW.f16.npy}`, `cell_emb_scf/<slide>/scf.f16.npy`, and `sample_meta/<slide>/` with expression, genes, positions, and boundaries. These indices describe user-provided files.

Arrays share expression-row order: crops are uint8 `[N,224,224,3]`, masks are normalized `[N,16,16]`, and scFoundation features are `[N,3072]`. Crops cover 55 µm.

```bash
bash train/run_phase1.sh
LORA_CKPT=/models/stage1.pt bash train/run_phase2.sh
python experiments/benchmark/gene_lists.py --slides slides.yaml --out gene_lists/
python experiments/benchmark/eval_crossslide.py --slides slides.yaml --global_genes genes.tsv \
  --gene_lists gene_lists/ --save_preds predictions/ --out crossslide.csv
python experiments/benchmark/eval_inslide.py --slides slides.yaml --global_genes genes.tsv --out inslide.csv
```

Stage 1 writes `VOICE_V2_ROOT/ckpts/clip_lora_<tag>_*`; Stage 2 writes `VOICE_CKPT_DIR/se2_lora_<tag>_*`. Set `NPROC`, `TAG`, and other script variables for your run. Evaluation writes all-gene and HVG/SVG PCC tables. In-slide evaluation uses spatial five-fold fitting; cross-slide evaluation applies fixed weights.

No-Stage-1 ablation: `bash train/run_nostage1.sh` trains directly from UNI2-h using your prepared cache, without Stage-1 weights or a Stage-2 warm-start. See [inputs and usage](experiments/README.md#no-stage-1-training).

Evaluation, ablation, and downstream inputs, commands, and outputs: [experiments/README.md](experiments/README.md).

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
