# Model weights

Download the model and tissue reference banks from [Hugging Face](https://huggingface.co/zchx/VOICE-23M). Inference accepts a release directory or two checkpoint paths.

```
voice-23m/
  stage1.safetensors
  stage2.safetensors
  config.json
  genes.tsv
  beta/lung.json
  bank/lung/
    manifest.json
    part-00000.h5
    ...
```

```bash
python predict/predict.py --prepared /data/target --release /models/voice-23m --out pred.h5ad
python predict/predict.py --prepared /data/target \
  --stage1 /models/stage1.safetensors \
  --stage2 /models/stage2.safetensors --out pred.h5ad
```

Stage 1 supplies the LoRA adapter; Stage 2 overlays its trained adapter tensors and supplies the decoder. `genes.tsv` lists output genes in head order. The UNI2-h base is downloaded separately.

Build reference banks with the same encoder checkpoints used for the query.

## Released tissue banks

Each tissue directory contains a manifest and compressed HDF5 shards. Shards store `E` as float16 normalized cell features. Under `gene_sets`, each group stores measured global gene IDs, shard-local row indices and float16 standardized expression `Y`. Grouping describes measured gene sets, not source slides. All cell axes remain paired when shuffled.

Expression was standardized within each original reference before pooling. Read it directly; a second z-score changes the retrieval values. Unmeasured genes are excluded from the neighbour pool for that gene. Retrieval uses K=200 and tau=0.03. Features are streamed when the selected bank does not fit in GPU memory.

Published pooled banks omit source identities and do not support source-level exclusion or leave-one-reference-out fitting. For those operations, use `predict/build_bank.py` to build a slide-resolved bank from your own references. Input paths are supplied by the user.

The bank data terms and source credits are provided in [BANK_DATA_TERMS.md](https://huggingface.co/zchx/VOICE-23M/blob/main/BANK_DATA_TERMS.md). The code license does not replace the applicable data or UNI2-h terms.
