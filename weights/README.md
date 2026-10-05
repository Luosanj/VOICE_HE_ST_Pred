# Model weights

Inference accepts either a release directory or the original two checkpoints.

```
voice-23m/
  stage1.safetensors
  stage2.safetensors
  config.json
  genes.tsv
```

```bash
python predict/predict.py --prepared /data/target --release /models/voice-23m --out pred.h5ad
python predict/predict.py --prepared /data/target \
  --stage1 /models/clip_lora_v2_final.pt \
  --stage2 /models/se2_lora_p2v2_noinslide_epoch0.pt --out pred.h5ad
```

Stage 1 supplies the LoRA adapter; Stage 2 overlays its trained adapter tensors and supplies the decoder. `genes.tsv` lists output genes in head order. The UNI2-h base is downloaded separately.

Build reference banks with the same encoder checkpoints used for the query.
