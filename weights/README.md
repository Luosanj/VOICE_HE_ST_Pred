# weights/

Model weights are **not** in the repository — `weights/` is git-ignored. This directory is where you put a
released model after downloading it, and where a shared copy is kept for collaborators.

A model is one directory:

```
weights/voice-23m/
    stage1.safetensors     Stage-1 LoRA adapter + projection towers
    stage2.safetensors     Stage-2 LoRA overlay + SE(2) decoder + NB head
    config.json            the architecture
    genes.tsv              the head's 6029 gene symbols, in head order
    README.md
```

Point any entry point at it with `--release weights/voice-23m`.

The image encoder is not included: VOICE adapts **MahmoodLab/UNI2-h**, which is gated and carries its own
licence. Request access on the model page, then `huggingface-cli download MahmoodLab/UNI2-h`.
