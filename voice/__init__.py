"""VOICE: predicting single-cell gene expression from H&E.

    paths.py          every environment-dependent path, from env vars or configs/default.yaml
    encoder.py        UNI2-h + LoRA + cell-mask pooling -- all inference needs
    _se2_arch.py      SE(2)-equivariant decoder over a cell's spatial neighbourhood
    _nb_head.py       negative-binomial output head
    scale_train.py    ScaleHE2Cell: encoder features -> SE(2) decoder -> NB head
    panel.py          real genes vs control probes vs antibody channels; panel <-> head mapping
    metrics_bench.py  per-gene Pearson, Moran's I, HVG/SVG ranking
    genes.py          gene-space bookkeeping used by training
    cache_io.py       corpus readers used by training
    scale_dataset.py  spatial-patch dataset used by training
    util.py           config loading and seeding
"""
