# spatial_fm — both training phases, integrated + validated
```
bash run_phase1.sh                    # 2-GPU, 3 epochs, 10% spatial val band, auto-resume
bash run_phase2.sh                    # 3-GPU, 75 slides, 2 epochs, 10% spatial val band, auto-resume
python train_phase2.py --smoke        # 2 slides / 8 steps end-to-end (verified)
python train_phase2.py --check_only   # mask gate + one fwd/bwd, no training
```

Both `run_*.sh` **hard-code the hyper-parameters the shipped 23M runs actually used** 
Both phases use the **same split rule** (per-slide x-quantile band + margin, patch-level in Phase-2 / cell-level in
Phase-1) so a Phase-2 run inherits a Phase-1 backbone that never saw the val region either.

## Phase-1 specifics

- **Why the margin matters more here**: crops are 224 px wide, so two cells closer than 224 px have *physically
  overlapping images*. A random cell split would put near-duplicate crops in train and val. `--val_margin` is
  asserted `>= 224` at startup; the default 256 gives pixel-disjoint train/val crops.
- **Metrics**: val InfoNCE (selects the checkpoint) plus in-batch retrieval **R@1 both directions** (he→scF,
  scF→he) — the interpretable number.
- **`--val_batch` is fixed with `drop_last`** (default = `eff_batch`) and the val subset is seeded: InfoNCE depends
  on the number of in-batch negatives, so it would not be comparable across steps otherwise.
- **scF mu/sd are computed over the TRAIN split only**, cached at a split-aware path so a val run never silently
  reuses the all-cell statistics. First launch streams the scF bank once (~130 GB); pass
  `--scf_stats $VOICE_V2_ROOT/scf_stats.npz` to reuse the existing all-cell stats and skip it (the numeric
  difference from 10 % of cells is negligible, but it does technically touch val inputs).
- **`x_pixel.npy` cache**: the split needs each cell's x, which lives in `patch_cell_boundaries.npz` alongside the
  per-cell polygon vertices — pulling it out costs ~11 s/slide, i.e. ~15 min *per launch*. `_cell_x` caches it to a
  131 MB set of `.npy` (already pre-built), taking corpus indexing from **894 s → 21 s**.

## Why a spatial band, not random 10% of cells

The training unit is a 256 px patch of ≤256 **neighbouring** cells and the SE(2) decoder consumes that neighbourhood.
Under a random cell split the *same patch* would hold both train and val cells, so the model sees a val cell's exact
neighbourhood while training — val loss then just tracks train loss. Cells are also strongly autocorrelated (SVG
Moran's I up to ~0.9). Band + margin removes both, and matches the in-slide benchmark protocol
(`se2_lora_v2_incv.py`: vertical bands + inner-val strip).

Held-out **slides** were rejected for a different reason: 9 tissues have ≤2 slides and 5 have exactly 1, so a
slide-level val would delete whole tissues from training. Cross-slide generalization is already measured by the 5
held-out benchmark slides + 5 cross-slide test slides.


