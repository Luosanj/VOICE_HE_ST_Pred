"""The crop / mask geometry the model was trained under. Every number here is a contract, not a preference.

VOICE was trained on Xenium H&E at the vendor's morphology resolution. Per cell:

    crop_px     = 201    source pixels, a square centred on the cell centroid
    output_size = 224    the crop is resized to 224 x 224 before the encoder
    token grid  = 16x16  UNI2-h at 224 px with patch 14 emits 16*16 = 256 patch tokens
    mask        = the cell's polygon rasterised in the 224-px frame, area-pooled to 16x16, summing to 1

201 px at 0.2125 um/px (Xenium morphology) is a field of view of ~42.7 um. What has to match on YOUR slide is
that PHYSICAL size, not the pixel count: at a different resolution, pass --mpp and crop_px is rescaled so the
crop still covers ~42.7 um. Feeding the model a 201-px crop from a 40x scan (~0.25 um/px) shows it a different
amount of tissue than it was trained on, and the predictions degrade quietly rather than failing.

The pooled feature is a cell-mask weighted average over the 256 patch tokens (`voice.encoder.pooled_feat`),
so the mask decides which tokens speak for the cell. A degenerate polygon (fewer than 3 vertices, or zero area
after rasterisation) falls back to the single centre token, matching training.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

CROP_PX_REF = 201.0        # source pixels at the reference resolution
OUTPUT_SIZE = 224          # encoder input side
TOKEN_GRID = 16            # UNI2-h patch-token grid at 224 px
MPP_REF = 0.2125           # um per pixel of the reference (Xenium morphology) images
FOV_UM = CROP_PX_REF * MPP_REF     # ~42.7 um


def crop_px_for(mpp: float | None) -> float:
    """Source-pixel crop side that reproduces the training field of view at YOUR resolution."""
    if mpp is None:
        return CROP_PX_REF
    return FOV_UM / float(mpp)


def polygon_mask(vx, vy, output_size: int = OUTPUT_SIZE, grid: int = TOKEN_GRID) -> np.ndarray:
    """Rasterise one cell polygon (crop-local pixel coords) and area-pool it to [grid, grid], summing to 1.

    Identical to the training-time construction: fill the polygon at full crop resolution, average within each
    token block, then normalise. Degenerate polygons fall back to the centre token.
    """
    tok = output_size // grid
    w = np.zeros((grid, grid), np.float32)
    if len(vx) >= 3:
        im = Image.new("L", (output_size, output_size), 0)
        ImageDraw.Draw(im).polygon(list(zip(np.asarray(vx).tolist(), np.asarray(vy).tolist())), fill=1)
        w = np.asarray(im, np.float32).reshape(grid, tok, grid, tok).mean((1, 3))
    if w.sum() <= 1e-6:
        w[grid // 2, grid // 2] = 1.0
    return w / w.sum()


def tile(pos: np.ndarray, patch_size: int = 256):
    """Group cells into non-overlapping square tiles so the SE(2) decoder sees a local neighbourhood.

    pos is [N,2] = (y, x) in slide pixels. Every cell lands in exactly one tile, so every cell is predicted
    exactly once -- the same tiling the cross-slide evaluation uses.
    """
    cy, cx = pos[:, 0], pos[:, 1]
    H, W = cy.max() + patch_size, cx.max() + patch_size
    out = []
    for y0 in np.arange(0, H, patch_size):
        rm = (cy >= y0) & (cy < y0 + patch_size)
        if not rm.any():
            continue
        cyr = np.where(rm)[0]; cxr = cx[rm]
        for x0 in np.arange(0, W, patch_size):
            cm = (cxr >= x0) & (cxr < x0 + patch_size)
            if cm.any():
                out.append(cyr[cm])
    return out
