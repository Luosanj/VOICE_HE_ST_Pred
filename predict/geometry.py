"""Build crops, cell masks, and spatial tiles. Input: microns per pixel, polygon vertices, and coordinates. Output: 55-um crop sizes, normalized 16x16 masks, and tile indices."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

FOV_UM = 55.0
MPP_REF = 0.2738
CROP_PX_REF = 201.0
OUTPUT_SIZE = 224
TOKEN_GRID = 16
MPP_OUT = FOV_UM / OUTPUT_SIZE


def crop_px_for(mpp: float | None) -> float:
    """Return crop pixels for a 55-um field of view; use 201 pixels when mpp is absent."""
    if mpp is None:
        return CROP_PX_REF
    return FOV_UM / float(mpp)


def polygon_mask(vx, vy, output_size: int = OUTPUT_SIZE, grid: int = TOKEN_GRID) -> np.ndarray:
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
