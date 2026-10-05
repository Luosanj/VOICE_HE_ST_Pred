"""Read slide images and extract crops. Input: TIFF, SVS, NDPI, or raster images. Output: image dimensions, resolution, and RGB crops."""
from __future__ import annotations
import numpy as np


def _try_openslide(path):
    try:
        import openslide
    except ImportError:
        return None
    try:
        return openslide.OpenSlide(str(path))
    except Exception:
        return None


class SlideReader:
    def __init__(self, path, level: int = 0):
        self.path = str(path); self.level = level
        self._os = _try_openslide(path)
        self._arr = None
        if self._os is not None:
            self.width, self.height = self._os.level_dimensions[level]
            self.backend = "openslide"
            self.mpp = self._guess_mpp_openslide()
            return
        self._arr = self._read_array(path)
        if self._arr.ndim == 2:
            self._arr = np.stack([self._arr] * 3, -1)
        self._arr = self._arr[..., :3]
        self.height, self.width = self._arr.shape[:2]
        self.backend = "array"
        self.mpp = None

    @staticmethod
    def _read_array(path):
        p = str(path).lower()
        if p.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
            try:
                import tifffile
            except ImportError:
                raise RuntimeError(
                    f"{path} is a TIFF but neither OpenSlide nor tifffile is installed.\n"
                    "  pip install tifffile        (plain / OME-TIFF)\n"
                    "  pip install openslide-python  + the OpenSlide C library (SVS, NDPI, pyramidal TIFF)")
            return np.asarray(tifffile.imread(path))
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required to read images: pip install pillow")
        Image.MAX_IMAGE_PIXELS = None
        return np.asarray(Image.open(path).convert("RGB"))

    def _guess_mpp_openslide(self):
        import openslide
        for k in (openslide.PROPERTY_NAME_MPP_X, "aperio.MPP", "tiff.XResolution"):
            v = self._os.properties.get(k)
            if v:
                try:
                    f = float(v)
                    return 10000.0 / f if k == "tiff.XResolution" and f > 1000 else f
                except ValueError:
                    pass
        return None

    def region(self, x0: int, y0: int, side: int) -> np.ndarray:
        """uint8 [side, side, 3], zero-padded where the window leaves the image."""
        out = np.zeros((side, side, 3), np.uint8)
        xs, ys = max(0, x0), max(0, y0)
        xe, ye = min(self.width, x0 + side), min(self.height, y0 + side)
        if xe <= xs or ye <= ys:
            return out
        if self.backend == "openslide":
            tile = np.asarray(self._os.read_region((xs, ys), self.level, (xe - xs, ye - ys)).convert("RGB"))
        else:
            tile = self._arr[ys:ye, xs:xe]
        out[ys - y0:ye - y0, xs - x0:xe - x0] = tile
        return out

    def crops(self, cx, cy, crop_px: float, output_size: int = 224) -> np.ndarray:
        """uint8 [n, 3, output_size, output_size], one crop per (cx, cy) centre, in the given order."""
        from PIL import Image
        side = int(round(crop_px))
        half = side // 2
        out = np.empty((len(cx), 3, output_size, output_size), np.uint8)
        for i, (x, y) in enumerate(zip(np.asarray(cx), np.asarray(cy))):
            reg = self.region(int(round(x)) - half, int(round(y)) - half, side)
            if side != output_size:
                reg = np.asarray(Image.fromarray(reg).resize((output_size, output_size), Image.BILINEAR))
            out[i] = np.ascontiguousarray(reg.transpose(2, 0, 1))
        return out

    def close(self):
        if self._os is not None:
            self._os.close()
