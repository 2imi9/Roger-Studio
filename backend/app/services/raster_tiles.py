"""Serve XYZ tiles from uploaded GeoTIFFs so they can be overlaid on the map.

The upload endpoint already lands the raw bytes in ``UPLOAD_DIR/{filename}``.
This module reads them on demand, projects each tile's WGS-84 bounds into the
raster's CRS, windowed-reads the pixels, and returns a 256×256 RGBA PNG.

Two rendering paths, picked per file:

  - **Multi-band** (≥3 bands) — treat the first 3 bands as RGB, percentile-
    stretched to [0, 1] for a natural-looking composite. Good default for
    uploaded Sentinel/Landsat/NAIP snippets.
  - **Single-band** — normalize to [0, 1] via a per-file min/max cached on
    first request, then colormap through the built-in viridis-like gradient.
    Fits NDVI / elevation / mask rasters that ship one band.

A stretch is cached per file so we don't re-read the whole raster for every
tile request.
"""
from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)

# Per-file normalization stats. Keyed by (path, mtime) so re-uploads reset.
_stretch_cache: dict[tuple[str, int], dict[str, Any]] = {}

# Viridis — perceptually uniform, colorblind-friendly, and distinct from both
# terra-draw's amber selection highlight AND the blue-indigo-amber gradient
# that olmoearth_inference uses for encoder embeddings. Using viridis here
# means user-uploaded 1-band rasters (NDVI, CHM, DEM) visually read as data
# instead of aliasing with the bbox selection fill or encoder-embedding
# output. Five stops give a smoother gradient than three.
_DEFAULT_STOPS: list[tuple[str, float]] = [
    ("#440154", 0.0),
    ("#3b528b", 0.25),
    ("#21918c", 0.5),
    ("#5ec962", 0.75),
    ("#fde725", 1.0),
]


def _tile_to_lonlat_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2.0 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_s = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    return (lon_w, math.degrees(lat_s), lon_e, math.degrees(lat_n))


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _colormap(scalar01: np.ndarray, stops: list[tuple[str, float]]) -> np.ndarray:
    h, w = scalar01.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    flat = scalar01.ravel()
    out = rgba.reshape(-1, 4)
    for i in range(flat.size):
        t = flat[i]
        if not np.isfinite(t):
            continue
        # Linear interpolation between the nearest two stops.
        for k in range(len(stops) - 1):
            c0, p0 = stops[k]
            c1, p1 = stops[k + 1]
            if p0 <= t <= p1:
                frac = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
                r0, g0, b0 = _hex_to_rgb(c0)
                r1, g1, b1 = _hex_to_rgb(c1)
                out[i] = (
                    int(r0 + frac * (r1 - r0)),
                    int(g0 + frac * (g1 - g0)),
                    int(b0 + frac * (b1 - b0)),
                    210,
                )
                break
    return rgba


def _compute_stretch(path: Path, n_bands: int) -> dict[str, Any]:
    """Compute per-band 2/98 percentile stretch. Cached by (path, mtime)."""
    key = (str(path), int(path.stat().st_mtime))
    hit = _stretch_cache.get(key)
    if hit is not None:
        return hit

    with rasterio.open(path) as src:
        # Use overviews if present to keep the percentile compute cheap; else
        # downsample by factor 8 on read.
        bands = min(n_bands, src.count)
        sample_shape = (max(1, src.height // 8), max(1, src.width // 8))
        low = []
        high = []
        for b in range(1, bands + 1):
            arr = src.read(b, out_shape=sample_shape, masked=True).astype(np.float32)
            # Drop masked / nodata pixels from the percentile compute.
            valid = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
            if valid.size == 0:
                low.append(0.0)
                high.append(1.0)
                continue
            lo_v, hi_v = np.percentile(valid, [2, 98])
            if hi_v - lo_v < 1e-9:
                hi_v = float(lo_v) + 1.0
            low.append(float(lo_v))
            high.append(float(hi_v))

        info = {
            "low": low,
            "high": high,
            "n_bands": bands,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "bounds": src.bounds,
        }
    _stretch_cache[key] = info
    return info


def render_geotiff_tile(
    path: Path, z: int, x: int, y: int, tile_px: int = 256,
) -> bytes:
    """Render one XYZ tile by reprojecting + windowed-reading the GeoTIFF.

    Returns a 256×256 RGBA PNG. Pixels outside the raster envelope are
    transparent; single-band files are colormapped, multi-band files use
    the first 3 as R/G/B.
    """
    from PIL import Image  # noqa: PLC0415

    if not path.exists():
        raise FileNotFoundError(path)

    info = _compute_stretch(path, n_bands=3)
    src_crs = info["crs"] or CRS.from_epsg(4326)
    tile_w, tile_s, tile_e, tile_n = _tile_to_lonlat_bounds(z, x, y)

    # Empty tile when outside the raster's own WGS-84 footprint.
    left, bottom, right, top = rasterio.warp.transform_bounds(
        src_crs, "EPSG:4326", *info["bounds"],
    )
    if (
        tile_e < left or tile_w > right or tile_n < bottom or tile_s > top
    ):
        img = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # Destination grid for this tile — tile_px pixels in the src CRS over
    # the projected tile extent.
    px_left, px_bottom, px_right, px_top = rasterio.warp.transform_bounds(
        "EPSG:4326", src_crs, tile_w, tile_s, tile_e, tile_n,
    )
    dst_transform = from_bounds(px_left, px_bottom, px_right, px_top, tile_px, tile_px)

    with rasterio.open(path) as src:
        n_bands = src.count
        use_rgb = n_bands >= 3
        read_bands = 3 if use_rgb else 1
        reprojected = np.zeros((read_bands, tile_px, tile_px), dtype=np.float32)
        for bi in range(read_bands):
            rasterio.warp.reproject(
                source=rasterio.band(src, bi + 1),
                destination=reprojected[bi],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=rasterio.warp.Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )

    rgba = np.zeros((tile_px, tile_px, 4), dtype=np.uint8)

    if use_rgb:
        for bi in range(3):
            lo = info["low"][bi]
            hi = info["high"][bi]
            denom = max(1e-9, hi - lo)
            stretched = np.clip((reprojected[bi] - lo) / denom, 0.0, 1.0)
            rgba[..., bi] = (stretched * 255).astype(np.uint8)
        finite = np.isfinite(reprojected).all(axis=0)
        rgba[..., 3] = np.where(finite, 230, 0).astype(np.uint8)
    else:
        lo = info["low"][0]
        hi = info["high"][0]
        denom = max(1e-9, hi - lo)
        stretched = (reprojected[0] - lo) / denom
        stretched = np.where(np.isfinite(reprojected[0]), np.clip(stretched, 0.0, 1.0), np.nan)
        rgba = _colormap(stretched, _DEFAULT_STOPS)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
