"""Tests for ``raster_tiles.render_geotiff_tile`` — renders XYZ tiles from
an uploaded GeoTIFF so it can be compared against an OlmoEarth inference
layer side-by-side.

Uses a synthetic 1-band GeoTIFF created in a tmp dir so tests stay offline.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.raster_tiles import render_geotiff_tile  # noqa: E402


@pytest.fixture
def single_band_geotiff(tmp_path: Path) -> Path:
    """Create a 256×256 EPSG:4326 GeoTIFF over [0,1]×[0,1] with a left/right
    gradient. Makes the tile-render math easy to check."""
    path = tmp_path / "single_band.tif"
    data = np.tile(np.linspace(0.0, 100.0, 256, dtype=np.float32), (256, 1))
    transform = from_bounds(0.0, 0.0, 1.0, 1.0, 256, 256)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=256, width=256, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int(
        (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    )
    return x, y


def test_render_geotiff_tile_returns_valid_png(single_band_geotiff: Path) -> None:
    # Tile z=14 over (0.5, 0.5) lat/lon — roughly in-bounds for the [0,1]×[0,1] raster.
    x, y = _latlon_to_tile(0.5, 0.5, 14)
    png = render_geotiff_tile(single_band_geotiff, 14, x, y)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    assert im.size == (256, 256)


def test_render_geotiff_tile_paints_colormap_on_valid_pixels(
    single_band_geotiff: Path,
) -> None:
    x, y = _latlon_to_tile(0.5, 0.5, 14)
    png = render_geotiff_tile(single_band_geotiff, 14, x, y)
    arr = np.array(Image.open(io.BytesIO(png)))
    # At least some pixels should be opaque (inside the raster envelope).
    alphas = arr[..., 3]
    assert int(alphas.max()) > 0, "expected opaque pixels in-bounds"
    # Colormap stops cycle through distinct channels, so not every pixel is pure grey.
    rgb = arr[..., :3]
    assert rgb.max() > 0


def test_render_geotiff_tile_out_of_bounds_is_transparent(
    single_band_geotiff: Path,
) -> None:
    # Tile at (50, 50) lat/lon — far outside the [0,1] raster.
    x, y = _latlon_to_tile(50.0, 50.0, 6)
    png = render_geotiff_tile(single_band_geotiff, 6, x, y)
    arr = np.array(Image.open(io.BytesIO(png)))
    # Fully transparent tile.
    assert int(arr[..., 3].max()) == 0


def test_render_geotiff_tile_handles_multi_band_as_rgb(tmp_path: Path) -> None:
    """3-band GeoTIFF → RGB composite path rather than colormap path."""
    path = tmp_path / "rgb.tif"
    rgb = np.stack(
        [
            np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1)),
            np.full((128, 128), 128, dtype=np.uint8),
            np.tile(np.linspace(255, 0, 128, dtype=np.uint8), (128, 1)),
        ],
        axis=0,
    )
    transform = from_bounds(0.0, 0.0, 1.0, 1.0, 128, 128)
    with rasterio.open(
        path, "w", driver="GTiff", height=128, width=128, count=3, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(rgb)

    x, y = _latlon_to_tile(0.5, 0.5, 14)
    png = render_geotiff_tile(path, 14, x, y)
    im = Image.open(io.BytesIO(png))
    arr = np.array(im)
    # Left column should be red-ish, right column should be blue-ish —
    # picked from the first and last pixels of our linear ramps.
    assert arr[..., 3].max() > 0
    # R channel varies left→right, B channel varies right→left.
    left_r = arr[:, 10, 0].mean()
    right_r = arr[:, 245, 0].mean()
    assert right_r > left_r, "R channel should increase left→right"


def test_render_geotiff_tile_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        render_geotiff_tile(Path("/does/not/exist.tif"), 0, 0, 0)
