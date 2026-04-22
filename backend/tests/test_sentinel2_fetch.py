"""Unit tests for :mod:`app.services.sentinel2_fetch`.

Pure-Python bits (shape helpers, timestamp conversion) run offline.
The actual STAC search + band read is marked ``network``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services.sentinel2_fetch import (  # noqa: E402
    image_to_bhwtc,
    timestamp_from_iso,
    _sign,
)


def test_image_to_bhwtc_reshapes_correctly() -> None:
    img = np.zeros((10, 20, 12), dtype=np.float32)
    out = image_to_bhwtc(img)
    assert out.shape == (1, 10, 20, 1, 12)
    assert out.dtype == np.float32


def test_image_to_bhwtc_rejects_wrong_band_count() -> None:
    bad = np.zeros((4, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(H, W, 12\)"):
        image_to_bhwtc(bad)


def test_timestamp_from_iso_converts_to_dmy_zero_indexed_month() -> None:
    # August 22 2025 -> (day=22, month=7 [0-indexed], year=2025).
    assert timestamp_from_iso("2025-08-22T10:12:03Z") == (22, 7, 2025)


def test_timestamp_from_iso_accepts_date_only() -> None:
    assert timestamp_from_iso("2024-06-15") == (15, 5, 2024)


def test_sign_appends_sas_token_correctly() -> None:
    assert _sign("https://x.blob/asset.tif", "sv=abc") == "https://x.blob/asset.tif?sv=abc"
    assert _sign("https://x.blob/asset.tif?foo=1", "sv=abc") == "https://x.blob/asset.tif?foo=1&sv=abc"


# ---------------------------------------------------------------------------
# Network-gated: hits Microsoft Planetary Computer for S2 bands.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_fetch_s2_composite_returns_12_bands_seattle() -> None:
    from app.services.sentinel2_fetch import fetch_s2_composite

    # Small Seattle bbox in the exact shape of the Inference-Quickstart sample.
    bbox = BBox(west=-122.35, south=47.60, east=-122.32, north=47.63)
    scene = await fetch_s2_composite(
        bbox=bbox,
        datetime_range="2024-06-01/2024-09-30",
        max_size_px=64,
        max_cloud_cover=20.0,
    )

    # Shape is (H, W, 12) in OlmoEarth's band order.
    assert scene.image.ndim == 3
    assert scene.image.shape[-1] == 12
    assert scene.image.dtype == np.float32

    # At least one band should carry non-zero reflectance over Seattle.
    assert float(scene.image.max()) > 0.0

    # Timestamp round-trip is sane.
    dmy = timestamp_from_iso(scene.datetime_str)
    assert len(dmy) == 3
    assert 1 <= dmy[0] <= 31
    assert 0 <= dmy[1] <= 11
    assert dmy[2] >= 2015  # S2 archive starts in 2015.
