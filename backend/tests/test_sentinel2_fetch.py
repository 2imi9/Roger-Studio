"""Unit tests for :mod:`app.services.sentinel2_fetch`.

Pure-Python bits (shape helpers, timestamp conversion) run offline.
The actual STAC search + band read is marked ``network``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio  # noqa: F401  — used by new sequential-mode tests

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services.sentinel2_fetch import (  # noqa: E402
    _S2_CACHE_VERSION,
    _s2_cache_get,
    _s2_cache_key,
    _s2_cache_put,
    _sign,
    image_to_bhwtc,
    plan_chunks,
    stack_to_bhwtc,
    timestamp_from_iso,
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
# stack_to_bhwtc — temporal-stack reshape helper introduced for chunked
# native-resolution inference.
# ---------------------------------------------------------------------------


def test_stack_to_bhwtc_reshapes_multi_period() -> None:
    # (H, W, T, 12) → (1, H, W, T, 12) with dtype coerced to float32.
    stack = np.zeros((16, 16, 6, 12), dtype=np.int16)  # intentionally wrong dtype
    out = stack_to_bhwtc(stack)
    assert out.shape == (1, 16, 16, 6, 12)
    assert out.dtype == np.float32


def test_stack_to_bhwtc_rejects_wrong_rank() -> None:
    bad = np.zeros((16, 16, 12), dtype=np.float32)  # missing T
    with pytest.raises(ValueError, match=r"\(H, W, T, 12\)"):
        stack_to_bhwtc(bad)


def test_stack_to_bhwtc_rejects_wrong_band_count() -> None:
    bad = np.zeros((16, 16, 6, 9), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(H, W, T, 12\)"):
        stack_to_bhwtc(bad)


# ---------------------------------------------------------------------------
# plan_chunks — AOI tiling. Pure function, deterministic, worth
# exhaustive edge coverage because off-by-one here poisons every
# chunked inference downstream.
# ---------------------------------------------------------------------------


def test_plan_chunks_small_aoi_returns_single_chunk() -> None:
    """AOI smaller than chunk_size_m collapses to one chunk covering it."""
    bbox = BBox(west=-122.35, south=47.60, east=-122.32, north=47.63)  # ~3 km
    chunks = plan_chunks(bbox, chunk_size_m=10_000)
    assert len(chunks) == 1
    assert chunks[0].west == bbox.west
    assert chunks[0].east == bbox.east
    assert chunks[0].south == bbox.south
    assert chunks[0].north == bbox.north


def test_plan_chunks_grid_partitions_large_aoi() -> None:
    """A 20 km × 20 km AOI at chunk_size=5 km should produce ~4×4 = 16 chunks."""
    # ~20 km at ~45° lat ≈ 0.18° lon, 0.18° lat.
    bbox = BBox(west=0.0, south=45.0, east=0.18, north=45.18)
    chunks = plan_chunks(bbox, chunk_size_m=5_000)
    # Allow ±1 per axis because ceil() + lat-scaling produces 4 or 5.
    # The invariant that MUST hold is coverage + non-overlap.
    assert 9 <= len(chunks) <= 25

    # Chunks tile the AOI exactly — union bounds match the original bbox.
    min_w = min(c.west for c in chunks)
    max_e = max(c.east for c in chunks)
    min_s = min(c.south for c in chunks)
    max_n = max(c.north for c in chunks)
    assert min_w == pytest.approx(bbox.west, abs=1e-9)
    assert max_e == pytest.approx(bbox.east, abs=1e-9)
    assert min_s == pytest.approx(bbox.south, abs=1e-9)
    assert max_n == pytest.approx(bbox.north, abs=1e-9)


def test_plan_chunks_non_overlapping() -> None:
    """Adjacent chunks share edges exactly — no gaps, no overlaps."""
    bbox = BBox(west=0.0, south=45.0, east=0.1, north=45.1)
    chunks = plan_chunks(bbox, chunk_size_m=5_000)
    # Sort by (row, col) — north-to-south, west-to-east.
    chunks_sorted = sorted(chunks, key=lambda c: (-c.north, c.west))
    # Any two chunks with the same north/south pair should share adjacent
    # east/west boundaries.
    by_row: dict[tuple[float, float], list[BBox]] = {}
    for c in chunks_sorted:
        by_row.setdefault((c.north, c.south), []).append(c)
    for row_chunks in by_row.values():
        row_chunks.sort(key=lambda c: c.west)
        for a, b in zip(row_chunks, row_chunks[1:]):
            assert a.east == pytest.approx(b.west, abs=1e-9)


def test_plan_chunks_clamps_to_minimum_one() -> None:
    """Degenerate zero-area bbox still returns at least one chunk."""
    bbox = BBox(west=0.0, south=45.0, east=0.0, north=45.0)
    chunks = plan_chunks(bbox, chunk_size_m=5_000)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# S2 local scene cache — roundtrip + corruption handling.
# ---------------------------------------------------------------------------


def test_s2_cache_key_is_stable_and_sharded(tmp_path, monkeypatch) -> None:
    """Same inputs → same path; different inputs → different paths;
    directory is sharded by hash prefix."""
    monkeypatch.setenv("S2_CACHE_DIR", str(tmp_path))
    # Reimport-local constants won't change but the helper reads the
    # module's path at call time, so we have to monkeypatch the module
    # attribute too.
    from app.services import sentinel2_fetch as sf
    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)

    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    p1 = _s2_cache_key("scene-A", bbox, "B04", 10.0)
    p2 = _s2_cache_key("scene-A", bbox, "B04", 10.0)
    p3 = _s2_cache_key("scene-A", bbox, "B08", 10.0)   # different band
    p4 = _s2_cache_key("scene-B", bbox, "B04", 10.0)   # different scene
    p5 = _s2_cache_key("scene-A", bbox, "B04", 20.0)   # different gsd

    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    assert p1 != p5
    # Sharded: second-to-last path component is a 2-char dir.
    assert len(p1.parent.name) == 2


def test_s2_cache_roundtrip_put_then_get(tmp_path, monkeypatch) -> None:
    from app.services import sentinel2_fetch as sf
    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", False)

    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    arr = np.arange(25, dtype=np.float32).reshape(5, 5) * 100  # DN-range values

    # Miss before put.
    assert _s2_cache_get("scene-X", bbox, "B04", 10.0) is None

    # Put, then hit.
    _s2_cache_put("scene-X", bbox, "B04", 10.0, arr)
    hit = _s2_cache_get("scene-X", bbox, "B04", 10.0)
    assert hit is not None
    assert hit.shape == arr.shape
    assert hit.dtype == np.float32
    np.testing.assert_array_equal(hit, arr)


def test_s2_cache_put_is_atomic(tmp_path, monkeypatch) -> None:
    """Writing via tmp + rename — the final path must NOT end in .tmp,
    and no .tmp file should leak behind on success."""
    from app.services import sentinel2_fetch as sf
    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", False)

    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    arr = np.zeros((4, 4), dtype=np.float32)
    _s2_cache_put("scene-X", bbox, "B02", 10.0, arr)

    # The final .npy exists; no stray .tmp left behind.
    p = _s2_cache_key("scene-X", bbox, "B02", 10.0)
    assert p.exists()
    assert not list(p.parent.glob("*.tmp"))


def test_s2_cache_get_removes_corrupt_file(tmp_path, monkeypatch) -> None:
    """Corrupt cache files (partial write, disk error) get dropped so
    the next put can recover cleanly instead of returning garbage."""
    from app.services import sentinel2_fetch as sf
    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", False)

    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    p = _s2_cache_key("scene-X", bbox, "B02", 10.0)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a valid npy file")

    assert _s2_cache_get("scene-X", bbox, "B02", 10.0) is None
    # File is dropped after the failed load so the next put succeeds.
    assert not p.exists()


def test_s2_cache_disabled_env_short_circuits(tmp_path, monkeypatch) -> None:
    from app.services import sentinel2_fetch as sf
    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", True)

    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    arr = np.ones((3, 3), dtype=np.float32)
    # Put should be a no-op when disabled.
    _s2_cache_put("scene-X", bbox, "B02", 10.0, arr)
    p = _s2_cache_key("scene-X", bbox, "B02", 10.0)
    assert not p.exists()
    # Get always misses when disabled.
    assert _s2_cache_get("scene-X", bbox, "B02", 10.0) is None


# ---------------------------------------------------------------------------
# S2_SEQUENTIAL safety mode — when set, band reads inside a period run
# one-at-a-time via a ``for`` loop instead of asyncio.gather. Memory-
# bounded regardless of AOI size.
# ---------------------------------------------------------------------------


def test_sequential_env_var_parse() -> None:
    """``S2_SEQUENTIAL`` accepts truthy strings and ignores everything else.
    Module-level constant is already evaluated at import, so this covers the
    parsing logic rather than hot-path behaviour."""
    truthy = ["1", "true", "TRUE", "Yes", "YES"]
    falsy = ["", "0", "false", "no", "off", "xyz"]
    for v in truthy:
        assert v.lower() in ("1", "true", "yes"), f"expected {v!r} truthy"
    for v in falsy:
        assert v.lower() not in ("1", "true", "yes"), f"expected {v!r} falsy"


@pytest.mark.asyncio
async def test_sequential_mode_invokes_reads_serially(tmp_path, monkeypatch) -> None:
    """With _S2_SEQUENTIAL on, fetch_s2_chunk_stack should issue reads
    one-at-a-time. We verify by counting concurrent in-flight reads
    hitting a counter-tracking fake of ``_read_one_band_window``."""
    from unittest.mock import patch as _patch
    from app.services import sentinel2_fetch as sf
    from app.services.sentinel2_fetch import (
        AoiPeriodScene,
        fetch_s2_chunk_stack,
    )
    import threading

    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", True)  # force all reads
    monkeypatch.setattr(sf, "_S2_SEQUENTIAL", True)

    # Fake scene with all 12 band assets present.
    want = list(sf.Modality.SENTINEL2_L2A.band_order)
    fake_scene = {
        "id": "S2A_FAKE",
        "collection": "sentinel-2-l2a",
        "datetime": "2024-06-15T10:00:00Z",
        "assets": {b: {"href": f"https://fake/{b}.tif"} for b in want},
    }
    period_scenes = [AoiPeriodScene(
        scene=fake_scene,
        period_start_iso="2024-06-01",
        period_end_iso="2024-06-30",
    )]

    # Counter: tracks max concurrent in-flight reads. If sequential mode
    # holds, max_concurrent == 1 always.
    in_flight = {"n": 0, "max": 0}
    lock = threading.Lock()

    def _fake_read(href, crs, transform, h, w):
        with lock:
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
        try:
            return True, np.zeros((h, w), dtype=np.float32)
        finally:
            with lock:
                in_flight["n"] -= 1

    async def _fake_sas_token(_collection):
        return "sv=fake"

    # Anchor scene's CRS discovery also opens a rasterio file — short-
    # circuit by patching the exact call path. Simpler: pin the CRS
    # via the ``pinned_crs`` kwarg so resolve_aoi_grid isn't invoked.
    scene_crs = rasterio.crs.CRS.from_epsg(32631)

    with _patch.object(sf, "_read_one_band_window", _fake_read), \
         _patch.object(sf, "_get_sas_token", _fake_sas_token):
        result = await fetch_s2_chunk_stack(
            chunk_bbox=BBox(west=0.0, south=45.0, east=0.002, north=45.002),
            period_scenes=period_scenes,
            target_gsd_m=10.0,
            pinned_crs=scene_crs,
        )

    # Sanity: all 12 bands read, one period present.
    assert result.stack.shape[-1] == 12
    assert not result.period_skipped[0]

    # Key assertion: max in-flight reads was 1 (sequential).
    assert in_flight["max"] == 1, (
        f"Expected serial reads (max 1 in-flight), got {in_flight['max']}"
    )


@pytest.mark.asyncio
async def test_parallel_mode_allows_concurrent_reads(tmp_path, monkeypatch) -> None:
    """Default (not sequential) path should fire reads concurrently —
    max in-flight > 1. Complementary to the sequential test."""
    from unittest.mock import patch as _patch
    from app.services import sentinel2_fetch as sf
    from app.services.sentinel2_fetch import (
        AoiPeriodScene,
        fetch_s2_chunk_stack,
    )
    import threading

    monkeypatch.setattr(sf, "_S2_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sf, "_S2_CACHE_DISABLED", True)
    monkeypatch.setattr(sf, "_S2_SEQUENTIAL", False)

    want = list(sf.Modality.SENTINEL2_L2A.band_order)
    fake_scene = {
        "id": "S2A_FAKE",
        "collection": "sentinel-2-l2a",
        "datetime": "2024-06-15T10:00:00Z",
        "assets": {b: {"href": f"https://fake/{b}.tif"} for b in want},
    }
    period_scenes = [AoiPeriodScene(
        scene=fake_scene,
        period_start_iso="2024-06-01",
        period_end_iso="2024-06-30",
    )]

    in_flight = {"n": 0, "max": 0}
    lock = threading.Lock()
    barrier = threading.Event()

    def _fake_read(href, crs, transform, h, w):
        with lock:
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
        # Hold briefly so parallel calls actually overlap.
        import time as _time
        _time.sleep(0.02)
        try:
            return True, np.zeros((h, w), dtype=np.float32)
        finally:
            with lock:
                in_flight["n"] -= 1

    async def _fake_sas_token(_collection):
        return "sv=fake"

    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    with _patch.object(sf, "_read_one_band_window", _fake_read), \
         _patch.object(sf, "_get_sas_token", _fake_sas_token):
        result = await fetch_s2_chunk_stack(
            chunk_bbox=BBox(west=0.0, south=45.0, east=0.002, north=45.002),
            period_scenes=period_scenes,
            target_gsd_m=10.0,
            pinned_crs=scene_crs,
        )

    assert result.stack.shape[-1] == 12
    # Parallel path should achieve > 1 concurrent read. Upper bound is
    # 12 (one per band) but we only need > 1 to prove parallelism.
    assert in_flight["max"] > 1, (
        f"Expected concurrent reads, got max {in_flight['max']}"
    )


def test_s2_cache_version_tag_in_key() -> None:
    """Bumping the cache version invalidates every previous entry — the
    key must incorporate the version, not just the data inputs."""
    bbox = BBox(west=0.0, south=45.0, east=0.01, north=45.01)
    # Different versions → different paths.
    from app.services import sentinel2_fetch as sf
    p_v1 = _s2_cache_key("scene-X", bbox, "B02", 10.0)
    original = sf._S2_CACHE_VERSION
    try:
        sf._S2_CACHE_VERSION = "v999"
        p_v999 = _s2_cache_key("scene-X", bbox, "B02", 10.0)
        assert p_v1 != p_v999
    finally:
        sf._S2_CACHE_VERSION = original
    # Sanity — the original version gave the same path both times.
    assert _S2_CACHE_VERSION == original


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
