"""Unit tests for the embedding export pipeline.

Exercises three layers:

  1. ``build_embedding_cog_bytes`` — int8 quantization round-trip via the
     AlphaEarth-compatible scheme + COG/GTiff serialization. Verified by
     re-reading the written bytes with rasterio and comparing against the
     pre-quantize input.

  2. ``_run_chunked_embedding_export`` — end-to-end orchestration with
     every network + model boundary mocked. Catches bugs in the chunk
     planner + stitching + int8 conversion without ever touching
     Planetary Computer or HuggingFace.

  3. ``/api/olmoearth/export-embedding`` — FastAPI TestClient request
     with the orchestrator itself mocked. Catches schema / validation /
     response-header regressions.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import rasterio
import torch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services import olmoearth_inference  # noqa: E402
from app.services.olmoearth_inference import (  # noqa: E402
    build_embedding_cog_bytes,
)
from app.services.sentinel2_fetch import (  # noqa: E402
    AoiPeriodScene,
    SentinelTemporalStack,
)


# ---------------------------------------------------------------------------
# COG writer — int8 quantize → write → read → dequantize round-trip.
# Catches: wrong byte order, dtype confusion, nodata placement, band
# count mismatch, provenance tags missing.
# ---------------------------------------------------------------------------


def _make_fake_export_result(
    h: int = 8, w: int = 12, d: int = 16, embed_fill: float = 0.5
) -> dict:
    """Synthesize a stitched embedding result matching what
    _run_chunked_embedding_export returns. float32 embedding first gets
    quantized via the same path the real orchestrator uses, then wrapped
    with the metadata the COG writer expects."""
    from olmoearth_pretrain.evals.embedding_transforms import (
        quantize_embeddings,
    )

    emb = np.full((h, w, d), embed_fill, dtype=np.float32)
    q = quantize_embeddings(torch.from_numpy(emb)).numpy().astype(np.int8)

    # Fake UTM transform at 40 m/pixel (10m GSD × 4 patch).
    transform = rasterio.Affine(40.0, 0, 500_000.0, 0, -40.0, 4_500_000.0)
    return {
        "embedding_int8": q,
        "embedding_dim": d,
        "transform": transform,
        "crs": rasterio.crs.CRS.from_epsg(32618),
        "patch_size": 4,
        "target_gsd_m": 10.0,
        "chunks_processed": 3,
        "chunks_failed": 0,
        "chunks_total": 3,
        "n_periods": 6,
        "period_days": 30,
        "modality": "sentinel2_l2a",
        "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
        "scene_ids": ["S2A_...", "S2B_...", "S2A_...", None, None, None],
        "scene_datetimes": ["2024-06-28T10:00:00Z"] * 3 + [None] * 3,
    }


def test_cog_bytes_are_readable_by_rasterio() -> None:
    result = _make_fake_export_result(h=8, w=12, d=16)
    cog_bytes, filename = build_embedding_cog_bytes(result)

    assert cog_bytes[:2] in (b"II", b"MM")  # TIFF magic (little/big endian)
    assert filename.endswith(".tif")
    assert "OlmoEarth-v1-Tiny" in filename or "Tiny" in filename

    with rasterio.open(io.BytesIO(cog_bytes)) as ds:
        assert ds.count == 16          # one band per embedding dim
        assert ds.height == 8
        assert ds.width == 12
        assert ds.dtypes[0] == "int8"
        assert ds.nodata == -128
        # CRS + transform preserved.
        assert ds.crs.to_epsg() == 32618
        assert ds.transform.a == pytest.approx(40.0)
        assert ds.transform.e == pytest.approx(-40.0)


def test_cog_preserves_int8_values_exactly() -> None:
    """The int8 bytes we wrote should match what we read — no silent
    dtype coercion (e.g. int8 → uint8 wrap) during serialization."""
    result = _make_fake_export_result(h=4, w=4, d=3, embed_fill=0.5)
    cog_bytes, _ = build_embedding_cog_bytes(result)

    with rasterio.open(io.BytesIO(cog_bytes)) as ds:
        read_back = ds.read()  # (D, H, W)

    # rasterio gives us (D, H, W); our source was (H, W, D) — transpose
    # to match what we stored.
    source = np.transpose(result["embedding_int8"], (2, 0, 1))
    np.testing.assert_array_equal(read_back, source)


def test_cog_dequantize_roundtrip_recovers_float() -> None:
    """Dequantizing the written int8 COG returns approximately the
    pre-quantize float values. Same contract as Ai2 Studio's exports."""
    from olmoearth_pretrain.evals.embedding_transforms import (
        dequantize_embeddings,
    )

    result = _make_fake_export_result(h=4, w=4, d=8, embed_fill=0.3)
    cog_bytes, _ = build_embedding_cog_bytes(result)

    with rasterio.open(io.BytesIO(cog_bytes)) as ds:
        q = ds.read()                     # (D, H, W) int8
    q_hwd = np.transpose(q, (1, 2, 0))    # (H, W, D)
    deq = dequantize_embeddings(torch.from_numpy(q_hwd)).numpy()

    # Round-trip error from the sqrt/int8 quantization is ~<0.02 for
    # values in the typical encoder range.
    expected = np.full((4, 4, 8), 0.3, dtype=np.float32)
    assert np.abs(deq - expected).max() < 0.02


def test_cog_embeds_provenance_tags() -> None:
    """GIS tools + downstream scripts depend on reading the model repo /
    patch_size / n_periods back from the file. They must survive the
    write → read cycle."""
    result = _make_fake_export_result(h=4, w=4, d=4)
    cog_bytes, _ = build_embedding_cog_bytes(result)

    with rasterio.open(io.BytesIO(cog_bytes)) as ds:
        tags = ds.tags()

    assert tags["model_repo_id"] == "allenai/OlmoEarth-v1-Tiny"
    assert tags["embedding_dim"] == "4"
    assert tags["patch_size"] == "4"
    assert tags["n_periods"] == "6"
    assert tags["modality"] == "sentinel2_l2a"
    assert tags["nodata_value"] == "-128"
    # Quantization provenance — lets future consumers pick the right
    # dequantize scheme if we ever support multiple.
    assert "quantize_embeddings" in tags["quantization"]


def test_cog_has_per_band_descriptions() -> None:
    """Each band advertises ``dim_NNN`` so QGIS shows meaningful labels
    instead of "Band 1", "Band 2", ..."""
    result = _make_fake_export_result(h=4, w=4, d=5)
    cog_bytes, _ = build_embedding_cog_bytes(result)

    with rasterio.open(io.BytesIO(cog_bytes)) as ds:
        descs = ds.descriptions

    assert len(descs) == 5
    for i, desc in enumerate(descs):
        assert desc == f"dim_{i:03d}"


# ---------------------------------------------------------------------------
# _run_chunked_embedding_export — full orchestrator with EVERY network +
# model boundary mocked. Verifies chunking / stitching / int8 conversion
# without touching PC / HF.
# ---------------------------------------------------------------------------


def _fake_inference_result(
    h_patch: int, w_patch: int, dim: int, fill: float = 0.2
) -> object:
    """Minimal stand-in for olmoearth_model.InferenceResult."""
    class _R:
        embedding = np.full((h_patch, w_patch, dim), fill, dtype=np.float32)
        scalar = np.zeros((h_patch, w_patch), dtype=np.float32)
        patch_size = 4
        embedding_dim = dim
    return _R()


def _fake_stack_result(
    chunk_h_px: int, chunk_w_px: int, scene_crs, transform, n_periods: int = 2
) -> SentinelTemporalStack:
    """SentinelTemporalStack with plausible DN reflectance values."""
    stack = np.full(
        (chunk_h_px, chunk_w_px, n_periods, 12), 3000.0, dtype=np.float32,
    )
    return SentinelTemporalStack(
        stack=stack,
        transform=transform,
        crs=scene_crs,
        timestamps=[(15, 5, 2024)] * n_periods,
        scene_ids=[f"S2A_fake_{i}" for i in range(n_periods)],
        scene_datetimes=["2024-06-15T10:00:00Z"] * n_periods,
        cloud_covers=[2.0] * n_periods,
        period_skipped=[False] * n_periods,
        bbox_wgs84=(0.0, 45.0, 0.01, 45.01),
    )


@pytest.mark.asyncio
async def test_chunked_embedding_export_stitches_single_chunk(monkeypatch) -> None:
    """Tiny AOI → 1 chunk → output embedding has the expected shape and
    is int8 quantized."""
    bbox = BBox(west=0.0, south=45.0, east=0.005, north=45.005)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    # 32x32 chunk at 10 m/pixel.
    chunk_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)

    # Mock the 4 network/heavy boundaries.
    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
        for _ in range(2)
    ]
    stack_result = _fake_stack_result(32, 32, scene_crs, chunk_transform, n_periods=2)

    async def _fake_period_scenes(**_kw):
        return period_scenes

    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 32, 32

    async def _fake_chunk_stack(**_kw):
        return stack_result

    # Mock olmoearth_model.run_s2_inference to avoid loading a real encoder.
    fake_inf = _fake_inference_result(h_patch=8, w_patch=8, dim=192, fill=0.25)

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes), \
         patch.object(olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid), \
         patch.object(olmoearth_inference, "fetch_s2_chunk_stack", _fake_chunk_stack), \
         patch.object(
             olmoearth_inference.olmoearth_model,
             "run_s2_inference",
             return_value=fake_inf,
         ):
        result = await olmoearth_inference._run_chunked_embedding_export(
            bbox=bbox,
            model=_FakeModel(),
            device="cpu",
            model_repo_id="allenai/OlmoEarth-v1-Tiny",
            date_range="2024-06-01/2024-07-01",
            n_periods=2,
            period_days=30,
            chunk_size_m=5_000,
            target_gsd_m=10.0,
            patch_size=4,
        )

    # Shape: 32 px / 4 patch = 8 patches per side.
    assert result["embedding_int8"].shape == (8, 8, 192)
    assert result["embedding_int8"].dtype == np.int8
    assert result["embedding_dim"] == 192
    assert result["patch_size"] == 4
    assert result["target_gsd_m"] == 10.0
    assert result["chunks_processed"] == 1
    assert result["chunks_failed"] == 0
    assert result["chunks_total"] == 1
    assert result["model_repo_id"] == "allenai/OlmoEarth-v1-Tiny"

    # Quantized values should be non-zero (fill=0.25 maps to a positive
    # int8 via the sqrt scheme) and well inside the valid ±127 range.
    q = result["embedding_int8"]
    assert q.min() > -128    # no accidental nodata fill
    assert q.max() < 127
    assert q.mean() > 0      # fill was positive


@pytest.mark.asyncio
async def test_chunked_embedding_export_trips_breaker_and_stops_early(monkeypatch) -> None:
    """When chunks fail back-to-back, the breaker trips at
    ``OE_CIRCUIT_BREAKER_FAILS`` consecutive failures and the orchestrator
    raises ``CircuitBreakerTrippedError`` instead of grinding through all
    remaining chunks. This is the feature that turns a 35-min grind on a
    dead network into a ~60 s bail."""
    from app.services.sentinel2_fetch import SentinelFetchError
    from app.services.system_health import CircuitBreakerTrippedError

    # Pin threshold to 2 so the test is fast — 2 consecutive fails should
    # trip immediately, not after waiting 3+ × 300s per-chunk timeouts.
    monkeypatch.setenv("OE_CIRCUIT_BREAKER_FAILS", "2")

    # Plan a multi-chunk AOI. With chunk_size_m=1000 and a ~1.6×2.2 km
    # bbox at 45° lat we get ~6 chunks — enough headroom that a
    # threshold of 2 trips before exhausting the chunk list.
    bbox = BBox(west=0.0, south=45.0, east=0.02, north=45.02)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)

    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
    ]

    async def _fake_period_scenes(**_kw):
        return period_scenes

    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 32, 32

    attempted_chunks = []

    async def _fake_chunk_stack(chunk_bbox, **_kw):
        attempted_chunks.append((chunk_bbox.west, chunk_bbox.south))
        raise SentinelFetchError("simulated network failure")

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(
        olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes,
    ), patch.object(
        olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid,
    ), patch.object(
        olmoearth_inference, "fetch_s2_chunk_stack", _fake_chunk_stack,
    ):
        with pytest.raises(CircuitBreakerTrippedError) as exc_info:
            await olmoearth_inference._run_chunked_embedding_export(
                bbox=bbox,
                model=_FakeModel(),
                device="cpu",
                model_repo_id="allenai/OlmoEarth-v1-Tiny",
                date_range="2024-06-01/2024-07-01",
                n_periods=1,
                period_days=30,
                chunk_size_m=1_000,    # smaller chunks → more of them
                target_gsd_m=10.0,
                patch_size=4,
            )

    # The breaker's stats should reflect what actually happened.
    err = exc_info.value
    assert err.threshold == 2
    assert err.failed >= 2  # at least the 2 that tripped
    # Total chunks should be > 2 — we planned more than the threshold so
    # this test actually exercises the "stop early" behaviour.
    assert err.total > err.threshold


@pytest.mark.asyncio
async def test_chunked_embedding_export_raises_on_all_chunks_failed(monkeypatch) -> None:
    """Every chunk skips → the orchestrator raises, not silently returns
    an empty raster."""
    from app.services.sentinel2_fetch import SentinelFetchError
    bbox = BBox(west=0.0, south=45.0, east=0.005, north=45.005)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
    ]

    async def _fake_period_scenes(**_kw):
        return period_scenes

    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 32, 32

    async def _fake_chunk_stack(**_kw):
        raise SentinelFetchError("simulated fetch failure")

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes), \
         patch.object(olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid), \
         patch.object(olmoearth_inference, "fetch_s2_chunk_stack", _fake_chunk_stack):
        with pytest.raises(SentinelFetchError, match="all .* chunks failed"):
            await olmoearth_inference._run_chunked_embedding_export(
                bbox=bbox,
                model=_FakeModel(),
                device="cpu",
                model_repo_id="allenai/OlmoEarth-v1-Tiny",
                date_range="2024-06-01/2024-07-01",
                n_periods=1,
                period_days=30,
                chunk_size_m=5_000,
                target_gsd_m=10.0,
                patch_size=4,
            )


@pytest.mark.asyncio
async def test_chunked_embedding_export_raises_on_aoi_too_small_for_patch(monkeypatch) -> None:
    """If the AOI resolves to a grid smaller than patch_size, the function
    should refuse to produce a zero-pixel raster."""
    bbox = BBox(west=0.0, south=45.0, east=0.0001, north=45.0001)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
    ]

    async def _fake_period_scenes(**_kw):
        return period_scenes

    # Grid < patch_size → should raise.
    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 2, 2

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes), \
         patch.object(olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid):
        with pytest.raises(ValueError, match="too small for patch_size"):
            await olmoearth_inference._run_chunked_embedding_export(
                bbox=bbox,
                model=_FakeModel(),
                device="cpu",
                model_repo_id="allenai/OlmoEarth-v1-Tiny",
                date_range="2024-06-01/2024-07-01",
                n_periods=1,
                period_days=30,
                chunk_size_m=5_000,
                target_gsd_m=10.0,
                patch_size=4,
            )


# ---------------------------------------------------------------------------
# Router endpoint — /api/olmoearth/export-embedding.
# TestClient covers the schema + validation + headers + download flow
# without hitting the orchestrator's heavy mocks.
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_export_embedding_rejects_fine_tuned_repo(test_client) -> None:
    """FT heads produce task outputs, not embeddings — endpoint must 400
    rather than run and return nonsense."""
    r = test_client.post(
        "/api/olmoearth/export-embedding",
        json={
            "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
            "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        },
    )
    assert r.status_code == 400
    assert "base encoder" in r.text.lower()


def test_export_embedding_validates_bbox_shape(test_client) -> None:
    """Missing required bbox field → 422 from Pydantic, not a 500."""
    r = test_client.post(
        "/api/olmoearth/export-embedding",
        json={"model_repo_id": "allenai/OlmoEarth-v1-Tiny"},
    )
    assert r.status_code == 422


def test_export_embedding_validates_gsd_bounds(test_client) -> None:
    """target_gsd_m is constrained to 10..80 m by the Field validator."""
    r = test_client.post(
        "/api/olmoearth/export-embedding",
        json={
            "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
            "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            "target_gsd_m": 5.0,   # below 10 m floor
        },
    )
    assert r.status_code == 422


def test_export_embedding_returns_tiff_and_headers(test_client) -> None:
    """Happy path with orchestrator mocked — endpoint streams TIFF bytes
    + the X-Embedding-* metadata headers the frontend surfaces."""
    fake_export = _make_fake_export_result(h=4, w=4, d=4)

    async def _fake_orch(**_kw):
        return fake_export

    # Also mock the model loader so the test doesn't download HF weights.
    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    fake_device = "cpu"

    with patch.object(
        olmoearth_inference, "_run_chunked_embedding_export",
        side_effect=_fake_orch,
    ), patch(
        "app.routers.olmoearth.olmoearth_model.load_encoder",
        return_value=(_FakeModel(), fake_device),
    ):
        r = test_client.post(
            "/api/olmoearth/export-embedding",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
                "n_periods": 2,
                "period_days": 30,
                "target_gsd_m": 10.0,
                "patch_size": 4,
                "chunk_size_m": 5000,
            },
        )
    assert r.status_code == 200, r.text
    # TIFF content.
    assert r.content[:2] in (b"II", b"MM")
    # Headers the frontend reads to surface "192 dims · 40 m/pixel".
    assert r.headers.get("x-embedding-dim") == "4"
    assert r.headers.get("x-embedding-patch-gsd-m") == "40.0"
    assert r.headers.get("x-chunks-processed") == "3"
    assert r.headers.get("x-chunks-failed") == "0"
    # Attachment disposition so browsers auto-download.
    assert "attachment" in r.headers.get("content-disposition", "").lower()
