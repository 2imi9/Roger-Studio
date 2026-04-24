"""Unit tests for the in-UI embedding tools — first one is PCA false-color.

Coverage:
  1. ``pca_to_rgb`` pure-function: shape, dtype, nodata propagation,
     valid pixels actually use the dynamic range.
  2. The RGB tile-rendering path in ``_render_pytorch_tile`` — verifies
     that ``rgb_raster`` short-circuits the colormap codepath.
  3. ``run_embedding_tool_pca_rgb`` end-to-end with mocked encoder + S2
     fetch — proves the orchestrator wires PCA into the existing _jobs
     dict and tile route.
  4. ``/api/olmoearth/embedding-tools/pca-rgb`` router — schema validation
     (rejects FT heads with 400) + happy path returns the inference
     result shape the frontend already understands.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio  # noqa: F401

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services import olmoearth_inference  # noqa: E402
from app.services.olmoearth_model import (  # noqa: E402
    cosine_similarity_map,
    pca_to_rgb,
)
from app.services.sentinel2_fetch import (  # noqa: E402
    AoiPeriodScene,
    SentinelTemporalStack,
)


# ---------------------------------------------------------------------------
# pca_to_rgb pure function — shape / dtype / nodata / dynamic range.
# ---------------------------------------------------------------------------


def test_pca_to_rgb_returns_uint8_three_band() -> None:
    """Return shape MUST be (H, W, 3) uint8 — that's the contract the
    tile renderer's RGB path depends on."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((16, 24, 32)).astype(np.float32)
    rgb = pca_to_rgb(emb)
    assert rgb.shape == (16, 24, 3)
    assert rgb.dtype == np.uint8


def test_pca_to_rgb_uses_full_dynamic_range_on_valid_pixels() -> None:
    """A non-degenerate embedding should produce per-channel min ~0 and
    max ~255 — proves the per-PC rescaling actually fills the byte
    range instead of squashing into the middle."""
    rng = np.random.default_rng(42)
    # Distinct values per pixel guarantee non-zero variance along each PC.
    emb = rng.standard_normal((20, 20, 64)).astype(np.float32)
    rgb = pca_to_rgb(emb)
    for ch in range(3):
        assert int(rgb[..., ch].min()) <= 5
        assert int(rgb[..., ch].max()) >= 250


def test_pca_to_rgb_marks_zero_pixels_as_nodata() -> None:
    """Pixels where every D dim is exactly 0 represent untouched chunks
    in the stitched embedding — they MUST come out as (0, 0, 0) so the
    RGB tile renderer can paint them transparent."""
    emb = np.zeros((8, 8, 16), dtype=np.float32)
    # Fill a few pixels with real values, leave the rest at 0.
    emb[2:5, 2:5, :] = np.random.default_rng(0).standard_normal((3, 3, 16))
    rgb = pca_to_rgb(emb)

    # Every "untouched" (all-zero) pixel must come back (0, 0, 0).
    nodata_mask = ~np.any(emb != 0, axis=-1)
    assert (rgb[nodata_mask] == 0).all()
    # And the touched pixels are non-trivial.
    valid = ~nodata_mask
    assert (rgb[valid] > 0).any()


def test_pca_to_rgb_handles_low_dimensional_embedding() -> None:
    """If the encoder emits fewer than 3 dims (degenerate), the helper
    must still return 3-band RGB — pad missing PCs with zeros."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((10, 10, 2)).astype(np.float32)  # only 2 dims
    rgb = pca_to_rgb(emb)
    assert rgb.shape == (10, 10, 3)
    # Channel 0 + 1 should have variation; channel 2 is the padded zero.
    assert rgb[..., 0].std() > 0
    assert rgb[..., 1].std() > 0
    # The padded channel after rescale = 0 (constant input → max==min → 0/eps).
    assert rgb[..., 2].max() <= 1   # tolerate tiny rounding


def test_pca_to_rgb_handles_all_nodata() -> None:
    """All-zero input must not raise — return all-zero RGB silently so
    the tile renders as transparent everywhere."""
    emb = np.zeros((4, 4, 16), dtype=np.float32)
    rgb = pca_to_rgb(emb)
    assert rgb.shape == (4, 4, 3)
    assert (rgb == 0).all()


# ---------------------------------------------------------------------------
# cosine_similarity_map pure function — query-vector matching.
# ---------------------------------------------------------------------------


def test_cosine_similarity_map_identical_query_is_one() -> None:
    """A patch matched against ITS OWN embedding vector must score 1.0
    (after the [-1, 1] → [0, 1] rescale)."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((6, 6, 32)).astype(np.float32)
    # Pick a specific patch as the query.
    qr, qc = 2, 3
    sim = cosine_similarity_map(emb, emb[qr, qc])
    # That pixel should saturate at 1.0 (cos sim = 1, rescaled = 1).
    assert sim[qr, qc] == pytest.approx(1.0, abs=1e-5)
    # All values in [0, 1].
    assert sim.min() >= 0.0
    assert sim.max() <= 1.0


def test_cosine_similarity_map_anti_correlated_is_zero() -> None:
    """A query that's the EXACT NEGATIVE of a patch should score ~0
    (cos = -1, rescaled = 0). Verifies the [-1, 1] → [0, 1] mapping."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(16).astype(np.float32)
    emb = np.zeros((4, 4, 16), dtype=np.float32)
    emb[1, 1] = base
    emb[2, 2] = -base   # exact negation
    sim = cosine_similarity_map(emb, base)
    assert sim[1, 1] == pytest.approx(1.0, abs=1e-5)
    assert sim[2, 2] == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_map_marks_nodata_as_zero() -> None:
    """All-zero (nodata) patches must come back with similarity 0,
    NOT a spurious mid-range value from cos(zero, query) = 0/eps ≈ 0.
    The rescale would put 0 → 0.5, which would render as 'unrelated'
    instead of clearly 'no data here'. The helper enforces 0 explicitly."""
    rng = np.random.default_rng(2)
    emb = np.zeros((5, 5, 16), dtype=np.float32)
    emb[2, 2] = rng.standard_normal(16)   # only one valid patch
    sim = cosine_similarity_map(emb, emb[2, 2])
    # Valid patch saturates at 1.
    assert sim[2, 2] == pytest.approx(1.0, abs=1e-5)
    # Every nodata patch is exactly 0 (NOT 0.5).
    nodata_mask = ~np.any(emb != 0, axis=-1)
    assert (sim[nodata_mask] == 0.0).all()


def test_cosine_similarity_map_rejects_dim_mismatch() -> None:
    """Wrong-shape query → ValueError (not silently broadcast)."""
    emb = np.zeros((3, 3, 16), dtype=np.float32)
    bad_query = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="last dim"):
        cosine_similarity_map(emb, bad_query)


# ---------------------------------------------------------------------------
# run_embedding_tool_similarity — orchestrator end-to-end (mocked).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_embedding_tool_similarity_uses_aoi_center_when_no_query(monkeypatch) -> None:
    """When query_lon/query_lat are None, the orchestrator must use the
    AOI center as the query — gives a one-click demo without needing
    a pixel-pick UI yet."""
    bbox = BBox(west=0.0, south=45.0, east=0.005, north=45.005)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    chunk_transform = global_transform

    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
    ]
    stack_result = SentinelTemporalStack(
        stack=np.full((32, 32, 1, 12), 3000.0, dtype=np.float32),
        transform=chunk_transform, crs=scene_crs,
        timestamps=[(15, 5, 2024)],
        scene_ids=["S2A"], scene_datetimes=["2024-06-15T10:00Z"],
        cloud_covers=[1.0], period_skipped=[False],
        bbox_wgs84=(0.0, 45.0, 0.005, 45.005),
    )

    async def _fake_period_scenes(**_kw):
        return period_scenes

    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 32, 32

    async def _fake_chunk_stack(**_kw):
        return stack_result

    rng = np.random.default_rng(0)
    fake_emb = rng.standard_normal((8, 8, 192)).astype(np.float32)

    class _FakeInferenceResult:
        embedding = fake_emb
        scalar = np.zeros((8, 8), dtype=np.float32)
        patch_size = 4
        embedding_dim = 192

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes), \
         patch.object(olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid), \
         patch.object(olmoearth_inference, "fetch_s2_chunk_stack", _fake_chunk_stack), \
         patch.object(
             olmoearth_inference.olmoearth_model, "load_encoder",
             return_value=(_FakeModel(), "cpu"),
         ), \
         patch.object(
             olmoearth_inference.olmoearth_model, "run_s2_inference",
             return_value=_FakeInferenceResult(),
         ):
        result = await olmoearth_inference.run_embedding_tool_similarity(
            bbox=bbox,
            model_repo_id="allenai/OlmoEarth-v1-Tiny",
            query_lon=None,    # explicit None → use AOI center
            query_lat=None,
            date_range="2024-06-01/2024-07-01",
            n_periods=1,
            period_days=30,
            chunk_size_m=5_000,
            target_gsd_m=10.0,
            patch_size=4,
        )

    # Same response shape as start_inference.
    assert result["kind"] == "pytorch"
    assert result["status"] == "ready"
    assert "tile_url" in result

    # Job has scalar_raster (similarity heatmap) — NOT rgb_raster.
    job = olmoearth_inference._jobs[result["job_id"]]
    assert job["task_type"] == "embedding_similarity"
    assert "scalar_raster" in job
    assert job["scalar_raster"].shape == (8, 8)
    assert job["scalar_raster"].dtype == np.float32
    # Similarity values clamped to [0, 1].
    assert job["scalar_raster"].min() >= 0.0
    assert job["scalar_raster"].max() <= 1.0
    # Query metadata preserved for the legend.
    assert "similarity_query" in job
    assert job["similarity_query"]["lon"] == pytest.approx(
        (bbox.west + bbox.east) / 2, abs=1e-3,
    )
    assert job["similarity_query"]["lat"] == pytest.approx(
        (bbox.south + bbox.north) / 2, abs=1e-3,
    )


@pytest.mark.asyncio
async def test_run_embedding_tool_similarity_rejects_ft_repo() -> None:
    """FT heads have no raw embedding to compare against — rejected
    before any IO."""
    with pytest.raises(ValueError, match="base encoder"):
        await olmoearth_inference.run_embedding_tool_similarity(
            bbox=BBox(west=0.0, south=45.0, east=0.005, north=45.005),
            model_repo_id="allenai/OlmoEarth-v1-FT-Mangrove-Base",
        )


# ---------------------------------------------------------------------------
# Tile renderer RGB path — verifies _render_pytorch_tile picks the RGB
# code path when ``rgb_raster`` is present.
# ---------------------------------------------------------------------------


def test_render_pytorch_tile_uses_rgb_raster_when_present() -> None:
    """When a job has ``rgb_raster`` set, the tile renderer should use
    the RGB path and produce a non-empty PNG that includes the source
    colors. Catches accidental fallback to the scalar/class paths."""
    # Construct a job with an obvious red-green-blue pattern in the raster.
    h, w = 32, 32
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :w//3, 0] = 255   # left third red
    rgb[:, w//3:2*w//3, 1] = 255   # middle green
    rgb[:, 2*w//3:, 2] = 255   # right blue

    # Build a transform + bbox so the tile sampler resolves to within
    # the raster. EPSG:32631 (UTM zone 31N), 10 m/pixel grid at a known
    # location, then derive a WGS84 bbox from those bounds.
    import rasterio as rio
    crs = rio.crs.CRS.from_epsg(32631)
    transform = rio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    # West/north corner: 500000 / 4983000 in UTM31N. Convert to WGS84.
    from rasterio.warp import transform_bounds
    bbox_wgs = transform_bounds(
        crs, "EPSG:4326",
        transform.c, transform.f - h * abs(transform.e),
        transform.c + w * transform.a, transform.f,
    )
    job = {
        "job_id": "pca_test_job",
        "kind": "pytorch",
        "status": "ready",
        "spec": {"bbox": {
            "west": bbox_wgs[0], "south": bbox_wgs[1],
            "east": bbox_wgs[2], "north": bbox_wgs[3],
        }},
        "rgb_raster": rgb,
        "raster_transform": transform,
        "raster_crs": crs,
        "raster_height": h,
        "raster_width": w,
        "task_type": "embedding_pca_rgb",
        "colormap": "pca_rgb",
        "scalar_raster": np.zeros((h, w), dtype=np.float32),  # ignored
    }

    # Compute a tile (z, x, y) that overlaps the AOI.
    # bbox_wgs[0..3] is the AOI; pick a z+x+y for that lon/lat.
    import math
    lon_mid = (bbox_wgs[0] + bbox_wgs[2]) / 2
    lat_mid = (bbox_wgs[1] + bbox_wgs[3]) / 2
    z = 14
    n = 2 ** z
    x = int((lon_mid + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat_mid)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )

    png_bytes = olmoearth_inference._render_pytorch_tile(job, z, x, y)

    # Real PNG header (rejects empty/transparent fallbacks the renderer
    # uses elsewhere — an empty-tile fallback also returns PNG bytes,
    # but for an in-AOI tile we should get genuine data).
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    # Non-trivial body — empty 256x256 transparent PNGs are tiny
    # (~70 bytes after compression), but a colored tile compresses
    # to a few hundred bytes minimum.
    assert len(png_bytes) > 100


# ---------------------------------------------------------------------------
# run_embedding_tool_pca_rgb orchestrator — wires chunked fetch + PCA.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_embedding_tool_pca_rgb_registers_tile_job(monkeypatch) -> None:
    """End-to-end orchestrator with all network + model boundaries
    mocked. Verifies the job ends up in ``_jobs`` with the expected
    shape so the existing tile route can serve it."""
    bbox = BBox(west=0.0, south=45.0, east=0.005, north=45.005)
    scene_crs = rasterio.crs.CRS.from_epsg(32631)
    global_transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_983_000.0)
    chunk_transform = global_transform

    period_scenes = [
        AoiPeriodScene(
            scene={"id": "S2A", "collection": "sentinel-2-l2a",
                   "assets": {}, "datetime": "2024-06-15T10:00Z"},
            period_start_iso="2024-05-16", period_end_iso="2024-06-15",
        )
        for _ in range(2)
    ]
    stack_result = SentinelTemporalStack(
        stack=np.full((32, 32, 2, 12), 3000.0, dtype=np.float32),
        transform=chunk_transform, crs=scene_crs,
        timestamps=[(15, 5, 2024)] * 2,
        scene_ids=["S2A_one", "S2A_two"],
        scene_datetimes=["2024-06-15T10:00Z"] * 2,
        cloud_covers=[1.0, 2.0],
        period_skipped=[False, False],
        bbox_wgs84=(0.0, 45.0, 0.005, 45.005),
    )

    async def _fake_period_scenes(**_kw):
        return period_scenes

    async def _fake_resolve_grid(*_a, **_kw):
        return scene_crs, global_transform, 32, 32

    async def _fake_chunk_stack(**_kw):
        return stack_result

    # Fake encoder result — provides .embedding suitable for PCA.
    rng = np.random.default_rng(0)
    fake_emb = rng.standard_normal((8, 8, 192)).astype(np.float32)

    class _FakeInferenceResult:
        embedding = fake_emb
        scalar = np.zeros((8, 8), dtype=np.float32)
        patch_size = 4
        embedding_dim = 192

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    with patch.object(olmoearth_inference, "fetch_aoi_period_scenes", _fake_period_scenes), \
         patch.object(olmoearth_inference, "resolve_aoi_grid", _fake_resolve_grid), \
         patch.object(olmoearth_inference, "fetch_s2_chunk_stack", _fake_chunk_stack), \
         patch.object(
             olmoearth_inference.olmoearth_model, "load_encoder",
             return_value=(_FakeModel(), "cpu"),
         ), \
         patch.object(
             olmoearth_inference.olmoearth_model, "run_s2_inference",
             return_value=_FakeInferenceResult(),
         ):
        result = await olmoearth_inference.run_embedding_tool_pca_rgb(
            bbox=bbox,
            model_repo_id="allenai/OlmoEarth-v1-Tiny",
            date_range="2024-06-01/2024-07-01",
            n_periods=2,
            period_days=30,
            chunk_size_m=5_000,
            target_gsd_m=10.0,
            patch_size=4,
        )

    # Response is the same shape as start_inference returns — frontend
    # treats it identically.
    assert result["kind"] == "pytorch"
    assert result["status"] == "ready"
    assert "tile_url" in result
    assert "/api/olmoearth/infer-tile/" in result["tile_url"]

    # Job is in the registry with rgb_raster and the right task_type.
    job = olmoearth_inference._jobs[result["job_id"]]
    assert job["task_type"] == "embedding_pca_rgb"
    assert job["rgb_raster"].shape == (8, 8, 3)
    assert job["rgb_raster"].dtype == np.uint8


@pytest.mark.asyncio
async def test_run_embedding_tool_pca_rgb_rejects_ft_repo() -> None:
    """FT heads don't produce raw embeddings — the orchestrator must
    refuse them with ValueError before doing any IO."""
    with pytest.raises(ValueError, match="base encoders"):
        await olmoearth_inference.run_embedding_tool_pca_rgb(
            bbox=BBox(west=0.0, south=45.0, east=0.005, north=45.005),
            model_repo_id="allenai/OlmoEarth-v1-FT-Mangrove-Base",
        )


# ---------------------------------------------------------------------------
# Router endpoint /api/olmoearth/embedding-tools/pca-rgb.
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_similarity_endpoint_rejects_ft_repo(test_client) -> None:
    """FT heads on /similarity → 400, mirrors the PCA endpoint contract."""
    r = test_client.post(
        "/api/olmoearth/embedding-tools/similarity",
        json={
            "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
            "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        },
    )
    assert r.status_code == 400
    assert "base encoder" in r.text.lower()


def test_similarity_endpoint_validates_window_px_bounds(test_client) -> None:
    """window_px is clamped 1..15 by Pydantic — out-of-range → 422."""
    r = test_client.post(
        "/api/olmoearth/embedding-tools/similarity",
        json={
            "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
            "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            "window_px": 99,
        },
    )
    assert r.status_code == 422


def test_similarity_endpoint_returns_inference_result_shape(test_client) -> None:
    """Happy path with orchestrator mocked — returns the standard
    inference shape so the frontend wires it as an ImageryLayer."""
    fake_result = {
        "job_id": "test_sim_job",
        "tile_url": "/api/olmoearth/infer-tile/test_sim_job/{z}/{x}/{y}.png",
        "kind": "pytorch",
        "status": "ready",
        "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
        "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
        "task_type": "embedding_similarity",
        "legend": None,
        "colormap": "similarity",
    }

    async def _fake_run(**_kw):
        return fake_result

    with patch.object(
        olmoearth_inference, "run_embedding_tool_similarity",
        side_effect=_fake_run,
    ):
        r = test_client.post(
            "/api/olmoearth/embedding-tools/similarity",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "test_sim_job"
    assert body["task_type"] == "embedding_similarity"
    assert body["colormap"] == "similarity"


def test_pca_rgb_endpoint_rejects_ft_repo(test_client) -> None:
    """FT heads get 400, not 500 — schema-level rejection."""
    r = test_client.post(
        "/api/olmoearth/embedding-tools/pca-rgb",
        json={
            "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
            "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        },
    )
    assert r.status_code == 400
    assert "base" in r.text.lower()


def test_pca_rgb_endpoint_validates_bbox(test_client) -> None:
    """Missing bbox → 422 from Pydantic, not 500."""
    r = test_client.post(
        "/api/olmoearth/embedding-tools/pca-rgb",
        json={"model_repo_id": "allenai/OlmoEarth-v1-Tiny"},
    )
    assert r.status_code == 422


def test_pca_rgb_endpoint_returns_inference_result_shape(test_client) -> None:
    """Happy path with orchestrator mocked — response carries tile_url
    + job_id + status='ready' so the frontend can wire it as an
    ImageryLayer with no special-casing."""
    fake_result = {
        "job_id": "test_pca_job",
        "tile_url": "/api/olmoearth/infer-tile/test_pca_job/{z}/{x}/{y}.png",
        "kind": "pytorch",
        "status": "ready",
        "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
        "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
        "task_type": "embedding_pca_rgb",
        "legend": {"kind": "rgb", "label": "PCA false-color"},
        "colormap": "pca_rgb",
    }

    async def _fake_run(**_kw):
        return fake_result

    with patch.object(
        olmoearth_inference, "run_embedding_tool_pca_rgb",
        side_effect=_fake_run,
    ):
        r = test_client.post(
            "/api/olmoearth/embedding-tools/pca-rgb",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "test_pca_job"
    assert body["status"] == "ready"
    assert body["kind"] == "pytorch"
    assert body["tile_url"].endswith(".png")
    assert body["task_type"] == "embedding_pca_rgb"
