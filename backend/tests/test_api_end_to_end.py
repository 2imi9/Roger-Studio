"""End-to-end FastAPI tests that prove the full request pipeline works —
routers → services → real OlmoEarth / PC / WorldCover backends.

These tests use ``httpx.AsyncClient`` against the live FastAPI app (no uvicorn
process needed). Everything here that hits PC/HF is gated behind the
``network`` marker so offline CI stays green.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture
def app():
    from app.main import app as fastapi_app  # noqa: PLC0415
    return fastapi_app


@pytest.fixture
async def client(app):
    """ASGI transport so we skip the real TCP layer."""
    import httpx  # noqa: PLC0415

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.mark.network
@pytest.mark.asyncio
async def test_analyze_returns_real_worldcover_for_seattle(client) -> None:
    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    r = await client.post("/api/analyze", json={"area": bbox}, timeout=120.0)
    assert r.status_code == 200, r.text
    data = r.json()

    # Real WorldCover path should have flagged itself.
    assert data["olmoearth"]["land_cover_source"] == "worldcover-2021"

    # Seattle downtown = Built-up should be the majority class.
    classes_by_name = {c["name"]: c["percentage"] for c in data["land_cover"]}
    assert classes_by_name.get("Built-up", 0) > 50.0
    assert classes_by_name.get("Permanent water bodies", 0) > 1.0


@pytest.mark.network
@pytest.mark.asyncio
async def test_infer_endpoint_runs_real_forward_and_serves_tile(client) -> None:
    from app.services import olmoearth_inference as OI  # noqa: PLC0415

    OI.clear_jobs()

    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    payload = {
        "bbox": bbox,
        "model_repo_id": "allenai/OlmoEarth-v1-Nano",
        "date_range": "2024-06-01/2024-09-30",
    }
    r = await client.post("/api/olmoearth/infer", json=payload, timeout=300.0)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "pytorch", data
    assert data["status"] == "ready"
    assert data["scene_id"].startswith("S2")

    # Pull a tile out of the prediction raster.
    job_id = data["job_id"]
    # zoom 14, tile that covers Seattle downtown.
    tile_resp = await client.get(
        f"/api/olmoearth/infer-tile/{job_id}/14/2624/5721.png", timeout=30.0
    )
    assert tile_resp.status_code == 200
    assert tile_resp.headers["content-type"] == "image/png"
    assert tile_resp.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(tile_resp.content) > 0


@pytest.mark.network
@pytest.mark.asyncio
async def test_infer_endpoint_runs_ft_mangrove_and_surfaces_tags(client) -> None:
    """FT path: response carries task_type + class_names + per-class probs."""
    from app.services import olmoearth_inference as OI  # noqa: PLC0415

    OI.clear_jobs()

    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    payload = {
        "bbox": bbox,
        "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        "date_range": "2024-06-01/2024-09-30",
    }
    r = await client.post("/api/olmoearth/infer", json=payload, timeout=300.0)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "pytorch"
    # Per olmoearth_projects config, Mangrove is per-pixel segmentation.
    assert data["task_type"] == "segmentation"
    assert data["num_classes"] == 4
    # Class names now sourced from the published rslearn legend.
    assert data["class_names"] == ["nodata", "mangrove", "water", "other"]
    assert data["class_names_tentative"] is False
    # Segmentation jobs don't return scene-level class_probs — the argmax
    # class raster is the primary output, rendered through the tile layer.
    assert data.get("class_probs") is None
    # Legend carries the published per-class hex colors.
    legend = data["legend"]
    assert legend["kind"] == "segmentation"
    assert legend.get("colors_source") == "published"
    assert [c["name"] for c in legend["classes"]] == data["class_names"]
    assert [c["color"] for c in legend["classes"]] == [
        "#6b7280", "#94eb63", "#63d8eb", "#eba963",
    ]

    # Tile route still works on the FT job (uses max-prob scalar raster).
    tile_resp = await client.get(
        f"/api/olmoearth/infer-tile/{data['job_id']}/14/2624/5721.png", timeout=30.0
    )
    assert tile_resp.status_code == 200
    assert tile_resp.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_infer_endpoint_falls_back_to_stub_on_bad_model(client) -> None:
    """Hitting a non-existent repo should fall back to the stub path, not 500."""
    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    payload = {
        "bbox": bbox,
        "model_repo_id": "allenai/does-not-exist-xyz",
        "date_range": "2024-06-01/2024-09-30",
    }
    r = await client.post("/api/olmoearth/infer", json=payload, timeout=120.0)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "stub"
    assert "stub_reason" in data


@pytest.mark.network
@pytest.mark.asyncio
async def test_geojson_export_rejects_lfmc_regression_with_actionable_400(client) -> None:
    """LFMC is a regression head — there's no class raster to vectorise.
    The endpoint must 400 with a message that points the user at the
    Export-as-COG path, NOT 500 or fall back to a stub. Network-marked
    because the underlying inference must actually run for task_type
    to land in the cached job dict.
    """
    from app.services import olmoearth_inference as OI  # noqa: PLC0415

    OI.clear_jobs()

    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    payload = {
        "bbox": bbox,
        "model_repo_id": "allenai/OlmoEarth-v1-FT-LFMC-Base",
    }
    r = await client.post("/api/olmoearth/ft-classification/geojson", json=payload, timeout=300.0)
    assert r.status_code == 400, r.text
    detail = r.json().get("detail", "")
    assert "task_type='regression'" in detail
    assert "Export-as-COG" in detail


@pytest.mark.asyncio
async def test_few_shot_router_returns_400_when_all_class_points_fall_outside_aoi(
    client, monkeypatch,
) -> None:
    """Few-shot validation: if a class's labelled points all land outside
    the AOI / in nodata patches, the service raises SentinelFetchError
    with a "labelled point" message. The router must surface that as a
    400 (user input error), not a 500. Without the disambiguating
    catch, the same exception type covers transient infra outages
    (which deserve 503), so the router introspects the message text.
    """
    from app.services import olmoearth_inference as OI  # noqa: PLC0415
    from app.services import olmoearth_model as M  # noqa: PLC0415
    from rasterio.transform import from_bounds  # noqa: PLC0415
    from rasterio.crs import CRS as _CRS  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    OI.clear_jobs()

    # Synth a tiny embedding tensor over a small AOI; the second class's
    # labelled point will be outside the AOI bbox so the service raises.
    h_patch, w_patch, embed_dim = 8, 8, 4
    emb = np.ones((h_patch, w_patch, embed_dim), dtype=np.float32)
    transform = from_bounds(0.0, 0.0, 1.0, 1.0, w_patch, h_patch)
    crs = _CRS.from_string("EPSG:4326")

    async def _fake_export(**kwargs):
        return {
            "embedding_float32": emb,
            "transform": transform,
            "crs": crs,
            "embedding_dim": embed_dim,
            "patch_size": 4,
            "target_gsd_m": 10.0,
            "chunks_total": 1,
            "chunks_processed": 1,
            "chunks_failed": 0,
            "scene_ids": [None],
            "scene_datetimes": [None],
            "n_periods": 1,
            "period_days": 30,
            "modality": "sentinel2_l2a",
            "model_repo_id": "x",
        }

    monkeypatch.setattr(OI, "_run_chunked_embedding_export", _fake_export)
    monkeypatch.setattr(M, "load_encoder", lambda repo_id: ("FAKE_MODEL", "cpu"))

    payload = {
        "bbox": {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0},
        "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
        "classes": [
            {"name": "in", "color": "#ff0000", "points": [{"lon": 0.5, "lat": 0.5}]},
            # Outside the AOI bbox → patch (row, col) round outside the
            # raster → service raises SentinelFetchError → router returns 400.
            {"name": "out", "color": "#00ff00", "points": [{"lon": 50.0, "lat": 50.0}]},
        ],
    }
    r = await client.post("/api/olmoearth/embedding-tools/few-shot", json=payload, timeout=60.0)
    assert r.status_code == 400, r.text
    detail = r.json().get("detail", "")
    assert "labelled point" in detail
    assert "out" in detail.lower() or "outside" in detail.lower() or "nodata" in detail.lower()


@pytest.mark.asyncio
async def test_few_shot_router_rejects_non_base_encoder_with_400(client) -> None:
    """FT heads (Mangrove, AWF, …) emit task-specific outputs, not raw
    embeddings — few-shot only works on the four base encoders. The
    router must reject FT repo ids at validation time with a 400
    (not 500 or stub fallback)."""
    payload = {
        "bbox": {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0},
        "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        "classes": [
            {"name": "a", "color": "#ff0000", "points": [{"lon": 0.25, "lat": 0.25}]},
            {"name": "b", "color": "#00ff00", "points": [{"lon": 0.75, "lat": 0.75}]},
        ],
    }
    r = await client.post("/api/olmoearth/embedding-tools/few-shot", json=payload, timeout=10.0)
    assert r.status_code == 400, r.text
    detail = r.json().get("detail", "")
    assert "base encoder" in detail.lower() or "FT" in detail or "embedding" in detail.lower()


@pytest.mark.asyncio
async def test_geojson_export_surfaces_stub_reason_when_inference_failed(client) -> None:
    """Audit finding 2026-04-25: when inference falls back to a stub, the
    GeoJSON export endpoint used to report a confusing
    ``task_type=None, not a classification`` error. The real failure
    (PC outage / no scenes / breaker tripped) was buried in the stub
    reason. Now we surface that reason as a 503 with Retry-After."""
    from app.services import olmoearth_inference as OI  # noqa: PLC0415

    OI.clear_jobs()

    bbox = {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63}
    payload = {
        "bbox": bbox,
        "model_repo_id": "allenai/does-not-exist-xyz",
    }
    r = await client.post("/api/olmoearth/ft-classification/geojson", json=payload, timeout=120.0)
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "60"
    body = r.json()
    detail = body.get("detail", "")
    assert "fell back to a stub" in detail
    assert "Retry when network stabilises" in detail
    # Make sure the old confusing message is gone.
    assert "task_type=None" not in detail
