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
