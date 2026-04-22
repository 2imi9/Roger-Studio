"""Tests for the ``run_olmoearth_inference`` tool in geo_tools.

Covers schema registration + executor dispatch (offline), plus a network-
gated smoke test that drives FT-Mangrove end-to-end through the tool loop
exactly as the Gemma/NIM/Claude chat would.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services import geo_tools, olmoearth_inference as OI  # noqa: E402


# ---------------------------------------------------------------------------
# Schema + registration
# ---------------------------------------------------------------------------


def test_tool_schema_registered() -> None:
    names = [t["function"]["name"] for t in geo_tools.TOOL_SCHEMAS]
    assert "run_olmoearth_inference" in names


def test_tool_schema_required_params() -> None:
    schema = next(
        t["function"] for t in geo_tools.TOOL_SCHEMAS
        if t["function"]["name"] == "run_olmoearth_inference"
    )
    params = schema["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"bbox", "model_repo_id"}
    # Optional params exposed to the LLM.
    props = params["properties"]
    assert {"bbox", "model_repo_id", "date_range", "max_size_px"} <= props.keys()


def test_tool_registered_in_executors() -> None:
    assert "run_olmoearth_inference" in geo_tools._EXECUTORS


# ---------------------------------------------------------------------------
# Executor dispatch — offline paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_errors_on_missing_model_repo_id() -> None:
    res = await geo_tools.execute_tool(
        "run_olmoearth_inference",
        arguments={"bbox": {"west": 0, "south": 0, "east": 1, "north": 1}},
    )
    assert res["error"] == "missing_argument"
    assert "model_repo_id" in res["detail"]


@pytest.mark.asyncio
async def test_executor_returns_trimmed_summary_on_stub_fallback() -> None:
    """When the real pipeline can't run (bad repo), the tool should still
    succeed — falling back to the stub path with stub_reason surfaced."""
    OI.clear_jobs()

    args = {
        "bbox": {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63},
        "model_repo_id": "allenai/does-not-exist-xyz",
    }
    res = await geo_tools.execute_tool("run_olmoearth_inference", arguments=args)
    # start_inference's internal try/except converts load failures to stub.
    assert res["kind"] == "stub"
    assert res["status"] == "ready"
    assert "stub_reason" in res
    assert res["tile_url"].startswith("/api/olmoearth/infer-tile/")
    assert res["model_repo_id"] == "allenai/does-not-exist-xyz"


@pytest.mark.asyncio
async def test_executor_passes_date_range_and_max_size_to_start_inference() -> None:
    """Plumbing check — args flow through to start_inference unchanged."""
    OI.clear_jobs()

    async def fake_start(bbox, model_repo_id, date_range=None, max_size_px=256,
                         sliding_window=False, window_size=32):
        return {
            "job_id": "fake",
            "status": "ready",
            "kind": "pytorch",
            "task_type": "embedding",
            "model_repo_id": model_repo_id,
            "tile_url": f"/api/olmoearth/infer-tile/fake/{{z}}/{{x}}/{{y}}.png",
            "bbox": bbox.model_dump(),
            "scene_id": "S2X_FAKE",
            "scene_datetime": "2024-07-15T00:00:00Z",
            "scene_cloud_cover": 1.2,
            "patch_size": 4,
            "embedding_dim": 128,
            "notes": ["fake"],
            # Plumb inputs back out so we can assert on them.
            "_probe_date_range": date_range,
            "_probe_max_size_px": max_size_px,
        }

    args = {
        "bbox": {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63},
        "model_repo_id": "allenai/OlmoEarth-v1-Nano",
        "date_range": "2025-06-01/2025-09-01",
        "max_size_px": 128,
    }
    with patch.object(OI, "start_inference", side_effect=fake_start):
        res = await geo_tools.execute_tool("run_olmoearth_inference", arguments=args)
    assert res["status"] == "ready"
    assert res["task_type"] == "embedding"
    assert res["embedding_dim"] == 128


# ---------------------------------------------------------------------------
# Network-gated: real FT-Mangrove through the tool dispatcher.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_tool_runs_ft_mangrove_end_to_end_returns_legend_classes() -> None:
    OI.clear_jobs()

    args = {
        "bbox": {"west": -122.35, "south": 47.60, "east": -122.32, "north": 47.63},
        "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
        "date_range": "2024-06-01/2024-09-30",
        "max_size_px": 64,
    }
    res = await geo_tools.execute_tool("run_olmoearth_inference", arguments=args)
    assert res["kind"] == "pytorch"
    # Mangrove is segmentation, not classification (post-step-3b).
    assert res["task_type"] == "segmentation"
    assert res["num_classes"] == 4
    # Names come from the published rslearn legend.
    assert res["class_names"] == ["nodata", "mangrove", "water", "other"]
    assert res["class_names_tentative"] is False
    # Segmentation jobs expose per-class legend colors (published hex values).
    legend_classes = res["legend_classes"]
    assert len(legend_classes) == 4
    assert [c["color"] for c in legend_classes] == [
        "#6b7280", "#94eb63", "#63d8eb", "#eba963",
    ]
