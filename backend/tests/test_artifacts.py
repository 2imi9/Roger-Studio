"""Tests for the agent artifact pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services import artifacts  # noqa: E402


@pytest.fixture(autouse=True)
def _clear():
    artifacts.clear_artifacts()
    yield
    artifacts.clear_artifacts()


def test_save_and_get_round_trip() -> None:
    payload = b"month,ndvi\n2024-06,0.72\n"
    public = artifacts.save_artifact(
        content=payload,
        filename="ndvi.csv",
        content_type="text/csv",
        summary="6-month NDVI",
    )
    assert public["filename"] == "ndvi.csv"
    assert public["content_type"] == "text/csv"
    assert public["size_bytes"] == len(payload)
    assert public["download_url"].startswith("/api/artifacts/")
    assert "path" not in public  # never leaked to the LLM/UI

    meta = artifacts.get_artifact(public["id"])
    assert meta is not None
    assert meta.path.exists()
    assert meta.path.read_bytes() == payload


def test_save_keeps_file_inside_artifacts_dir_despite_hostile_name() -> None:
    """A filename with path separators / traversal segments must not let the
    resulting file escape the artifacts dir."""
    public = artifacts.save_artifact(
        content=b"x",
        filename="../oops/evil name.csv",
        content_type="text/csv",
        summary="",
    )
    meta = artifacts.get_artifact(public["id"])
    assert meta is not None
    # Security invariant: the on-disk file's immediate parent must be our
    # artifacts dir — no directory traversal possible, regardless of what
    # characters we let through for human-readable filenames.
    assert meta.path.parent.resolve() == artifacts._ARTIFACTS_DIR.resolve()
    # No path separators in the on-disk name.
    assert "/" not in meta.path.name and "\\" not in meta.path.name


def test_get_missing_artifact_returns_none() -> None:
    assert artifacts.get_artifact("nonexistent-id") is None


def test_public_dict_has_iso_timestamp() -> None:
    public = artifacts.save_artifact(b"x", "a.txt", "text/plain", "")
    assert "created_at_iso" in public
    # ISO 8601 starts with YYYY-
    assert public["created_at_iso"][:4].isdigit()


@pytest.mark.asyncio
async def test_route_streams_artifact_file() -> None:
    """GET /api/artifacts/{id} returns the persisted bytes with attachment header."""
    import httpx
    from app.main import app

    public = artifacts.save_artifact(
        b"hello,world\n1,2\n", "data.csv", "text/csv", "demo",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get(f"/api/artifacts/{public['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert r.content == b"hello,world\n1,2\n"


@pytest.mark.asyncio
async def test_route_404_on_unknown_artifact() -> None:
    import httpx
    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/artifacts/doesnotexist")
    assert r.status_code == 404
