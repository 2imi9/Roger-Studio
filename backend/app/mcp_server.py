"""Roger Studio MCP server — exposes the OlmoEarth toolbox to any
Model-Context-Protocol client (Claude Desktop, Cursor, Claude Code,
OpenAI Responses, NVIDIA NIM tool-calling, …).

Transport: stdio (the default for local MCP servers). Run with:

    python -m app.mcp_server

Or register with Claude Desktop by adding to
``%APPDATA%\\Claude\\claude_desktop_config.json`` (Windows) /
``~/Library/Application Support/Claude/claude_desktop_config.json`` (macOS):

    {
      "mcpServers": {
        "roger-studio": {
          "command": "python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "C:/Users/Frank/OneDrive/Desktop/Github/geoenv-studio/backend",
          "env": {
            "ROGER_API_BASE": "http://localhost:8000"
          }
        }
      }
    }

Architecture
------------
This server is a THIN CLIENT over the running FastAPI backend
(``uvicorn app.main:app --port 8000``). It does NOT duplicate inference
logic — it calls the same HTTP routes the browser UI uses. That means:

  * The FastAPI backend stays the single authoritative surface (one
    place to harden, time out, cache, and log).
  * Model cache + in-flight jobs are SHARED between the browser session
    and any MCP client — click "Run Mangrove" in Claude Desktop, see the
    same tile URL in Roger Studio's SplitMap.
  * User auth stays server-side (NVIDIA_API_KEY, ANTHROPIC_API_KEY, HF
    tokens in ``backend/.env``). MCP clients never see credentials.
  * If the FastAPI server is down, every tool returns an actionable
    error pointing at the start command — better than a hung stdio
    request.

Adding a tool
-------------
1. Write the input schema as a Pydantic model (or a plain dict arg on
   ``@mcp.tool``; FastMCP autogenerates the schema).
2. Implement an ``async def`` that calls ``_api(...)`` against the
   matching FastAPI route.
3. Keep docstrings sharp — they become the tool description LLMs see.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ``ROGER_API_BASE`` lets the user point the MCP server at a remote /
# dockerized Roger Studio instance while still running the MCP proxy
# locally in their Claude Desktop config. Default = the dev uvicorn.
API_BASE = os.environ.get("ROGER_API_BASE", "http://localhost:8000").rstrip("/")

# Per-call HTTP timeout. Inference can take ~30 s cold, catalog / explain
# calls are <15 s, STAC / stats < 10 s. 300 s covers the worst case
# (cold FT-head load + S2 fetch + forward pass) without blocking forever.
HTTP_TIMEOUT_S = float(os.environ.get("ROGER_MCP_TIMEOUT_S", "300"))

mcp = FastMCP(
    "roger-studio",
    instructions=(
        "Roger Studio's Earth-observation toolbox. Use these tools to "
        "run OlmoEarth inference on an AOI, look up Sentinel-2 imagery, "
        "query WorldCover land-cover, compute polygon statistics, and "
        "get plain-language explanations of raster results. Every tool "
        "proxies to a running Roger Studio backend (default "
        f"{API_BASE}); if you hit connection errors, tell the user to "
        "start the backend with `uvicorn app.main:app --port 8000` "
        "inside `geoenv-studio/backend`."
    ),
)


# -----------------------------------------------------------------------------
# HTTP plumbing — one async client reused across tools.
# -----------------------------------------------------------------------------


async def _api(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Issue a single call against the FastAPI backend and unwrap JSON.

    All tool errors are converted into structured ``{error: ..., hint: ...}``
    dicts rather than raised exceptions, so the MCP client surfaces a
    human-readable failure instead of an opaque protocol error. Network
    / connection errors add an explicit hint to start the backend.
    """
    url = f"{API_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            r = await client.request(method, url, json=json, params=params)
    except httpx.ConnectError as e:
        return {
            "error": f"cannot reach Roger Studio backend at {API_BASE}: {e}",
            "hint": (
                "Start the backend with `uvicorn app.main:app --port 8000` "
                "inside geoenv-studio/backend, or set ROGER_API_BASE to "
                "your deployed instance."
            ),
        }
    except httpx.TimeoutException as e:
        return {
            "error": f"request timed out after {HTTP_TIMEOUT_S}s: {e}",
            "hint": (
                "Cold inference on a large AOI can exceed the default "
                "timeout. Either shrink the bbox or raise "
                "ROGER_MCP_TIMEOUT_S."
            ),
        }
    if r.status_code >= 400:
        return {
            "error": f"HTTP {r.status_code}: {r.text[:500]}",
            "status": r.status_code,
        }
    try:
        return r.json()
    except ValueError:
        # Binary response (tile PNG, etc.) — not expected on our JSON
        # routes. Return a lightweight summary so the caller knows.
        return {"ok": True, "content_type": r.headers.get("content-type", "")}


# -----------------------------------------------------------------------------
# Tools — grouped by workflow phase.
# -----------------------------------------------------------------------------


# ---- Catalog + state ---------------------------------------------------------


@mcp.tool()
async def list_olmoearth_ft_heads() -> dict[str, Any]:
    """Return the five OlmoEarth fine-tuned heads currently supported by
    the inference pipeline, plus cache status for each.

    Use this BEFORE calling ``run_olmoearth_inference`` so you can pick a
    head whose training distribution matches the user's AOI (Mangrove →
    tropical coasts, AWF → southern Kenya savanna, LFMC → fire-prone
    regions, ForestLossDriver → pantropical, EcosystemType → global).
    Each entry tells you whether the head is cached on disk (fast) or
    still needs download (~1 GB, slow first call).
    """
    cache = await _api("GET", "/api/olmoearth/cache-status")
    if "error" in cache:
        return cache
    repos = cache.get("repos", {})
    heads = [
        {"repo_id": "allenai/OlmoEarth-v1-FT-LFMC-Base", "task": "Live fuel moisture (regression)"},
        {"repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base", "task": "Mangrove extent (classification)"},
        {"repo_id": "allenai/OlmoEarth-v1-FT-AWF-Base", "task": "Southern-Kenya land use (classification)"},
        {"repo_id": "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base", "task": "Forest loss driver (classification)"},
        {"repo_id": "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base", "task": "Ecosystem type (segmentation, 110 classes)"},
    ]
    # Enrich with cache status
    for h in heads:
        info = repos.get(h["repo_id"]) or {}
        h["cached"] = info.get("status") == "cached"
        h["size_mb"] = (
            round(info["size_bytes"] / 1_000_000, 1)
            if info.get("size_bytes")
            else None
        )
    return {"heads": heads, "note": "Use these repo_ids in run_olmoearth_inference."}


@mcp.tool()
async def get_olmoearth_catalog(
    west: float | None = None,
    south: float | None = None,
    east: float | None = None,
    north: float | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch the live OlmoEarth catalog from HuggingFace — lists every
    allenai/OlmoEarth-v1-* model + dataset with download counts, likes,
    last-modified dates, and (when a bbox is given) a recommended model
    for the user's AOI.

    Backed by a 10-minute TTL cache, so repeated calls within that
    window are cheap. Pass ``force_refresh=True`` to bypass the cache
    (rare — mostly useful when verifying a new FT head is visible).
    """
    params: dict[str, Any] = {}
    if all(v is not None for v in (west, south, east, north)):
        params.update({"west": west, "south": south, "east": east, "north": north})
    if force_refresh:
        params["force"] = "true"
    return await _api("GET", "/api/olmoearth/catalog", params=params or None)


@mcp.tool()
async def get_loaded_olmoearth_models() -> dict[str, Any]:
    """Return the set of OlmoEarth repo_ids currently resident in the
    backend's in-memory model cache.

    A repo_id in this list means the next ``run_olmoearth_inference`` call
    on that head skips the 2–10 s weights-reload from disk — warm path
    ≈ 3 s total vs cold ≈ 20–30 s. The cache clears on backend restart.
    """
    return await _api("GET", "/api/olmoearth/loaded-models")


# ---- Inference ---------------------------------------------------------------


@mcp.tool()
async def run_olmoearth_inference(
    west: float,
    south: float,
    east: float,
    north: float,
    model_repo_id: str,
    date_range: str | None = None,
    sliding_window: bool = False,
) -> dict[str, Any]:
    """Run OlmoEarth inference on a WGS-84 bbox and return the result:
    a deterministic ``job_id``, an XYZ tile URL template for rendering,
    plus task-specific metadata (task_type, class_names, colormap, scene
    info, prediction_value, etc.).

    Use ``list_olmoearth_ft_heads`` first to pick a model whose task
    matches the user's question. ``date_range`` is an RFC-3339 interval
    like "2024-06-01/2024-09-01"; default is a recent summer window.

    Tiles render lazily — the first tile request kicks off the forward
    pass. Use ``explain_raster`` with the returned metadata to get a
    plain-language summary of what the colors mean.
    """
    body: dict[str, Any] = {
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "model_repo_id": model_repo_id,
    }
    if date_range:
        body["date_range"] = date_range
    if sliding_window:
        body["sliding_window"] = True
    return await _api("POST", "/api/olmoearth/infer", json=body)


@mcp.tool()
async def get_olmoearth_demo_pairs() -> dict[str, Any]:
    """Return curated A/B inference demos (Mangrove Niger Delta 2020 vs
    2024, AWF Tsavo 2020 vs 2024, Ecosystem California 2020 vs 2024).

    Each demo has a pre-computed job_id + tile URL for both sides — good
    for showcasing compare-mode to a user without making them define an
    AOI + date range first.
    """
    return await _api("GET", "/api/olmoearth/demo-pairs")


# ---- Explain + context -------------------------------------------------------


@mcp.tool()
async def explain_raster(
    model_repo_id: str,
    task_type: str | None = None,
    colormap: str | None = None,
    west: float | None = None,
    south: float | None = None,
    east: float | None = None,
    north: float | None = None,
    scene_id: str | None = None,
    scene_datetime: str | None = None,
    scene_cloud_cover: float | None = None,
    class_names: list[str] | None = None,
    top_classes: list[dict[str, Any]] | None = None,
    prediction_value: float | None = None,
    units: str | None = None,
    stub_reason: str | None = None,
) -> dict[str, Any]:
    """Ask Roger's LLM agent (NIM → Claude → Gemma → templated fallback)
    to explain what an OlmoEarth raster represents in 2–3 plain-English
    paragraphs.

    Pass whatever metadata you have from a prior ``run_olmoearth_inference``
    call — at minimum ``model_repo_id`` and ``task_type``. Richer context
    (bbox, scene info, top classes with scores) yields a better answer.

    Returns ``{explanation, source, model}`` — ``source`` tells you which
    provider answered (``nim`` / ``claude`` / ``gemma`` / ``fallback``).
    """
    bbox = None
    if all(v is not None for v in (west, south, east, north)):
        bbox = {"west": west, "south": south, "east": east, "north": north}
    body: dict[str, Any] = {"model_repo_id": model_repo_id}
    if task_type is not None:
        body["task_type"] = task_type
    if colormap is not None:
        body["colormap"] = colormap
    if bbox is not None:
        body["bbox"] = bbox
    if scene_id is not None:
        body["scene_id"] = scene_id
    if scene_datetime is not None:
        body["scene_datetime"] = scene_datetime
    if scene_cloud_cover is not None:
        body["scene_cloud_cover"] = scene_cloud_cover
    if class_names is not None:
        body["class_names"] = class_names
    if top_classes is not None:
        body["top_classes"] = top_classes
    if prediction_value is not None:
        body["prediction_value"] = prediction_value
    if units is not None:
        body["units"] = units
    if stub_reason is not None:
        body["stub_reason"] = stub_reason
    return await _api("POST", "/api/explain-raster", json=body)


# ---- Imagery + land-cover ----------------------------------------------------


@mcp.tool()
async def search_sentinel2_imagery(
    west: float,
    south: float,
    east: float,
    north: float,
    datetime_range: str,
    max_cloud_cover: float = 20.0,
    limit: int = 10,
) -> dict[str, Any]:
    """Search Microsoft Planetary Computer's Sentinel-2 L2A STAC catalog
    for scenes intersecting a bbox + time range.

    ``datetime_range`` is an RFC-3339 interval ("2024-06-01/2024-09-01").
    Returns up to ``limit`` scenes sorted by cloud cover ascending, so
    the first entries are the least-cloudy. Useful for picking a scene
    to feed back into ``run_olmoearth_inference``.
    """
    body = {
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "datetime": datetime_range,
        "max_cloud_cover": max_cloud_cover,
        "limit": limit,
    }
    return await _api("POST", "/api/stac/search", json=body)


@mcp.tool()
async def get_sentinel2_composite_tile_url(
    west: float,
    south: float,
    east: float,
    north: float,
    datetime_range: str,
    max_cloud_cover: float = 20.0,
) -> dict[str, Any]:
    """Register a least-cloudy Sentinel-2 L2A composite for the bbox and
    return an XYZ tile URL template you can render via any map library.

    Handy for sanity-checking an AOI (use this as a true-color basemap)
    before running OlmoEarth inference on top of it.
    """
    body = {
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "datetime": datetime_range,
        "max_cloud_cover": max_cloud_cover,
    }
    return await _api("POST", "/api/stac/composite-tile-url", json=body)


@mcp.tool()
async def classify_land_cover(
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict[str, Any]:
    """Return ESA WorldCover 10 m/pixel land-cover class percentages for
    the bbox (global 2021 release, falls back to 2020 for regions not
    covered by the newer version).

    Output includes per-class percentage (Forest, Cropland, Urban,
    Water, Grassland, Barren, Wetland, Snow/Ice, etc.). Useful as a
    sanity check or a global baseline when the user hasn't picked an FT
    head yet.
    """
    return await _api(
        "POST",
        "/api/analyze",
        json={"area": {"west": west, "south": south, "east": east, "north": north}},
    )


# ---- Polygon analysis --------------------------------------------------------


@mcp.tool()
async def polygon_stats(
    geometry: dict[str, Any],
    include_elevation: bool = True,
    resolution: int = 20,
) -> dict[str, Any]:
    """Compute perimeter, area, centroid, bbox, and (optionally) elevation
    statistics for a GeoJSON Polygon or MultiPolygon.

    ``geometry`` must be a GeoJSON geometry dict (``{"type": "Polygon",
    "coordinates": [[...]]}``), not a Feature. Elevation is sampled from
    SRTM at ``resolution`` m/px (5–60); pass ``include_elevation=False``
    to skip the SRTM fetch if you only need the 2D metrics.
    """
    body = {
        "geometry": geometry,
        "include_elevation": include_elevation,
        "resolution": resolution,
    }
    return await _api("POST", "/api/polygon-stats", json=body)


# ---- Admin -------------------------------------------------------------------


@mcp.tool()
async def load_olmoearth_repo(
    repo_id: str,
    repo_type: str = "model",
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Kick off a background ``snapshot_download`` of a HuggingFace
    OlmoEarth repo into the local cache.

    Useful when you want to pre-warm an FT head before running inference
    so the first ``run_olmoearth_inference`` call is fast. ``repo_type``
    is ``"model"`` for FT heads / encoders, ``"dataset"`` for olmoearth
    project datasets. ``hf_token`` only needed for gated repos.
    """
    body: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type}
    if hf_token:
        body["hf_token"] = hf_token
    return await _api("POST", "/api/olmoearth/load", json=body)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio. Invoked via ``python -m app.mcp_server``."""
    # Log to stderr only — stdout is reserved for MCP protocol messages,
    # and any stray print() there breaks the JSON-RPC framing.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Roger Studio MCP server starting — API_BASE=%s", API_BASE)
    mcp.run()


if __name__ == "__main__":
    main()
