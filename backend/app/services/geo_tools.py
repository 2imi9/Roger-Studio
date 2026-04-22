"""OpenAI-compatible tool schemas + executors for the Gemma 4 geo-agent.

Tools registered:

  ``query_polygon``              — look up one polygon from the chat's scene_context
  ``query_olmoearth``            — OlmoEarth catalog + project coverage for a bbox
  ``query_polygon_stats``        — perimeter / area / elevation stats for a polygon
  ``query_ndvi_timeseries``      — monthly Sentinel-2 NDVI over a bbox (REAL)
  ``search_stac_imagery``        — Planetary Computer STAC search
  ``get_composite_tile_url``     — PC mosaic endpoint → XYZ tile URL
  ``run_olmoearth_inference``    — run a real OlmoEarth forward pass over a bbox,
                                   return tile URL + task-tagged prediction (REAL)
  ``get_higher_res_patch``       — zoomed basemap tile for a polygon (STUB)

The schemas match the ``--tool-call-parser gemma4`` output from vLLM, so the
assistant message ``tool_calls`` entries can be dispatched directly via
``execute_tool``. Each executor returns a plain-JSON dict suitable for the
``content`` field of a ``role=tool`` message.

Remaining stubs return honest ``not_implemented`` payloads with the inputs the
agent should surface to the user — never faked numeric data — so the LLM can
tell the user what's missing rather than hallucinate a response.
"""
from __future__ import annotations

import asyncio
import calendar
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.models.schemas import BBox
from app.services import (
    artifacts as artifacts_svc,
    olmoearth_datasets,
    olmoearth_inference,
    polygon_stats as polygon_stats_svc,
    stac_imagery as stac_svc,
)
from app.services.sentinel2_fetch import (
    SentinelFetchError,
    fetch_s2_composite,
)

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_polygon",
            "description": (
                "Look up a single polygon from the current scene by its id or "
                "index. Returns the polygon's class, confidence, bbox, and any "
                "validation block attached by prior Gemma runs. Use this when "
                "the user asks 'why was polygon N labelled X?' or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "polygon_id": {
                        "type": "string",
                        "description": (
                            "Polygon id from properties.id, or a stringified "
                            "integer index into the feature list."
                        ),
                    },
                },
                "required": ["polygon_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_olmoearth",
            "description": (
                "Return the Ai2 OlmoEarth catalog (datasets + models) and "
                "list any OlmoEarth project-labelled regions that overlap "
                "the given bbox. Use when the user asks what OlmoEarth "
                "data or models exist for an area."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "description": "WGS84 bounding box",
                        "properties": {
                            "west":  {"type": "number"},
                            "south": {"type": "number"},
                            "east":  {"type": "number"},
                            "north": {"type": "number"},
                        },
                        "required": ["west", "south", "east", "north"],
                    },
                },
                "required": ["bbox"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_olmoearth_inference",
            "description": (
                "Run a real OlmoEarth forward pass over a bbox and return a "
                "tile-layer URL the UI can drop onto the map, plus the "
                "task-tagged prediction. Four task paths:\n"
                " - Base encoders (Nano / Tiny / Base / Large) → embedding "
                "scalar raster (PCA of per-patch features) for generic "
                "visualisation.\n"
                " - Classification fine-tunes (Mangrove, ForestLossDriver, "
                "EcosystemTypeMapping) → scene-level argmax class + per-class "
                "softmax probabilities, plus per-class legend colors.\n"
                " - Segmentation fine-tunes (AWF) → per-patch class raster "
                "rendered as a discrete thematic map.\n"
                " - Regression fine-tunes (LFMC) → per-scene scalar value in "
                "task units (e.g. % live fuel moisture).\n"
                "The backend fetches a least-cloudy Sentinel-2 L2A composite "
                "from Planetary Computer automatically, so this tool is "
                "self-contained — just pick a bbox + repo id. Prefer "
                "calling `query_olmoearth` first to see the live FT-model "
                "catalog + which project regions overlap the bbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "description": "WGS84 bounding box for the inference area.",
                        "properties": {
                            "west":  {"type": "number"},
                            "south": {"type": "number"},
                            "east":  {"type": "number"},
                            "north": {"type": "number"},
                        },
                        "required": ["west", "south", "east", "north"],
                    },
                    "model_repo_id": {
                        "type": "string",
                        "description": (
                            "Hugging Face repo id. Base encoders: "
                            "allenai/OlmoEarth-v1-{Nano,Tiny,Base,Large}. "
                            "Fine-tuned: allenai/OlmoEarth-v1-FT-{Mangrove,"
                            "LFMC,AWF,ForestLossDriver,EcosystemTypeMapping}-Base."
                        ),
                    },
                    "date_range": {
                        "type": "string",
                        "description": (
                            "RFC-3339 interval like '2024-06-01/2024-09-30'. "
                            "Pick a season that matches the task "
                            "(summer for vegetation, dry season for mangroves). "
                            "Defaults to '2024-04-01/2024-10-01'."
                        ),
                    },
                    "max_size_px": {
                        "type": "integer",
                        "description": (
                            "Longest side of the fetched S2 tile in pixels "
                            "at 10 m/pixel. 256 matches OlmoEarth's 2.56 km "
                            "pretraining tile (recommended). Smaller = "
                            "faster CPU inference; larger = slower."
                        ),
                        "minimum": 32,
                        "maximum": 512,
                    },
                    "sliding_window": {
                        "type": "boolean",
                        "description": (
                            "When true, tile the S2 image into non-overlapping "
                            "window_size windows and run the FT head per-tile, "
                            "stitching the outputs into a spatial prediction "
                            "raster. Use this for scene-level tasks "
                            "(classification / regression like LFMC) where "
                            "otherwise the whole bbox gets a single uniform "
                            "prediction — or for larger bboxes that would "
                            "otherwise move off-distribution in one pass. "
                            "~O(n² / window_size²) extra forward passes, so "
                            "slower."
                        ),
                    },
                    "window_size": {
                        "type": "integer",
                        "description": (
                            "Tile size in pixels at 10 m/pixel for "
                            "sliding-window inference. 32 matches training "
                            "tile size for most FT heads. Must be divisible "
                            "by the model's patch_size."
                        ),
                        "minimum": 16,
                        "maximum": 128,
                    },
                },
                "required": ["bbox", "model_repo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_ndvi_masked",
            "description": (
                "Like query_ndvi_timeseries but the NDVI statistics are "
                "computed ONLY over pixels that belong to specified class "
                "indices from a previous run_olmoearth_inference segmentation "
                "job. Use this when the user asks a question like 'has the "
                "mangrove area been getting drier?' — run FT-Mangrove first "
                "with run_olmoearth_inference, then pass that job_id here "
                "with class_indices=[1] (the 'mangrove' class) to get the "
                "mangrove-only NDVI trend instead of the bbox average "
                "(which dilutes with water and upland pixels). Produces a "
                "CSV artifact just like query_ndvi_timeseries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "job_id returned by a prior run_olmoearth_inference "
                            "call. Its cached segmentation raster + bbox + CRS "
                            "are reused."
                        ),
                    },
                    "class_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Class indices from the job's legend to keep "
                            "(all other classes are masked out). E.g. [1] "
                            "for just mangrove, [1, 2] for mangrove + water."
                        ),
                    },
                    "months": {
                        "type": "integer",
                        "description": "Trailing months to sample. Max 24.",
                        "minimum": 1,
                        "maximum": 24,
                    },
                    "max_size_px": {
                        "type": "integer",
                        "description": "Longest side in pixels at 10 m/pixel.",
                        "minimum": 16,
                        "maximum": 256,
                    },
                },
                "required": ["job_id", "class_indices", "months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_ndvi_timeseries",
            "description": (
                "Monthly Sentinel-2 L2A NDVI mean/median/p10/p90 over a bbox "
                "for the past N months. Backed by Microsoft Planetary Computer "
                "— least-cloudy scene per month, B08 and B04 read at 10 m/pixel. "
                "Returns per-month scene id + datetime + cloud cover alongside "
                "the NDVI stats, or a 'no_scene' status when the month was too "
                "cloudy to sample."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "west":  {"type": "number"},
                            "south": {"type": "number"},
                            "east":  {"type": "number"},
                            "north": {"type": "number"},
                        },
                        "required": ["west", "south", "east", "north"],
                    },
                    "months": {
                        "type": "integer",
                        "description": "How many trailing months to cover (capped at 24).",
                        "minimum": 1,
                        "maximum": 24,
                    },
                    "max_size_px": {
                        "type": "integer",
                        "description": (
                            "Longest side of the sampled grid per month, in "
                            "pixels at 10 m/pixel. 64 keeps latency low; "
                            "raise for more spatial averaging."
                        ),
                        "minimum": 16,
                        "maximum": 256,
                    },
                },
                "required": ["bbox", "months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_polygon_stats",
            "description": (
                "Return perimeter (km), area (km²), and elevation "
                "min/median/max/mean (m) for a polygon — the same readout "
                "as Google Earth's measurement tool. Pass either "
                "'polygon_id' (looked up from the scene) or 'geometry' "
                "(GeoJSON Polygon). Use when the user asks 'how big is this' "
                "or wants terrain stats for a drawn shape."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "polygon_id": {
                        "type": "string",
                        "description": "Polygon id or integer index in scene_context.polygon_features.",
                    },
                    "geometry": {
                        "type": "object",
                        "description": "GeoJSON Polygon (type + coordinates). Use this when no polygon_id is available.",
                    },
                    "include_elevation": {
                        "type": "boolean",
                        "description": "Skip Open-Meteo elevation lookup for a faster perimeter/area-only response.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_stac_imagery",
            "description": (
                "Search Microsoft Planetary Computer STAC for satellite imagery "
                "scenes over a bbox + time window. Returns scene IDs, capture "
                "datetime, cloud cover, and asset bands. Use when the user asks "
                "'what imagery exists here?' or 'find recent Sentinel-2 scenes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "west":  {"type": "number"},
                            "south": {"type": "number"},
                            "east":  {"type": "number"},
                            "north": {"type": "number"},
                        },
                        "required": ["west", "south", "east", "north"],
                    },
                    "datetime": {
                        "type": "string",
                        "description": "RFC 3339 interval like '2024-06-01/2024-09-01' or a single date.",
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "STAC collection ids. Default: ['sentinel-2-l2a']. Others: 'landsat-c2-l2', 'sentinel-1-grd', 'naip'.",
                    },
                    "max_cloud_cover": {
                        "type": "number",
                        "description": "Drop scenes above this percent (Sentinel-2/Landsat only).",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["bbox", "datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_composite_tile_url",
            "description": (
                "Register a least-cloudy mosaic on Microsoft Planetary Computer "
                "and return an XYZ tile URL the frontend can drop onto the map "
                "as a raster layer. Great for 'show me cloud-free Sentinel-2 "
                "for this bbox in June'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "west":  {"type": "number"},
                            "south": {"type": "number"},
                            "east":  {"type": "number"},
                            "north": {"type": "number"},
                        },
                        "required": ["west", "south", "east", "north"],
                    },
                    "datetime": {"type": "string"},
                    "collection": {
                        "type": "string",
                        "description": "STAC collection id. Default 'sentinel-2-l2a'.",
                    },
                    "assets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Band names. Default true-color per collection (S2: B04,B03,B02).",
                    },
                    "max_cloud_cover": {"type": "number"},
                },
                "required": ["bbox", "datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_higher_res_patch",
            "description": (
                "Fetch a higher-zoom basemap tile for a polygon. Returns "
                "'not_implemented' until a basemap tile source is wired in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "polygon_id": {"type": "string"},
                    "zoom": {
                        "type": "integer",
                        "description": "Web-mercator zoom (12-20).",
                        "minimum": 10,
                        "maximum": 20,
                    },
                },
                "required": ["polygon_id", "zoom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_raster_histogram",
            "description": (
                "Return the per-class pixel distribution of a finished "
                "OlmoEarth classification / segmentation inference raster. "
                "Call this FIRST when explaining a classification raster — "
                "it tells you which classes actually appear on the tile "
                "(with pixel counts + percentages + colors + names) so you "
                "can ground your explanation in real pixel truth instead "
                "of paraphrasing the full class catalog. Only works for "
                "classification / segmentation tasks; returns an error "
                "for regression / embedding — use query_raster_scalar_stats "
                "for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "Inference job_id returned by start_inference. "
                            "Optional — defaults to the job_id in scene_context."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return only the top N most-present classes. Default 20.",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_raster_scalar_stats",
            "description": (
                "Summary stats (mean / min / max / p10 / p50 / p90) of a "
                "finished OlmoEarth regression or embedding raster. Values "
                "are un-normalized to the task's native units (e.g. % "
                "moisture for LFMC). Call this for regression tasks instead "
                "of query_raster_histogram."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "Inference job_id. Optional — defaults to "
                            "scene_context.job_id."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


def _coerce_bbox(raw: Any) -> BBox:
    """Turn an LLM-provided bbox dict into a validated BBox."""
    if isinstance(raw, BBox):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"bbox must be an object, got {type(raw).__name__}")
    return BBox(**raw)


def _find_polygon(scene_context: dict[str, Any], polygon_id: str) -> dict[str, Any] | None:
    """Locate a feature in scene_context by id or integer index."""
    features = (scene_context or {}).get("polygon_features") or []
    for feat in features:
        props = feat.get("properties") or {}
        if str(props.get("id")) == polygon_id or str(props.get("polygon_id")) == polygon_id:
            return feat
    try:
        idx = int(polygon_id)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(features):
        return features[idx]
    return None


async def _tool_query_polygon(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    polygon_id = str(args.get("polygon_id", ""))
    feat = _find_polygon(scene_context, polygon_id)
    if feat is None:
        features = (scene_context or {}).get("polygon_features") or []
        return {
            "found": False,
            "polygon_id": polygon_id,
            "reason": (
                "No polygon with that id/index in the current scene "
                f"({len(features)} features loaded)."
            ),
        }
    props = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates", [[]])
    ring = coords[0] if coords else []
    if ring:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        bbox = {"west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats)}
    else:
        bbox = None
    return {
        "found": True,
        "polygon_id": polygon_id,
        "class": props.get("class_name") or props.get("class"),
        "confidence": props.get("confidence"),
        "bbox": bbox,
        "validation": props.get("validation"),
        "pipeline": props.get("pipeline"),
    }


async def _tool_query_olmoearth(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    del scene_context
    bbox = _coerce_bbox(args.get("bbox"))
    return await olmoearth_datasets.catalog_summary(bbox=bbox)


async def _tool_query_ndvi_timeseries(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    """Monthly NDVI timeseries over a bbox from Sentinel-2 L2A.

    For each of the last ``months`` months, we search the PC Sentinel-2 L2A
    archive for the least-cloudy scene intersecting the bbox, fetch the B04
    (red) and B08 (NIR) bands through the normal fetch-plus-resample
    pipeline, compute ``NDVI = (B08 - B04) / (B08 + B04)`` per pixel, and
    report summary stats (mean / median / p10 / p90) over non-zero pixels.
    """
    del scene_context
    bbox = _coerce_bbox(args.get("bbox"))
    months = max(1, min(24, int(args.get("months", 12))))
    max_size = int(args.get("max_size_px", 64))

    # STAC's eo:cloud_cover filter isn't perfectly reliable across all S2
    # tiles — a scene reported as 45 % cloud can still slip through a
    # "<40 %" filter, depending on PC's field indexing. Re-verify after the
    # fetch returns. The threshold matches the one passed to fetch_s2_composite.
    max_cc_pct = 40.0

    today = datetime.now(tz=timezone.utc).date().replace(day=1)

    # Parallelize per-month fetches. Each month is independent — a sequential
    # loop over 12 months × ~50 s cold-TLS-to-PC-STAC on Windows blew past
    # the 120 s tool timeout (observed: 147 s at concurrency=4, 126 s at 8).
    # Bumped to 12 so all months fire in one wave; 12 is the max `months`
    # the schema allows so this never balloons. PC hasn't throttled at this
    # level in our tests, and the per-tool timeout override (below) gives
    # enough margin for slow months.
    sem = asyncio.Semaphore(12)

    async def _one_month(i: int) -> dict[str, Any]:
        month_end = _subtract_months(today, i)
        month_start = _subtract_months(month_end, 1)
        range_str = f"{month_start:%Y-%m-%d}/{(month_end - timedelta(days=1)):%Y-%m-%d}"
        entry: dict[str, Any] = {"month": f"{month_start:%Y-%m}", "range": range_str}
        async with sem:
            try:
                scene = await fetch_s2_composite(
                    bbox=bbox,
                    datetime_range=range_str,
                    max_size_px=max_size,
                    max_cloud_cover=40.0,
                )
            except SentinelFetchError as e:
                entry.update({"status": "no_scene", "detail": str(e)[:200]})
                return entry
            except Exception as e:  # network, rasterio, etc.
                logger.info("NDVI fetch failed for %s: %s", range_str, e)
                entry.update({"status": "error", "detail": f"{type(e).__name__}: {e}"[:200]})
                return entry

        # Defensive cloud check — PC's STAC filter sometimes admits scenes
        # above the threshold. Drop them here so bad months surface as
        # "too_cloudy" rather than being smuggled into the ok set with
        # nonsense NDVI values.
        cc = float(scene.cloud_cover or 0.0)
        if cc > max_cc_pct:
            entry.update({
                "status": "too_cloudy",
                "scene_id": scene.scene_id,
                "scene_datetime": scene.datetime_str,
                "scene_cloud_cover": cc,
                "detail": f"scene cloud cover {cc:.1f}% exceeds {max_cc_pct:.0f}% threshold",
            })
            return entry

        # Band order is (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09)
        # → B04 is index 2, B08 is index 3.
        b04 = scene.image[:, :, 2].astype(np.float32)
        b08 = scene.image[:, :, 3].astype(np.float32)
        valid = (b04 > 0) & (b08 > 0)
        denom = b08 + b04
        ndvi = np.where(denom > 0, (b08 - b04) / denom, np.nan)
        ndvi_valid = ndvi[valid & np.isfinite(ndvi)]
        if ndvi_valid.size == 0:
            entry.update({"status": "empty", "scene_id": scene.scene_id,
                          "scene_datetime": scene.datetime_str})
            return entry
        entry.update(
            {
                "status": "ok",
                "scene_id": scene.scene_id,
                "scene_datetime": scene.datetime_str,
                "scene_cloud_cover": scene.cloud_cover,
                "ndvi_mean": float(ndvi_valid.mean()),
                "ndvi_median": float(np.median(ndvi_valid)),
                "ndvi_p10": float(np.percentile(ndvi_valid, 10)),
                "ndvi_p90": float(np.percentile(ndvi_valid, 90)),
                "n_pixels": int(ndvi_valid.size),
            }
        )
        return entry

    # Fire all months concurrently — the semaphore caps real parallelism at 4.
    # `i=0` is the month ending today; `i=months-1` is the earliest.
    gathered = await asyncio.gather(
        *[_one_month(i) for i in range(months)],
        return_exceptions=False,
    )
    # gather preserves input order, which is newest-first by our index scheme.
    # Reverse to chronological (earliest-first) for human-friendly display.
    timeseries: list[dict[str, Any]] = list(reversed(gathered))

    ok_count = sum(1 for e in timeseries if e.get("status") == "ok")
    too_cloudy = sum(1 for e in timeseries if e.get("status") == "too_cloudy")
    no_scene = sum(1 for e in timeseries if e.get("status") == "no_scene")

    # Emit a CSV artifact so the LLM can cite a download rather than paste
    # a 12-row markdown table into the chat (max_tokens truncation + not
    # reusable in Excel/pandas). All months land in the CSV, including
    # too_cloudy / no_scene rows — honest record.
    csv_lines = [
        "month,status,scene_id,scene_datetime,scene_cloud_cover,"
        "ndvi_mean,ndvi_median,ndvi_p10,ndvi_p90,n_pixels,detail",
    ]
    for e in timeseries:
        csv_lines.append(",".join(str(e.get(k, "")) for k in (
            "month", "status", "scene_id", "scene_datetime", "scene_cloud_cover",
            "ndvi_mean", "ndvi_median", "ndvi_p10", "ndvi_p90", "n_pixels", "detail",
        )))
    csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
    # Filename includes a short bbox hash so stacked downloads from
    # different sessions don't collide in the user's Downloads folder.
    import hashlib as _hashlib  # noqa: PLC0415
    bbox_tag = _hashlib.sha256(str(bbox.model_dump()).encode()).hexdigest()[:6]
    artifact = artifacts_svc.save_artifact(
        content=csv_bytes,
        filename=f"ndvi_timeseries_{months}mo_{bbox_tag}.csv",
        content_type="text/csv",
        summary=(
            f"{months}-month Sentinel-2 NDVI timeseries over bbox "
            f"({ok_count} ok, {too_cloudy} too-cloudy, {no_scene} no-scene)."
        ),
    )

    return {
        "status": "ok" if ok_count > 0 else "empty",
        "bbox": bbox.model_dump(),
        "months_requested": months,
        "months_with_data": ok_count,
        "months_too_cloudy": too_cloudy,
        "months_no_scene": no_scene,
        "timeseries": timeseries,
        "formula": "NDVI = (B08 - B04) / (B08 + B04)",
        "source": "Microsoft Planetary Computer · sentinel-2-l2a · least-cloudy-per-month",
        "max_cloud_cover_pct": max_cc_pct,
        "artifacts": [artifact],
    }


async def _tool_query_ndvi_masked(
    args: dict[str, Any], scene_context: dict[str, Any]
) -> dict[str, Any]:
    """Monthly NDVI restricted to pixels of chosen classes from a prior
    FT-segmentation run.

    Reuses the cached class_raster + bbox + CRS on the inference job so we
    don't re-run the FT model. For each month, we fetch S2 over the same
    bbox, resample the class raster to the monthly NDVI's grid, and
    compute per-month stats only over masked-in pixels.
    """
    del scene_context
    job_id = str(args.get("job_id") or "").strip()
    class_indices = args.get("class_indices") or []
    if not isinstance(class_indices, list) or not all(isinstance(i, int) for i in class_indices):
        return {"error": "bad_class_indices", "detail": "class_indices must be a list of ints"}
    if not job_id:
        return {
            "error": "missing_argument",
            "detail": "job_id is required — run run_olmoearth_inference first",
        }
    months = max(1, min(24, int(args.get("months", 12))))
    max_size = int(args.get("max_size_px", 64))

    job = olmoearth_inference.get_job(job_id)
    if job is None:
        return {
            "error": "job_not_found",
            "detail": (
                f"inference job {job_id} not in registry — was the backend "
                "restarted since run_olmoearth_inference? Re-run the "
                "inference and pass the new job_id."
            ),
        }
    if job.get("task_type") not in ("segmentation", "classification"):
        return {
            "error": "wrong_task_type",
            "detail": (
                f"job task_type={job.get('task_type')} — mask tool only "
                "supports segmentation / classification outputs."
            ),
        }
    class_raster = job.get("class_raster")
    if class_raster is None:
        return {
            "error": "no_class_raster",
            "detail": (
                "job did not produce a class_raster (may be embedding or "
                "regression). Re-run with a classification/segmentation "
                "model like FT-Mangrove or FT-AWF."
            ),
        }

    bbox = BBox(**job["spec"]["bbox"])
    class_indices_set = set(int(i) for i in class_indices)
    class_names_all = job.get("class_names") or []
    kept_names = [
        class_names_all[i] for i in class_indices if 0 <= i < len(class_names_all)
    ] or [f"class_{i}" for i in class_indices]

    today = datetime.now(tz=timezone.utc).date().replace(day=1)
    timeseries: list[dict[str, Any]] = []
    max_cc_pct = 40.0

    for i in range(months):
        month_end = _subtract_months(today, i)
        month_start = _subtract_months(month_end, 1)
        range_str = f"{month_start:%Y-%m-%d}/{(month_end - timedelta(days=1)):%Y-%m-%d}"
        entry: dict[str, Any] = {"month": f"{month_start:%Y-%m}", "range": range_str}
        try:
            scene = await fetch_s2_composite(
                bbox=bbox,
                datetime_range=range_str,
                max_size_px=max_size,
                max_cloud_cover=max_cc_pct,
            )
        except SentinelFetchError as e:
            entry.update({"status": "no_scene", "detail": str(e)[:200]})
            timeseries.append(entry)
            continue
        except Exception as e:
            logger.info("NDVI-masked fetch failed for %s: %s", range_str, e)
            entry.update({"status": "error", "detail": f"{type(e).__name__}: {e}"[:200]})
            timeseries.append(entry)
            continue

        cc = float(scene.cloud_cover or 0.0)
        if cc > max_cc_pct:
            entry.update({
                "status": "too_cloudy", "scene_id": scene.scene_id,
                "scene_datetime": scene.datetime_str, "scene_cloud_cover": cc,
                "detail": f"scene cloud cover {cc:.1f}% exceeds {max_cc_pct:.0f}% threshold",
            })
            timeseries.append(entry)
            continue

        # Nearest-neighbor resample the class raster onto the NDVI grid so
        # both share the same (H, W). Using integer index slicing — we
        # don't need sub-pixel accuracy for a mask.
        ndvi_h, ndvi_w = scene.image.shape[:2]
        cr_h, cr_w = class_raster.shape
        ys = np.clip((np.arange(ndvi_h) * cr_h / ndvi_h).astype(np.int32), 0, cr_h - 1)
        xs = np.clip((np.arange(ndvi_w) * cr_w / ndvi_w).astype(np.int32), 0, cr_w - 1)
        resampled_cr = class_raster[ys[:, None], xs[None, :]]

        b04 = scene.image[:, :, 2].astype(np.float32)
        b08 = scene.image[:, :, 3].astype(np.float32)
        denom = b08 + b04
        ndvi = np.where(denom > 0, (b08 - b04) / denom, np.nan)
        class_mask = np.isin(resampled_cr, list(class_indices_set))
        pixel_valid = class_mask & (b04 > 0) & (b08 > 0) & np.isfinite(ndvi)
        kept = ndvi[pixel_valid]

        if kept.size == 0:
            entry.update({
                "status": "no_class_pixels", "scene_id": scene.scene_id,
                "scene_datetime": scene.datetime_str,
                "detail": (
                    f"no pixels in bbox matched class_indices={sorted(class_indices_set)} "
                    "this month — class raster may not cover the same area."
                ),
            })
            timeseries.append(entry)
            continue

        entry.update({
            "status": "ok",
            "scene_id": scene.scene_id,
            "scene_datetime": scene.datetime_str,
            "scene_cloud_cover": cc,
            "ndvi_mean": float(kept.mean()),
            "ndvi_median": float(np.median(kept)),
            "ndvi_p10": float(np.percentile(kept, 10)),
            "ndvi_p90": float(np.percentile(kept, 90)),
            "n_pixels_masked_in": int(kept.size),
            "total_pixels_in_scene": int(pixel_valid.size),
        })
        timeseries.append(entry)

    timeseries.reverse()
    ok_count = sum(1 for e in timeseries if e.get("status") == "ok")

    # Emit a CSV artifact. Rows include non-ok months so the user sees why
    # some months are missing rather than getting a silently short file.
    csv_lines = [
        "month,status,scene_id,scene_datetime,scene_cloud_cover,"
        "ndvi_mean,ndvi_median,ndvi_p10,ndvi_p90,n_pixels_masked_in,detail",
    ]
    for e in timeseries:
        csv_lines.append(",".join(str(e.get(k, "")) for k in (
            "month", "status", "scene_id", "scene_datetime", "scene_cloud_cover",
            "ndvi_mean", "ndvi_median", "ndvi_p10", "ndvi_p90",
            "n_pixels_masked_in", "detail",
        )))
    csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
    artifact = artifacts_svc.save_artifact(
        content=csv_bytes,
        filename=f"ndvi_masked_{job_id[:8]}_{'-'.join(str(i) for i in sorted(class_indices_set))}.csv",
        content_type="text/csv",
        summary=(
            f"{months}-month NDVI over job {job_id[:8]} pixels in classes "
            f"{sorted(class_indices_set)} ({', '.join(kept_names)}). "
            f"{ok_count}/{months} months had data."
        ),
    )

    return {
        "status": "ok" if ok_count > 0 else "empty",
        "job_id": job_id,
        "class_indices": sorted(class_indices_set),
        "class_names_kept": kept_names,
        "months_requested": months,
        "months_with_data": ok_count,
        "timeseries": timeseries,
        "formula": "NDVI = (B08 - B04) / (B08 + B04), restricted to masked-in pixels",
        "source": "Microsoft Planetary Computer · sentinel-2-l2a + cached FT class_raster",
        "artifacts": [artifact],
    }


def _subtract_months(d: "datetime.date", n: int) -> "datetime.date":
    """Subtract ``n`` whole months from a ``date``. Clamps the day to the
    last valid day of the target month."""
    m = d.month - 1 - n  # 0-indexed
    year = d.year + m // 12
    month = (m % 12) + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


async def _tool_query_polygon_stats(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    geometry = args.get("geometry")
    if not isinstance(geometry, dict):
        polygon_id = str(args.get("polygon_id", ""))
        if not polygon_id:
            return {"error": "missing_argument", "detail": "Pass either 'polygon_id' or 'geometry'."}
        feat = _find_polygon(scene_context, polygon_id)
        if feat is None:
            features = (scene_context or {}).get("polygon_features") or []
            return {
                "found": False,
                "polygon_id": polygon_id,
                "reason": f"No polygon with that id/index in the scene ({len(features)} features loaded).",
            }
        geometry = feat.get("geometry") or {}

    include_elevation = bool(args.get("include_elevation", True))
    try:
        stats = await polygon_stats_svc.polygon_stats(
            geometry, include_elevation=include_elevation,
        )
    except ValueError as e:
        return {"error": "bad_geometry", "detail": str(e)}
    return stats


async def _tool_search_stac_imagery(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    del scene_context
    bbox = _coerce_bbox(args.get("bbox"))
    datetime_range = args.get("datetime")
    if not datetime_range:
        return {"error": "missing_argument", "detail": "`datetime` (RFC 3339 interval) is required."}
    return await stac_svc.search_stac_imagery(
        bbox=bbox,
        datetime_range=str(datetime_range),
        collections=args.get("collections"),
        max_cloud_cover=args.get("max_cloud_cover", 20.0),
        limit=int(args.get("limit", 10)),
    )


async def _tool_get_composite_tile_url(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    del scene_context
    bbox = _coerce_bbox(args.get("bbox"))
    datetime_range = args.get("datetime")
    if not datetime_range:
        return {"error": "missing_argument", "detail": "`datetime` (RFC 3339 interval) is required."}
    return await stac_svc.get_composite_tile_url(
        bbox=bbox,
        datetime_range=str(datetime_range),
        collection=args.get("collection", "sentinel-2-l2a"),
        assets=args.get("assets"),
        rescale=args.get("rescale"),
        color_formula=args.get("color_formula"),
        max_cloud_cover=args.get("max_cloud_cover", 20.0),
    )


async def _tool_run_olmoearth_inference(
    args: dict[str, Any], scene_context: dict[str, Any]
) -> dict[str, Any]:
    """Run a real OlmoEarth forward pass + return a compact summary for the LLM.

    Delegates to :func:`olmoearth_inference.start_inference` which handles
    S2 fetching, encoder/head dispatch, caching, and stub fallback on error.
    The returned dict is trimmed so the LLM isn't flooded with tensor-shaped
    metadata — the full response (including raster transforms etc.) is kept
    server-side and addressable via ``job_id``.
    """
    del scene_context
    bbox = _coerce_bbox(args.get("bbox"))
    model_repo_id = str(args.get("model_repo_id") or "").strip()
    if not model_repo_id:
        return {
            "error": "missing_argument",
            "detail": "model_repo_id is required",
            "hint": (
                "Pass an allenai/OlmoEarth-v1-* repo id. Call query_olmoearth "
                "first to see the live catalog."
            ),
        }

    date_range = args.get("date_range")
    if date_range is not None:
        date_range = str(date_range).strip() or None
    max_size_px = int(args.get("max_size_px", 256))
    sliding_window = bool(args.get("sliding_window", False))
    window_size = int(args.get("window_size", 32))

    try:
        resp = await olmoearth_inference.start_inference(
            bbox=bbox,
            model_repo_id=model_repo_id,
            date_range=date_range,
            max_size_px=max_size_px,
            sliding_window=sliding_window,
            window_size=window_size,
        )
    except Exception as e:
        logger.exception("run_olmoearth_inference failed for %s", model_repo_id)
        return {
            "error": "inference_failed",
            "detail": f"{type(e).__name__}: {e}"[:500],
            "bbox": bbox.model_dump(),
            "model_repo_id": model_repo_id,
        }

    # Trim the response to what the LLM needs to explain the result + tell
    # the UI how to render the layer. Raster transforms, CRS objects, and
    # numpy arrays stay server-side on the cached job.
    summary: dict[str, Any] = {
        "status": resp.get("status"),
        "kind": resp.get("kind"),
        "task_type": resp.get("task_type"),
        "model_repo_id": resp.get("model_repo_id"),
        "tile_url": resp.get("tile_url"),
        "job_id": resp.get("job_id"),
        "bbox": resp.get("bbox"),
    }
    if resp.get("kind") == "pytorch":
        summary.update(
            {
                "scene_id": resp.get("scene_id"),
                "scene_datetime": resp.get("scene_datetime"),
                "scene_cloud_cover": resp.get("scene_cloud_cover"),
                "patch_size": resp.get("patch_size"),
            }
        )
        task_type = resp.get("task_type")
        if task_type in ("classification", "segmentation"):
            summary.update(
                {
                    "num_classes": resp.get("num_classes"),
                    "class_names": resp.get("class_names"),
                    "class_names_tentative": resp.get("class_names_tentative"),
                    "decoder_key": resp.get("decoder_key"),
                }
            )
            if resp.get("class_probs") is not None:
                summary["class_probs"] = resp["class_probs"]
            # Legend carries per-class hex colors so the LLM can describe the
            # map legend accurately without re-fetching anything.
            legend = resp.get("legend") or {}
            if legend.get("classes"):
                summary["legend_classes"] = [
                    {"index": c["index"], "name": c["name"], "color": c["color"]}
                    for c in legend["classes"]
                ]
        elif task_type == "regression":
            summary["prediction_value"] = resp.get("prediction_value")
            summary["units"] = resp.get("units")
        elif task_type == "embedding":
            summary["embedding_dim"] = resp.get("embedding_dim")
    else:  # stub fallback
        summary["stub_reason"] = resp.get("stub_reason")
        # Pass through actionable retry suggestions. The agent loop uses
        # these to give the user concrete next steps (new date_range /
        # sliding_window=false / smaller bbox) instead of vague "want me
        # to retry?" questions.
        retries = resp.get("suggested_retries")
        if retries:
            summary["suggested_retries"] = retries

    summary["notes"] = resp.get("notes", [])
    return summary


async def _tool_get_higher_res_patch(args: dict[str, Any], scene_context: dict[str, Any]) -> dict[str, Any]:
    polygon_id = str(args.get("polygon_id", ""))
    zoom = int(args.get("zoom", 16))
    feat = _find_polygon(scene_context, polygon_id)
    return {
        "status": "not_implemented",
        "polygon_id": polygon_id,
        "zoom": zoom,
        "polygon_found": feat is not None,
        "next_step": (
            "Wire a basemap tile fetcher (Mapbox / ESRI World Imagery / "
            "MapTiler) that crops a z/x/y tile to the polygon's bbox and "
            "returns a base64 PNG. Feed that into chat_with_vision as an "
            "additional image for the next turn."
        ),
    }


async def _tool_query_raster_histogram(
    args: dict[str, Any], scene_context: dict[str, Any]
) -> dict[str, Any]:
    """Per-class pixel distribution of a completed classification /
    segmentation inference job. Lets the explainer LLM ground its
    answer in ACTUAL pixel percentages over the AOI instead of
    paraphrasing the full class catalog."""
    job_id = args.get("job_id") or scene_context.get("job_id")
    if not job_id:
        return {
            "error": "missing_job_id",
            "hint": (
                "Pass the job_id from the inference result, or set "
                "scene_context.job_id before calling the tool."
            ),
        }
    top_n = int(args.get("top_n", 20))
    # Histogram is pure numpy over an already-resident raster — no I/O,
    # no threadpool needed. Return synchronously to save an await hop.
    return olmoearth_inference.raster_class_histogram(str(job_id), top_n=top_n)


async def _tool_query_raster_scalar_stats(
    args: dict[str, Any], scene_context: dict[str, Any]
) -> dict[str, Any]:
    """Mean/min/max/quartile summary of a scalar raster (regression
    tasks like LFMC, or PCA-embedding base encoders). Use instead of
    ``query_raster_histogram`` when the task produces continuous values
    rather than discrete class labels."""
    job_id = args.get("job_id") or scene_context.get("job_id")
    if not job_id:
        return {"error": "missing_job_id"}
    return olmoearth_inference.raster_scalar_stats(str(job_id))


_EXECUTORS = {
    "query_polygon":              _tool_query_polygon,
    "query_olmoearth":            _tool_query_olmoearth,
    "run_olmoearth_inference":    _tool_run_olmoearth_inference,
    "query_ndvi_timeseries":      _tool_query_ndvi_timeseries,
    "query_ndvi_masked":          _tool_query_ndvi_masked,
    "query_polygon_stats":        _tool_query_polygon_stats,
    "search_stac_imagery":        _tool_search_stac_imagery,
    "get_composite_tile_url":     _tool_get_composite_tile_url,
    "get_higher_res_patch":       _tool_get_higher_res_patch,
    "query_raster_histogram":     _tool_query_raster_histogram,
    "query_raster_scalar_stats":  _tool_query_raster_scalar_stats,
}


async def execute_tool(
    name: str,
    arguments: str | dict[str, Any],
    scene_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch one tool call from an LLM tool_calls entry.

    ``arguments`` may be a JSON string (what OpenAI/vLLM actually emit) or
    already-parsed dict. Unknown tool names and JSON-decode errors are
    returned as structured error payloads rather than raised, so the caller
    can feed them back to the model as ``role=tool`` content and let it
    recover.
    """
    scene_context = scene_context or {}

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return {"error": "invalid_json_arguments", "detail": str(e), "raw": arguments[:500]}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        return {"error": "invalid_arguments_type", "got": type(arguments).__name__}

    executor = _EXECUTORS.get(name)
    if executor is None:
        return {"error": "unknown_tool", "name": name, "available": list(_EXECUTORS)}

    try:
        return await executor(args, scene_context)
    except Exception as e:  # surface to model so it can retry with fixed args
        logger.exception("Tool %s execution failed", name)
        return {"error": "tool_execution_failed", "name": name, "detail": str(e)}


# Default ceiling for one tool call. Real inference tools (OlmoEarth FT
# on a large bbox, STAC mosaic for a wide date range) can legitimately
# take ~60-90 s; anything past that is almost certainly hung (upstream
# timeout, stuck network, looped code). A bounded wait keeps the chat
# responsive — the LLM gets a structured ``tool_timeout`` error result
# and decides whether to retry, summarize what it has, or give up.
TOOL_EXECUTION_TIMEOUT_S = 120.0

# Per-tool overrides for tools whose honest runtime can exceed the global
# ceiling. ``query_ndvi_timeseries`` fans out up to 12 Sentinel-2 fetches
# over cold-TLS PC STAC connections which can take ~50 s per month on
# Windows; even with full parallelism the tool legitimately needs > 120 s.
# ``run_olmoearth_inference`` can hit ~60-90 s in the stub+retry path
# where one attempt fails and a second fetch + forward pass runs.
TOOL_TIMEOUT_OVERRIDES: dict[str, float] = {
    "query_ndvi_timeseries": 240.0,
    "query_ndvi_masked": 240.0,
    "run_olmoearth_inference": 240.0,
}


async def execute_tool_with_timeout(
    name: str,
    arguments: str | dict[str, Any],
    scene_context: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    # Per-tool overrides take priority over the global default; explicit
    # `timeout_s` arg (if caller supplied one) takes priority over both.
    if timeout_s is None:
        timeout_s = TOOL_TIMEOUT_OVERRIDES.get(name, TOOL_EXECUTION_TIMEOUT_S)
    """Same contract as :func:`execute_tool` but bounded by ``timeout_s``.

    On :class:`asyncio.TimeoutError`, returns a structured error payload
    so the LLM sees the failure as a normal ``role=tool`` result and can
    continue the conversation rather than the whole chat turn hanging.

    Cloud chat routers (NIM / Claude / Gemini / OpenAI) and the auto-
    label Gemma router all dispatch tools through this wrapper so the
    120 s ceiling is uniform across providers — matches the audit fix
    for "no timeout on tool execution; if a tool hangs, the whole
    chat turn hangs".
    """
    try:
        return await asyncio.wait_for(
            execute_tool(name, arguments, scene_context),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "tool %s exceeded %.1fs timeout — surfacing as tool_timeout",
            name,
            timeout_s,
        )
        return {
            "error": "tool_timeout",
            "name": name,
            "timeout_s": timeout_s,
            "detail": (
                f"Tool {name!r} did not return within {timeout_s:.0f} seconds. "
                "Most likely the upstream service (STAC, Open-Meteo, "
                "HuggingFace, or an OlmoEarth inference run) is slow or "
                "unreachable. Consider retrying with a smaller bbox, "
                "narrower date range, or a different tool."
            ),
        }
