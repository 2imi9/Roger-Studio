"""STAC imagery search + composite tile URLs via Microsoft Planetary Computer.

Why Microsoft Planetary Computer:
  - Public STAC API, no auth required for reads
  - Hosted titiler-pgstac mosaic endpoint builds cloud-free composites on the
    fly — no local xarray / stackstac compute needed
  - Covers Sentinel-2 L2A, Sentinel-1 GRD, Landsat C2 L2, NAIP, and ~100 more

Endpoints used:
  STAC search     POST  /api/stac/v1/search
  Mosaic register POST  /api/data/v1/mosaic/register
  Mosaic tiles    GET   /api/data/v1/mosaic/{searchId}/tiles/{z}/{x}/{y}
  Item tiles      GET   /api/data/v1/item/.../tilejson.json

All functions return JSON-serializable dicts so they drop directly into an
LLM tool_result, and they degrade gracefully (returning ``error`` fields)
instead of raising, so the agent can recover.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from app.models.schemas import BBox

logger = logging.getLogger(__name__)

PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1"


def _bbox_to_polygon_coords(bbox: BBox) -> list[list[list[float]]]:
    """GeoJSON Polygon coordinates for a bbox (single outer ring)."""
    return [[
        [bbox.west, bbox.south],
        [bbox.east, bbox.south],
        [bbox.east, bbox.north],
        [bbox.west, bbox.north],
        [bbox.west, bbox.south],
    ]]


async def search_stac_imagery(
    bbox: BBox,
    datetime_range: str,
    collections: list[str] | None = None,
    max_cloud_cover: float | None = 20.0,
    limit: int = 10,
) -> dict[str, Any]:
    """Search Planetary Computer STAC for scenes intersecting ``bbox``.

    Args:
        bbox: WGS84 bounding box.
        datetime_range: RFC 3339 interval — e.g. ``"2024-06-01/2024-09-01"`` or
            a single date ``"2024-06-15"``.
        collections: STAC collection ids. Default: ``["sentinel-2-l2a"]``.
        max_cloud_cover: drop scenes above this percent cloud cover
            (applies to collections that carry ``eo:cloud_cover``). Pass
            ``None`` to disable filtering.
        limit: max items returned (PC caps at 100 per page).

    Returns:
        ``{count, matched, items: [{id, collection, datetime, cloud_cover,
           bbox, assets, thumbnail_url}]}``
    """
    cols = list(collections) if collections else ["sentinel-2-l2a"]
    body: dict[str, Any] = {
        "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
        "datetime": datetime_range,
        "collections": cols,
        "limit": max(1, min(int(limit), 100)),
    }
    if max_cloud_cover is not None:
        body["query"] = {"eo:cloud_cover": {"lt": float(max_cloud_cover)}}

    try:
        async with httpx.AsyncClient(timeout=30.0, transport=httpx.AsyncHTTPTransport(retries=3)) as client:
            r = await client.post(f"{PC_STAC_API}/search", json=body)
            if r.status_code >= 400:
                logger.warning("PC STAC %s: %s", r.status_code, r.text[:400])
                return {
                    "error": "stac_search_failed",
                    "status": r.status_code,
                    "detail": r.text[:600],
                    "request_body": body,
                }
            data = r.json()
    except httpx.HTTPError as e:
        logger.exception("STAC search transport failed")
        return {
            "error": "stac_search_failed",
            "detail": f"{type(e).__name__}: {e}",
            "request_body": body,
        }

    items: list[dict[str, Any]] = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        assets = feat.get("assets") or {}
        items.append({
            "id": feat.get("id"),
            "collection": feat.get("collection"),
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "bbox": feat.get("bbox"),
            "assets": sorted(assets.keys()),
            "thumbnail_url": (assets.get("thumbnail") or assets.get("rendered_preview") or {}).get("href"),
        })

    return {
        "count": len(items),
        "matched": (data.get("context") or {}).get("matched"),
        "items": items,
    }


# Sensible defaults per collection — bands for true-color plus a rescale range
# that usually produces well-exposed tiles without per-scene tuning.
_COLLECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "sentinel-2-l2a": {
        "assets": ["B04", "B03", "B02"],
        "rescale": "0,3000",
        "color_formula": "Gamma RGB 3.5 Saturation 1.5",
    },
    "landsat-c2-l2": {
        "assets": ["red", "green", "blue"],
        "rescale": "7500,40000",
        "color_formula": "Gamma RGB 2.7 Saturation 1.4",
    },
    "naip": {
        "assets": ["image"],
        "asset_bidx": "1,2,3",
        "rescale": None,
        "color_formula": None,
    },
}


async def get_composite_tile_url(
    bbox: BBox,
    datetime_range: str,
    collection: str = "sentinel-2-l2a",
    assets: list[str] | None = None,
    rescale: str | None = None,
    color_formula: str | None = None,
    max_cloud_cover: float | None = 20.0,
) -> dict[str, Any]:
    """Register a least-cloudy mosaic on PC and return a tile URL template.

    The mosaic endpoint reduces overlapping scenes per-pixel in the order of
    the ``sortby`` clause, so sorting by ``eo:cloud_cover`` ascending picks
    the cleanest pixel available for every tile — effectively a best-scene
    composite without needing local raster math.

    Returns:
        ``{tile_url, tilejson_url, search_id, collection, assets, ...}``
        or ``{error, detail}`` on failure.
    """
    defaults = _COLLECTION_DEFAULTS.get(collection, {})
    bands = list(assets) if assets else list(defaults.get("assets") or [])
    if not bands:
        return {
            "error": "no_assets",
            "detail": f"Collection {collection!r} has no default band set; pass `assets` explicitly.",
        }
    rescale_val = rescale if rescale is not None else defaults.get("rescale")
    color_val = color_formula if color_formula is not None else defaults.get("color_formula")

    # Build a CQL2-JSON search: collection + time interval + bbox intersect +
    # optional cloud cap, sorted by cloud cover ascending so the mosaic prefers
    # clearer pixels.
    and_args: list[dict[str, Any]] = [
        {"op": "=", "args": [{"property": "collection"}, collection]},
        {"op": "anyinteracts", "args": [{"property": "datetime"}, datetime_range]},
        {
            "op": "s_intersects",
            "args": [
                {"property": "geometry"},
                {"type": "Polygon", "coordinates": _bbox_to_polygon_coords(bbox)},
            ],
        },
    ]
    if max_cloud_cover is not None and "cloud" in collection:
        and_args.append({
            "op": "<",
            "args": [{"property": "eo:cloud_cover"}, float(max_cloud_cover)],
        })

    register_body: dict[str, Any] = {
        "filter-lang": "cql2-json",
        "filter": {"op": "and", "args": and_args},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, transport=httpx.AsyncHTTPTransport(retries=3)) as client:
            r = await client.post(f"{PC_DATA_API}/mosaic/register", json=register_body)
            if r.status_code >= 400:
                logger.warning("PC mosaic register %s: %s", r.status_code, r.text[:400])
                return {
                    "error": "mosaic_register_failed",
                    "status": r.status_code,
                    "detail": r.text[:600],
                    "request_body": register_body,
                }
            data = r.json()
    except httpx.HTTPError as e:
        logger.exception("PC mosaic register transport failed")
        return {
            "error": "mosaic_register_failed",
            "detail": f"{type(e).__name__}: {e}",
            "request_body": register_body,
        }

    search_id = data.get("searchid") or data.get("id")
    if not search_id:
        return {"error": "no_search_id", "response": data}

    params: list[tuple[str, str]] = []
    for b in bands:
        params.append(("assets", b))
    if rescale_val:
        params.append(("rescale", rescale_val))
    if color_val:
        params.append(("color_formula", color_val))
    params.append(("collection", collection))

    qs = "&".join(f"{k}={quote(str(v), safe=',')}" for k, v in params)
    tile_url = f"{PC_DATA_API}/mosaic/{search_id}/tiles/{{z}}/{{x}}/{{y}}@2x?{qs}"
    tilejson_url = f"{PC_DATA_API}/mosaic/{search_id}/tilejson.json?{qs}"

    return {
        "tile_url": tile_url,
        "tilejson_url": tilejson_url,
        "search_id": search_id,
        "collection": collection,
        "assets": bands,
        "rescale": rescale_val,
        "color_formula": color_val,
        "datetime_range": datetime_range,
        "bbox": bbox.model_dump(),
        "notes": [
            "Least-cloudy scene wins per pixel (sortby eo:cloud_cover asc).",
            "The tile URL is an XYZ template ready for MapLibre addSource(type=raster).",
        ],
    }
