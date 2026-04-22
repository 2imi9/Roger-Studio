"""STAC imagery endpoints — thin wrapper around services.stac_imagery."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.models.schemas import BBox, CompositeTileResponse, StacSearchResponse
from app.services import stac_imagery as stac_svc

router = APIRouter()


@router.post("/stac/search", response_model=StacSearchResponse)
async def stac_search(payload: dict = Body(...)) -> dict:
    """Search Planetary Computer STAC. Body: {bbox, datetime, collections, max_cloud_cover, limit}."""
    try:
        bbox = BBox(**payload["bbox"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"bbox is required: {e}") from e
    datetime_range = payload.get("datetime")
    if not datetime_range:
        raise HTTPException(400, "datetime (RFC 3339 interval) is required")
    return await stac_svc.search_stac_imagery(
        bbox=bbox,
        datetime_range=datetime_range,
        collections=payload.get("collections"),
        max_cloud_cover=payload.get("max_cloud_cover", 20.0),
        limit=payload.get("limit", 10),
    )


@router.post("/stac/composite-tile-url", response_model=CompositeTileResponse)
async def stac_composite_tile_url(payload: dict = Body(...)) -> dict:
    """Register a least-cloudy mosaic and return an XYZ tile URL template."""
    try:
        bbox = BBox(**payload["bbox"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"bbox is required: {e}") from e
    datetime_range = payload.get("datetime")
    if not datetime_range:
        raise HTTPException(400, "datetime (RFC 3339 interval) is required")
    return await stac_svc.get_composite_tile_url(
        bbox=bbox,
        datetime_range=datetime_range,
        collection=payload.get("collection", "sentinel-2-l2a"),
        assets=payload.get("assets"),
        rescale=payload.get("rescale"),
        color_formula=payload.get("color_formula"),
        max_cloud_cover=payload.get("max_cloud_cover", 20.0),
    )
