"""POST /polygon-stats — perimeter, area, and elevation stats for a polygon."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.models.schemas import PolygonStatsResponse
from app.services import polygon_stats as polygon_stats_svc

router = APIRouter()


@router.post("/polygon-stats", response_model=PolygonStatsResponse)
async def compute_polygon_stats(
    payload: dict = Body(
        ...,
        description=(
            "{geometry: GeoJSON Polygon | MultiPolygon, "
            "include_elevation?: bool = true, resolution?: int = 20}"
        ),
    ),
) -> dict:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise HTTPException(400, "geometry (GeoJSON Polygon or MultiPolygon) is required")

    include_elevation = bool(payload.get("include_elevation", True))
    resolution = int(payload.get("resolution", 20))
    if not 5 <= resolution <= 60:
        raise HTTPException(400, "resolution must be in [5, 60]")

    try:
        return await polygon_stats_svc.polygon_stats(
            geometry,
            elevation_resolution=resolution,
            include_elevation=include_elevation,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
