"""Elevation + terrain-reconstruction routes.

Exposes two endpoints that both return the same ``ReconstructResponse``
envelope (``{status, terrain: ElevationResult}``). They differ only in
how the caller passes parameters:

* ``POST /api/reconstruct`` — bbox + resolution in the JSON body. Used
  by the frontend (``client.ts::getElevation``) and by analytics code
  that already has a ``BBox`` object in hand.
* ``GET /api/elevation`` — bbox + resolution as query-string params.
  Useful for quick cURL / browser-URL probes, LLM tool calls that
  serialize query params more cleanly than JSON bodies, etc.

Both routes share the same pydantic ``response_model``, so drift
between them is caught at import time rather than at the frontend
crash site.
"""
from fastapi import APIRouter, Body, Query

from app.models.schemas import BBox, ReconstructResponse
from app.services.terrain import get_elevation_grid

router = APIRouter()


class _ReconstructBody(BBox):
    """``POST /reconstruct`` body. Accepts the four bbox corners (inherited
    from ``BBox``) plus an optional ``resolution`` in meters (5 – 50).

    Previously the route accepted just a bare ``BBox`` and hardcoded
    ``resolution=20`` — that silently ignored the ``resolution`` arg the
    frontend's ``getElevation`` helper was trying to pass. Now they
    align: frontend ships resolution, backend honors it.
    """

    resolution: int = 20


@router.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct_terrain(body: _ReconstructBody = Body(...)) -> ReconstructResponse:
    """Get elevation grid for a bounding box using Open-Meteo SRTM."""
    bbox = BBox(west=body.west, south=body.south, east=body.east, north=body.north)
    # ``get_elevation_grid`` clamps ``resolution`` internally; we also
    # bounds-check here so an out-of-range body fails fast rather than
    # producing a huge grid.
    resolution = max(5, min(50, int(body.resolution)))
    grid = await get_elevation_grid(bbox, resolution=resolution)
    return ReconstructResponse(status="completed", terrain=grid)


@router.get("/elevation", response_model=ReconstructResponse)
async def get_elevation(
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
    resolution: int = Query(20, ge=5, le=50),
) -> ReconstructResponse:
    """Get elevation grid via GET for quick queries.

    Returns the same ``{status, terrain}`` envelope as ``/reconstruct``
    so frontend / LLM tool callers can switch between the two endpoints
    without reshaping the response. (Previous version returned the
    naked grid and quietly diverged — caught in the architecture audit.)
    """
    bbox = BBox(west=west, south=south, east=east, north=north)
    grid = await get_elevation_grid(bbox, resolution=resolution)
    return ReconstructResponse(status="completed", terrain=grid)
