"""Terrain / DEM elevation service using Open-Meteo Elevation API."""

from __future__ import annotations

import httpx
import numpy as np

from app.models.schemas import BBox

OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"


async def get_elevation_grid(
    bbox: BBox,
    resolution: int = 20,
) -> dict:
    """Sample elevation across a bbox grid.

    Args:
        bbox: Bounding box
        resolution: Grid points per axis (default 20x20 = 400 API calls batched)

    Returns:
        dict with lats, lons, elevations (2D grid), and stats
    """
    lats = np.linspace(bbox.south, bbox.north, resolution).tolist()
    lons = np.linspace(bbox.west, bbox.east, resolution).tolist()

    # Build flat lat/lon arrays for batch query
    flat_lats = []
    flat_lons = []
    for lat in lats:
        for lon in lons:
            flat_lats.append(round(lat, 6))
            flat_lons.append(round(lon, 6))

    # Open-Meteo accepts comma-separated lists (up to ~1000 points)
    elevations_flat: list[float] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Batch in chunks of 100 (API limit)
        chunk_size = 100
        for i in range(0, len(flat_lats), chunk_size):
            chunk_lats = flat_lats[i : i + chunk_size]
            chunk_lons = flat_lons[i : i + chunk_size]

            try:
                resp = await client.get(
                    OPEN_METEO_ELEVATION,
                    params={
                        "latitude": ",".join(str(v) for v in chunk_lats),
                        "longitude": ",".join(str(v) for v in chunk_lons),
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    elevations_flat.extend(data.get("elevation", [0] * len(chunk_lats)))
                else:
                    elevations_flat.extend([0] * len(chunk_lats))
            except httpx.RequestError:
                elevations_flat.extend([0] * len(chunk_lats))

    # Reshape to 2D grid
    grid = []
    idx = 0
    for _lat in lats:
        row = []
        for _lon in lons:
            row.append(elevations_flat[idx] if idx < len(elevations_flat) else 0)
            idx += 1
        grid.append(row)

    elev_arr = [v for v in elevations_flat if v is not None]
    return {
        "lats": lats,
        "lons": lons,
        "elevations": grid,
        "stats": {
            "min": min(elev_arr) if elev_arr else 0,
            "max": max(elev_arr) if elev_arr else 0,
            "mean": sum(elev_arr) / len(elev_arr) if elev_arr else 0,
            "range": (max(elev_arr) - min(elev_arr)) if elev_arr else 0,
        },
        "resolution": resolution,
        "bbox": bbox.model_dump(),
    }
