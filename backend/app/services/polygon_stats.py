"""Perimeter, area, and elevation statistics for a polygon.

Matches the "Path or polygon" readout from Google Earth's measurement tool:

  - perimeter_km     haversine over consecutive vertices
  - area_km2         shoelace on an equirectangular projection at the
                     polygon's mean latitude (good to <<1% for polygons
                     smaller than a few hundred km across)
  - elevation stats  min / median / max / mean over Open-Meteo samples
                     inside the polygon (ray-casting point-in-polygon filter)

All pure-stdlib + numpy — no shapely, no pyproj. That keeps this service
importable without the optional ``[geo]`` extras.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"
_MAX_ELEVATION_BATCH = 100  # Open-Meteo per-request cap


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _ring(geometry: dict[str, Any]) -> list[list[float]]:
    """Return the outer ring of a GeoJSON Polygon or the first ring of a
    MultiPolygon. GeoJSON rings are [lon, lat] pairs; we keep that order.
    """
    gtype = (geometry or {}).get("type")
    coords = (geometry or {}).get("coordinates") or []
    if gtype == "Polygon":
        return coords[0] if coords else []
    if gtype == "MultiPolygon":
        return coords[0][0] if coords and coords[0] else []
    raise ValueError(f"Unsupported geometry type: {gtype!r} (want Polygon or MultiPolygon)")


def perimeter_km(ring: list[list[float]]) -> float:
    """Sum of haversine distances along a closed ring of [lon, lat] points."""
    if len(ring) < 2:
        return 0.0
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(ring, ring[1:], strict=False):
        total += _haversine_km(lat1, lon1, lat2, lon2)
    # Ensure closure (GeoJSON rings are already closed but tolerate both)
    if ring[0] != ring[-1]:
        lon1, lat1 = ring[-1]
        lon2, lat2 = ring[0]
        total += _haversine_km(lat1, lon1, lat2, lon2)
    return total


def area_km2(ring: list[list[float]]) -> float:
    """Geodesic-ish area via equirectangular projection at the ring's mean lat.

    Accurate to well under 1 % for polygons up to a few hundred km wide.
    For larger areas prefer pyproj's geometry_area_perimeter, but we keep
    this stdlib-only so the module imports without ``[geo]`` extras.
    """
    if len(ring) < 3:
        return 0.0
    lats = [pt[1] for pt in ring]
    mean_lat_rad = math.radians(sum(lats) / len(lats))
    cos_lat = math.cos(mean_lat_rad)
    xs = [pt[0] * 111.32 * cos_lat for pt in ring]
    ys = [pt[1] * 110.574 for pt in ring]
    # Shoelace
    n = len(ring)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test. Ring is [lon, lat] pairs."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-20) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _sample_points_inside(ring: list[list[float]], resolution: int) -> list[tuple[float, float]]:
    """Regular lat/lon grid over the ring bbox, keeping only points inside."""
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    grid_lats = np.linspace(south, north, resolution)
    grid_lons = np.linspace(west, east, resolution)

    inside: list[tuple[float, float]] = []
    for la in grid_lats:
        for lo in grid_lons:
            if _point_in_ring(float(lo), float(la), ring):
                inside.append((float(la), float(lo)))
    return inside


async def _fetch_elevations(points: list[tuple[float, float]]) -> list[float]:
    """Batch Open-Meteo elevation queries. Returns NaN for any failed batch."""
    if not points:
        return []
    out: list[float] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for i in range(0, len(points), _MAX_ELEVATION_BATCH):
            chunk = points[i : i + _MAX_ELEVATION_BATCH]
            lats = ",".join(f"{p[0]:.6f}" for p in chunk)
            lons = ",".join(f"{p[1]:.6f}" for p in chunk)
            try:
                r = await client.get(
                    OPEN_METEO_ELEVATION,
                    params={"latitude": lats, "longitude": lons},
                )
                r.raise_for_status()
                out.extend(r.json().get("elevation", [math.nan] * len(chunk)))
            except httpx.HTTPError as e:
                logger.warning("Open-Meteo elevation batch failed: %s", e)
                out.extend([math.nan] * len(chunk))
    return out


async def polygon_stats(
    geometry: dict[str, Any],
    elevation_resolution: int = 20,
    include_elevation: bool = True,
) -> dict[str, Any]:
    """Full Google-Earth-style readout for a polygon.

    Args:
        geometry: GeoJSON Polygon or MultiPolygon.
        elevation_resolution: grid density per axis inside the bbox; final
            sample count is roughly ``resolution**2 * fill_fraction``. 20 is
            a fine default for sidebars (~200-400 samples for typical shapes).
        include_elevation: skip the Open-Meteo roundtrip for fast perimeter /
            area only.

    Returns:
        ``{"perimeter_km", "area_km2", "centroid": {lat, lon}, "bbox",
           "vertex_count", "elevation": {...} | None,
           "elevation_sample_count": int}``
    """
    ring = _ring(geometry)
    if len(ring) < 3:
        return {
            "error": "degenerate_polygon",
            "vertex_count": len(ring),
        }

    p_km = perimeter_km(ring)
    a_km2 = area_km2(ring)

    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    bbox = {"west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats)}
    centroid = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}

    elevation_block: dict[str, Any] | None = None
    sample_count = 0
    if include_elevation:
        samples = _sample_points_inside(ring, elevation_resolution)
        sample_count = len(samples)
        if samples:
            elevs = await _fetch_elevations(samples)
            clean = [e for e in elevs if isinstance(e, (int, float)) and not math.isnan(e)]
            if clean:
                arr = np.asarray(clean, dtype=float)
                elevation_block = {
                    "min_m":    float(arr.min()),
                    "max_m":    float(arr.max()),
                    "mean_m":   float(arr.mean()),
                    "median_m": float(np.median(arr)),
                    "range_m":  float(arr.max() - arr.min()),
                    "source":   "open-meteo",
                }

    return {
        "perimeter_km":            round(p_km, 4),
        "area_km2":                round(a_km2, 4),
        "centroid":                centroid,
        "bbox":                    bbox,
        "vertex_count":            len(ring),
        "elevation":               elevation_block,
        "elevation_sample_count":  sample_count,
    }
