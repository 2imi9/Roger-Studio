"""ESA WorldCover land-cover classification over a bbox.

ESA WorldCover is a globally consistent 10 m/pixel LULC map published by ESA
from Sentinel-1 + Sentinel-2 fusion. It replaces the latitude heuristic that
shipped in ``analyze.py`` with actual per-pixel classification data.

Collection on Microsoft Planetary Computer: ``esa-worldcover``
  - v200 (2021) and v100 (2020) releases
  - Single asset ``map`` stored as uint8 COG
  - Global coverage at ~3° tiles

Class encoding (official ESA scheme, same on PC and the original distribution):

    10  Tree cover              #006400
    20  Shrubland               #ffbb22
    30  Grassland               #ffff4c
    40  Cropland                #f096ff
    50  Built-up                #fa0000
    60  Bare / sparse vegetation #b4b4b4
    70  Snow and ice            #f0f0f0
    80  Permanent water bodies  #0064c8
    90  Herbaceous wetland      #0096a0
    95  Mangroves               #00cf75
    100 Moss and lichen         #fae6a0

The implementation streams the COG directly from the signed PC URL via
rasterio's HTTP/range-read support; no intermediate temp files.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

from app.models.schemas import BBox
from app.services.sentinel2_fetch import _http_retrying_request  # reuse the retry wrapper

logger = logging.getLogger(__name__)


PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_API = "https://planetarycomputer.microsoft.com/api/sas/v1"

# Official class metadata. Order drives how the UI legend renders.
WORLDCOVER_CLASSES: list[dict[str, Any]] = [
    {"code": 10,  "name": "Tree cover",               "color": "#006400"},
    {"code": 20,  "name": "Shrubland",                "color": "#ffbb22"},
    {"code": 30,  "name": "Grassland",                "color": "#ffff4c"},
    {"code": 40,  "name": "Cropland",                 "color": "#f096ff"},
    {"code": 50,  "name": "Built-up",                 "color": "#fa0000"},
    {"code": 60,  "name": "Bare / sparse vegetation", "color": "#b4b4b4"},
    {"code": 70,  "name": "Snow and ice",             "color": "#f0f0f0"},
    {"code": 80,  "name": "Permanent water bodies",   "color": "#0064c8"},
    {"code": 90,  "name": "Herbaceous wetland",       "color": "#0096a0"},
    {"code": 95,  "name": "Mangroves",                "color": "#00cf75"},
    {"code": 100, "name": "Moss and lichen",          "color": "#fae6a0"},
]
_CODE_TO_META: dict[int, dict[str, Any]] = {c["code"]: c for c in WORLDCOVER_CLASSES}


class WorldCoverError(RuntimeError):
    """Raised when WorldCover cannot be loaded for the bbox."""


@dataclass(frozen=True)
class WorldCoverResult:
    """Per-class pixel histogram over the requested bbox."""

    total_pixels: int
    counts: dict[int, int]      # class_code -> pixel count
    year: int                   # WorldCover release year (2020 / 2021)

    def as_percentages(self) -> list[dict[str, Any]]:
        """Normalize to {name, color, percentage} entries sorted descending."""
        if self.total_pixels == 0:
            return []
        out = []
        for code, count in self.counts.items():
            meta = _CODE_TO_META.get(code)
            if meta is None:
                continue
            out.append({
                "id": code,
                "code": code,
                "name": meta["name"],
                "color": meta["color"],
                "percentage": round(count / self.total_pixels * 100, 1),
            })
        out.sort(key=lambda x: x["percentage"], reverse=True)
        return out


# Cache one SAS token per WorldCover collection — 1 h TTL.
_sas_cache: dict[str, tuple[str, float]] = {}
_SAS_TTL_SAFETY = 300.0


async def _get_sas_token(collection: str) -> str:
    import time  # noqa: PLC0415
    cached = _sas_cache.get(collection)
    now = time.time()
    if cached and cached[1] - now > _SAS_TTL_SAFETY:
        return cached[0]
    r = await _http_retrying_request("GET", f"{PC_SAS_API}/token/{collection}")
    data = r.json()
    token = data["token"]
    expiry = data.get("msft:expiry")
    if expiry:
        try:
            from datetime import datetime  # noqa: PLC0415
            exp_ts = datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
        except ValueError:
            exp_ts = now + 3600.0
    else:
        exp_ts = now + 3600.0
    _sas_cache[collection] = (token, exp_ts)
    return token


async def _search_worldcover_items(
    bbox: BBox, year: int
) -> list[dict[str, Any]]:
    """STAC search on ``esa-worldcover`` intersecting the bbox.

    The PC ``esa-worldcover`` collection uses ``start_datetime`` on the
    2021 release and ``datetime`` on 2020. Filter by the release year.
    """
    body = {
        "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
        "collections": ["esa-worldcover"],
        "datetime": f"{year}-01-01/{year}-12-31",
        "limit": 16,
    }
    r = await _http_retrying_request("POST", f"{PC_STAC_API}/search", json=body)
    data = r.json()
    return data.get("features") or []


def _sign(href: str, token: str) -> str:
    if "?" in href:
        return f"{href}&{token}"
    return f"{href}?{token}"


def _read_worldcover_window(href: str, bbox: BBox) -> np.ndarray:
    """Windowed read of the WorldCover COG restricted to the bbox.

    WorldCover rasters are published in EPSG:4326 at 10 m/pixel, so the
    bbox coordinates can be used directly in ``rasterio.windows.from_bounds``
    without reprojection.
    """
    with rasterio.open(href) as src:
        window = window_from_bounds(
            bbox.west, bbox.south, bbox.east, bbox.north, src.transform
        )
        # Clamp the window to the raster's extent to avoid reading off-grid.
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        if window.width <= 0 or window.height <= 0:
            return np.array([], dtype=np.uint8)
        return src.read(1, window=window)


async def classify_land_cover(
    bbox: BBox, year: int = 2021
) -> WorldCoverResult:
    """Histogram WorldCover pixel values over the bbox.

    If multiple items intersect, their pixel counts are summed. Nodata (0)
    pixels are excluded from the denominator so the returned percentages
    reflect the classified area only.
    """
    features = await _search_worldcover_items(bbox, year)
    if not features:
        # Fall back to the 2020 release for areas only covered by v100.
        if year != 2020:
            logger.info("no WorldCover %d items for bbox; retrying with 2020", year)
            features = await _search_worldcover_items(bbox, 2020)
            year = 2020
    if not features:
        raise WorldCoverError(
            f"no ESA WorldCover items intersect bbox={bbox.model_dump()}"
        )

    token = await _get_sas_token("esa-worldcover")
    counts: dict[int, int] = {c["code"]: 0 for c in WORLDCOVER_CLASSES}
    total = 0
    read_errors: list[str] = []

    for feat in features:
        assets = feat.get("assets") or {}
        map_asset = assets.get("map")
        if not map_asset:
            continue
        href = _sign(map_asset["href"], token)
        try:
            arr = await asyncio.to_thread(_read_worldcover_window, href, bbox)
        except (rasterio.RasterioIOError, rasterio.errors.RasterioError, OSError) as e:
            read_errors.append(f"{feat.get('id')}: {e}")
            continue
        if arr.size == 0:
            continue
        # Histogram the known class codes only; unknown / nodata is dropped.
        flat = arr.ravel()
        valid = flat != 0
        total += int(valid.sum())
        for code in counts:
            counts[code] += int((flat == code).sum())

    if total == 0:
        raise WorldCoverError(
            f"WorldCover items found but all reads returned 0 valid pixels "
            f"(errors={read_errors})"
        )

    return WorldCoverResult(
        total_pixels=total,
        counts={k: v for k, v in counts.items() if v > 0},
        year=year,
    )
