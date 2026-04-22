"""Fetch a Sentinel-2 L2A multi-band tensor over a bbox via Microsoft
Planetary Computer, shaped for direct feed into the OlmoEarth encoder.

Why this file exists:
  The OlmoEarth ``Inference-Quickstart.md`` uses a local ``.SAFE`` folder and
  ``glob`` to find the 12 bands. For a web app that runs inference on
  arbitrary user-drawn bboxes we instead search STAC for the least-cloudy
  scene, sign each asset href with a Planetary Computer SAS token, and
  stream the bands directly through rasterio + WarpedVRT to a common 10 m
  grid. The output tensor already satisfies:

      shape:      (H, W, 12) float32 reflectance (DN * 1.0)
      band order: Modality.SENTINEL2_L2A.band_order
                  (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09)
      crs:        the scene's native UTM zone
      transform:  10 m/pixel, origin at the bbox's northwest corner

The caller then reshapes to ``(1, H, W, 1, 12)`` and feeds it to
``olmoearth_model.run_s2_inference`` — no further projection tricks.

References:
  - OlmoEarth band order: ``olmoearth_pretrain/data/constants.py::Modality``
  - PC STAC search: https://planetarycomputer.microsoft.com/api/stac/v1
  - PC SAS tokens: https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import asyncio

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from olmoearth_pretrain.data.constants import Modality

from app.models.schemas import BBox

logger = logging.getLogger(__name__)

PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_API = "https://planetarycomputer.microsoft.com/api/sas/v1"

# Windows + httpx async occasionally raises ConnectError on the first call in a
# fresh loop. Retry with exponential backoff on top of the transport retries.
_HTTP_MAX_ATTEMPTS = 4
_HTTP_BACKOFF_SEC = 1.5

# Cache SAS tokens per collection. PC issues them with a ~1 hour TTL so we
# refresh a few minutes before expiry to avoid mid-fetch 403s.
_sas_cache: dict[str, tuple[str, float]] = {}
_SAS_SAFETY_MARGIN_SEC = 300.0


class SentinelFetchError(RuntimeError):
    """Raised when no usable scene was found or all bands failed to download."""


@dataclass(frozen=True)
class SentinelScene:
    """Result of a successful Sentinel-2 fetch. ``image`` is HWC in the
    OlmoEarth band order."""

    image: np.ndarray                 # (H, W, 12) float32, raw DN reflectance
    transform: rasterio.Affine        # aligned to 10 m/pixel in ``crs``
    crs: Any                          # rasterio CRS object
    scene_id: str
    datetime_str: str                 # ISO-8601 of the chosen scene
    cloud_cover: float | None
    bbox_wgs84: tuple[float, float, float, float]  # (west, south, east, north)


async def fetch_s2_composite(
    bbox: BBox,
    datetime_range: str = "2024-04-01/2024-10-01",
    max_size_px: int = 256,
    max_cloud_cover: float = 40.0,
) -> SentinelScene:
    """Fetch and align the 12-band Sentinel-2 L2A tensor for a bbox.

    Args:
        bbox: WGS-84 bounding box the user drew.
        datetime_range: RFC-3339 interval (e.g. ``"2024-06-01/2024-09-01"``)
            or a single date. Choose a summer range for temperate regions to
            minimize cloud cover.
        max_size_px: hard cap on the output tensor's longer side. 256 matches
            OlmoEarth's pretraining tile (2.56 km at 10 m/pixel) and caps the
            CPU forward pass at a reasonable latency. Raise for more detail,
            at the cost of GPU/CPU time and download size.
        max_cloud_cover: drop scenes above this percent cloud cover.

    Raises:
        SentinelFetchError: if no scene was found or all bands failed to read.
    """
    scene = await _search_least_cloudy(bbox, datetime_range, max_cloud_cover)
    collection = scene["collection"]
    token = await _get_sas_token(collection)

    # Compute a destination transform & grid: 10 m/pixel in the scene's UTM CRS,
    # aligned to the bbox. We read the first band once to learn the scene CRS.
    assets = scene["assets"]
    want_bands = list(Modality.SENTINEL2_L2A.band_order)
    missing = [b for b in want_bands if b not in assets]
    if missing:
        raise SentinelFetchError(f"scene missing bands: {missing}")

    first_href = _sign(assets[want_bands[0]]["href"], token)
    with rasterio.open(first_href) as src0:
        scene_crs = src0.crs
    # Project the WGS-84 bbox into the scene CRS to choose a destination window.
    west, south, east, north = transform_bounds(
        "EPSG:4326", scene_crs, bbox.west, bbox.south, bbox.east, bbox.north
    )
    width_m = abs(east - west)
    height_m = abs(north - south)
    # Pick a pixel spacing that keeps the longer side ≤ max_size_px. Clamp
    # to 10 m/pixel at minimum — Sentinel-2's native 10 m band resolution.
    native_gsd_m = 10.0
    longest_m = max(width_m, height_m)
    candidate_gsd = longest_m / float(max_size_px)
    gsd_m = max(native_gsd_m, candidate_gsd)
    out_width = max(1, int(round(width_m / gsd_m)))
    out_height = max(1, int(round(height_m / gsd_m)))
    dst_transform = from_bounds(west, south, east, north, out_width, out_height)

    # Read each band with WarpedVRT to the (scene_crs, dst_transform) grid.
    image = np.zeros((out_height, out_width, len(want_bands)), dtype=np.float32)
    read_failures: list[tuple[str, str]] = []
    for band_idx, band_name in enumerate(want_bands):
        href = _sign(assets[band_name]["href"], token)
        try:
            with rasterio.open(href) as src:
                with WarpedVRT(
                    src,
                    crs=scene_crs,
                    transform=dst_transform,
                    width=out_width,
                    height=out_height,
                    resampling=Resampling.bilinear,
                ) as vrt:
                    image[:, :, band_idx] = vrt.read(1).astype(np.float32)
        except (rasterio.RasterioIOError, rasterio.errors.RasterioError) as e:
            logger.warning("band %s read failed: %s", band_name, e)
            read_failures.append((band_name, str(e)[:200]))

    # Scientific-accuracy guard: if even ONE band fails, the scene is
    # unusable. OlmoEarth is a multi-spectral encoder — dropping (e.g.)
    # B08 NIR while keeping the visible bands and zero-filling the slot
    # produces a spectrally-corrupt input that the encoder will happily
    # process and emit confident-looking but meaningless predictions for.
    #
    # Previously this branch only raised when ALL 12 bands failed; any
    # partial failure silently passed zeros through to inference, and
    # the user saw a plausible-looking classification tile that was
    # actually noise. That's the single worst class of bug in the
    # audit (silent corruption with confident output).
    #
    # Upstream callers (``olmoearth_inference._run_real_inference``)
    # catch ``SentinelFetchError`` and fall back to the stub renderer,
    # which the frontend now badges with "stub fallback" so the user
    # knows the result is synthetic.
    if read_failures:
        failed_names = ", ".join(name for name, _ in read_failures)
        raise SentinelFetchError(
            f"{len(read_failures)}/{len(want_bands)} bands failed to read "
            f"for scene {scene['id']} ({failed_names}) — rejecting scene "
            f"rather than feeding zero-filled bands to the encoder. "
            f"Details: {read_failures}"
        )

    return SentinelScene(
        image=image,
        transform=dst_transform,
        crs=scene_crs,
        scene_id=scene["id"],
        datetime_str=scene["datetime"],
        cloud_cover=scene.get("cloud_cover"),
        bbox_wgs84=(bbox.west, bbox.south, bbox.east, bbox.north),
    )


def image_to_bhwtc(image_hwc: np.ndarray) -> np.ndarray:
    """Reshape a ``(H, W, 12)`` scene to ``(1, H, W, 1, 12)`` BHWTC layout."""
    if image_hwc.ndim != 3 or image_hwc.shape[-1] != 12:
        raise ValueError(f"expected (H, W, 12) got {image_hwc.shape}")
    return image_hwc[np.newaxis, :, :, np.newaxis, :].astype(np.float32)


def timestamp_from_iso(iso_str: str) -> tuple[int, int, int]:
    """Convert an ISO datetime like ``"2024-08-22T10:12:03Z"`` to OlmoEarth's
    ``(day 1-31, month 0-11, year)`` tuple."""
    # Accept plain date strings too.
    date_part = iso_str.split("T", 1)[0]
    y, m, d = date_part.split("-")
    return (int(d), int(m) - 1, int(y))


# ---------------------------------------------------------------------------
# Internals — STAC search + PC SAS token signing.
# ---------------------------------------------------------------------------


async def _http_retrying_request(
    method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """httpx request with exponential-backoff retries on ConnectError /
    ReadTimeout. Layered on top of the transport's own retry budget because
    Windows sometimes burns through both on a cold first connection."""
    last_exc: Exception | None = None
    for attempt in range(_HTTP_MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=30.0, transport=httpx.AsyncHTTPTransport(retries=3)
            ) as client:
                r = await client.request(method, url, **kwargs)
                r.raise_for_status()
                return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < _HTTP_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_HTTP_BACKOFF_SEC * (2 ** attempt))
                logger.info("retrying %s %s after %s", method, url, type(e).__name__)
            else:
                raise
    assert last_exc is not None  # pragma: no cover
    raise last_exc


async def _search_least_cloudy(
    bbox: BBox, datetime_range: str, max_cloud_cover: float
) -> dict[str, Any]:
    body = {
        "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
        "datetime": datetime_range,
        "collections": ["sentinel-2-l2a"],
        "query": {"eo:cloud_cover": {"lt": float(max_cloud_cover)}},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
        "limit": 5,
    }
    r = await _http_retrying_request("POST", f"{PC_STAC_API}/search", json=body)
    data = r.json()

    features = data.get("features") or []
    if not features:
        raise SentinelFetchError(
            f"no Sentinel-2 scenes found for bbox={bbox.model_dump()} "
            f"range={datetime_range} cloud<{max_cloud_cover}"
        )
    feat = features[0]
    props = feat.get("properties") or {}
    return {
        "id": feat["id"],
        "collection": feat["collection"],
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "assets": feat.get("assets") or {},
    }


async def _get_sas_token(collection: str) -> str:
    cached = _sas_cache.get(collection)
    now = time.time()
    if cached and cached[1] - now > _SAS_SAFETY_MARGIN_SEC:
        return cached[0]
    r = await _http_retrying_request("GET", f"{PC_SAS_API}/token/{collection}")
    data = r.json()
    token = data["token"]
    # The SAS token URL's own expiry field. Treat missing as a short TTL.
    expiry = data.get("msft:expiry")
    if expiry:
        try:
            from datetime import datetime, timezone  # noqa: PLC0415
            exp_ts = datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
        except ValueError:
            exp_ts = now + 3600.0
    else:
        exp_ts = now + 3600.0
    _sas_cache[collection] = (token, exp_ts)
    return token


def _sign(href: str, token: str) -> str:
    """Append a SAS token to an https asset URL — PC's standard signing."""
    if "?" in href:
        return f"{href}&{token}"
    return f"{href}?{token}"
