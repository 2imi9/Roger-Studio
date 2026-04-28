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

# ---------------------------------------------------------------------------
# GDAL / CURL tuning for Planetary Computer COG reads. MUST be set BEFORE
# ``import rasterio`` — rasterio binds to the GDAL C library which reads
# several of these env vars (notably GDAL_HTTP_TIMEOUT and friends) at
# import time, not per-request. The previous revision set these AFTER
# the rasterio import and the timeouts never actually fired; the
# symptom was "every chunk stuck at 300 s with no progress" even though
# the config dict literally contained GDAL_HTTP_TIMEOUT=30.
#
# The settings themselves:
#   * HTTP/2 + 10 MB chunks matches Microsoft's COG layout for ~2× fetch
#   * Hard per-request timeout (30 s total, 10 s TCP connect, drop if
#     throughput < 1 KB/s for 15 s) — these are the ones that cannot be
#     applied post-hoc and caused the stuck-forever bug
#   * VSI cache (64 MB) + GDAL block cache (512 MB) absorb re-reads across
#     overlapping chunk windows on the same scene
#
# Later code ALSO wraps reads in ``rasterio.Env(**_GDAL_DEFAULTS)`` as
# belt-and-suspenders in case a fresh rasterio install caches some
# knobs differently across versions.
import os  # noqa: E402

_GDAL_DEFAULTS = {
    "GDAL_HTTP_MULTIPLEX": "YES",                       # HTTP/2 multiplex
    "GDAL_HTTP_VERSION": "2",                            # force HTTP/2
    "VSI_CURL_USE_HEAD": "NO",                           # skip the wasted HEAD
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.jp2,.TIF,.JP2",
    "GDAL_CACHEMAX": "512",                              # MB of in-RAM cache
    "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",              # 10 MB read chunks
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",         # don't list dirs
    "GDAL_HTTP_TIMEOUT": "15",                           # total request timeout (s)
    "GDAL_HTTP_CONNECTTIMEOUT": "5",                     # TCP connect timeout (s)
    "GDAL_HTTP_LOW_SPEED_TIME": "10",                    # abort if throughput < below
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1000",                 # ...1 KB/s for 10 s
    "CPL_VSIL_CURL_USE_S3_SIGNING_V4": "NO",             # harmless for PC; avoid probes
    "CPL_VSIL_CURL_NON_CACHED": "NO",                    # re-use curl handles across reads
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",                        # 64 MB per-process VSI cache
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",              # read more on open to skip subsequent HEADs
    # --- Retry flaky HTTP reads. Diagnosed live: ad-hoc ``curl`` against
    # the same PC endpoint sometimes succeeds (HTTP 200 in 1 s) and
    # sometimes fails with "Connection was reset" — intermittent TLS
    # handshake drops somewhere between the residential ISP and Azure.
    #
    # Cap GDAL retries LOW (1, not 5). Rationale: with 12 bands × 4 chunks
    # competing for a 20-thread asyncio pool, a single band burning its
    # full GDAL budget (5 retries × 30 s = 150 s) starves siblings and
    # blows past the 180 s per-chunk timeout — observed ALL 4 chunks
    # timing out simultaneously in dev. The application-level
    # ``failed_pass1`` retry inside ``_fetch_one_period`` re-attempts
    # failed bands once after the parallel pass returns, so we don't
    # need GDAL to retry hard. Net behaviour: 2 attempts per band (1
    # GDAL retry + 1 application-level), each fail-fast at 15 s — total
    # worst case ~30 s per band instead of 150+ s.
    "GDAL_HTTP_MAX_RETRY": "1",                          # one retry, fail fast
    "GDAL_HTTP_RETRY_DELAY": "1",                        # 1 s before retry
    "CPL_VSIL_CURL_MAX_RETRY": "1",                      # same for VSI layer
    "CPL_VSIL_CURL_RETRY_DELAY": "1",
}
for _k, _v in _GDAL_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


def _typed_gdal_options() -> dict[str, Any]:
    """``rasterio.Env(**kwargs)`` passes options to GDAL's C layer via
    ``set_gdal_config`` which enforces Python-native types: integers as
    ``int``, booleans as ``bool``, strings as ``str``. Passing a numeric
    string (``"512"``) raises ``TypeError: an integer is required`` and
    — because ``_read_one_band_window`` catches Exception broadly to
    survive stuck sockets — would silently fail EVERY band read with
    zero-filled output. Convert digit-only strings to int before handing
    to rasterio.
    """
    typed: dict[str, Any] = {}
    for k, v in _GDAL_DEFAULTS.items():
        if isinstance(v, str) and v.lstrip("-").isdigit():
            typed[k] = int(v)
        else:
            typed[k] = v
    return typed


_RASTERIO_ENV_KWARGS = _typed_gdal_options()

# Now safe to import rasterio — env vars are already in place.
import hashlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import asyncio  # noqa: E402

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_bounds  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import from_bounds as window_from_bounds  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402

from app.models.schemas import BBox  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global band-read concurrency cap.
#
# Period-parallel + chunk-parallel multiplies: 4 chunks × 12 bands × 6 periods
# = 288 potential concurrent rasterio.open calls to Planetary Computer.
# PC rate-limits aggressively at that fan-out — observed symptom was all
# chunks timing out at ~180 s because individual band reads were stuck in
# throttle queues on PC's side.
#
# 24 is empirically a safe ceiling for residential IPs: enough parallelism
# to saturate typical home bandwidth (~100 Mbps ÷ ~4 MB per read ≈ 25
# concurrent reads keep the pipe full) without tripping per-IP rate limits.
# Can be tuned via env var for power users on fatter pipes.
_BAND_READ_CONCURRENCY = int(os.environ.get("S2_BAND_READ_CONCURRENCY", "24"))
_band_read_sem: asyncio.Semaphore | None = None


def _get_band_read_sem() -> asyncio.Semaphore:
    """Lazily build the semaphore so it binds to the running event loop.

    Creating a semaphore at import time can bind it to an event loop
    that's different from the FastAPI worker's, causing subtle "Task got
    Future attached to a different loop" errors. Deferring until first
    use on the hot path sidesteps that."""
    global _band_read_sem
    if _band_read_sem is None:
        _band_read_sem = asyncio.Semaphore(_BAND_READ_CONCURRENCY)
    return _band_read_sem


# ---------------------------------------------------------------------------
# Local on-disk scene cache.
#
# First run over a bbox pays the full PC fetch cost (~60 s for a medium AOI
# over home internet). Every subsequent inference over the SAME bbox (any
# FT head, any date range that hits the same scenes) reads from SSD and
# skips PC entirely — the encoder + head then run in ~1 s on GPU.
#
# Cache key: sha256 of (scene_id, chunk_bbox with 6 decimal places, target
# gsd, band name). S2 scenes on PC are immutable once published, so a cache
# keyed on scene_id is valid forever (modulo cache-format version).
#
# Disk format: numpy .npy per (scene, band, window). One file ~1–5 MB for a
# typical 5 km chunk at 10 m/pixel. A region-wide cache for a 22 km AOI
# across 6 periods and 12 bands is ~700 MB — trivial on any modern SSD.
#
# To disable caching (e.g. to force re-fetch for debugging): set env var
# ``S2_CACHE_DISABLED=1`` OR delete the cache directory.
_S2_CACHE_DIR = Path(os.environ.get("S2_CACHE_DIR", "data/s2_cache"))
_S2_CACHE_VERSION = "v1"        # bump when cache format changes
_S2_CACHE_DISABLED = os.environ.get("S2_CACHE_DISABLED", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Sequential-read safety mode.
#
# When set via ``S2_SEQUENTIAL=1`` in the env, band reads inside a period
# run one-at-a-time via a plain ``for`` loop instead of ``asyncio.gather``.
# Slower but memory-bounded regardless of AOI size / chunk count. Matches
# the Ai2 OlmoEarth tutorial notebook's pattern, which extracts
# 1,460 windows sequentially without ever risking OOM.
#
# When to enable:
#   * Residential / shared machine where other apps (Chrome, Docker,
#     LLM containers) already use most of the RAM
#   * Anywhere that a hard-crash costs more than the time lost to serial IO
#   * Debugging / CI runs where determinism matters more than speed
#
# Worst-case memory footprint of one serial read: one GDAL block (~16 MB
# for PC's S2 L2A 10 m bands) + one per-band numpy array at chunk
# resolution (~1 MB for a 500×500 chunk). Hard ceiling: tens of MB per
# band regardless of concurrency.
_S2_SEQUENTIAL = os.environ.get("S2_SEQUENTIAL", "").lower() in ("1", "true", "yes")
if _S2_SEQUENTIAL:
    logger.info(
        "S2_SEQUENTIAL=1 — band reads will run one-at-a-time "
        "(safer, slower, OOM-proof).",
    )


def _s2_cache_key(
    scene_id: str, bbox: BBox, band: str, target_gsd_m: float
) -> Path:
    """Stable on-disk path for one (scene, bbox, band, gsd) tuple."""
    key = (
        f"{_S2_CACHE_VERSION}|{scene_id}|"
        f"{bbox.west:.6f}|{bbox.south:.6f}|{bbox.east:.6f}|{bbox.north:.6f}|"
        f"{target_gsd_m}|{band}"
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    # Shard by first 2 chars to avoid 10k+ files in a single dir — some
    # filesystems (Windows NTFS especially) degrade past a few thousand.
    return _S2_CACHE_DIR / h[:2] / f"{h}.npy"


def _s2_cache_get(
    scene_id: str, bbox: BBox, band: str, target_gsd_m: float
) -> np.ndarray | None:
    """Read a cached band array. Returns None on miss or corruption."""
    if _S2_CACHE_DISABLED:
        return None
    p = _s2_cache_key(scene_id, bbox, band, target_gsd_m)
    if not p.exists():
        return None
    try:
        arr = np.load(p)
        return arr
    except Exception as e:
        # Corrupt file (truncated write from a prior crash, disk error).
        # Drop it so the next put rewrites cleanly.
        logger.warning("s2_cache: corrupt %s (%s) — removing", p, e)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _s2_cache_put(
    scene_id: str, bbox: BBox, band: str, target_gsd_m: float, arr: np.ndarray
) -> None:
    """Write a band array to cache atomically (tmp + rename).

    Writes via an explicit file handle to bypass numpy's "helpful" behavior
    of appending ``.npy`` when called with a string / Path — that quirk
    silently broke the atomic-rename path (save to ``X.tmp.npy`` but look
    for ``X.tmp`` when renaming).
    """
    if _S2_CACHE_DISABLED:
        return
    p = _s2_cache_key(scene_id, bbox, band, target_gsd_m)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f"{p.name}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr.astype(np.float32, copy=False))
        # ``replace`` is atomic on POSIX + Windows — prevents a half-written
        # file being left behind if the process dies mid-save.
        tmp.replace(p)
    except Exception as e:
        logger.warning("s2_cache: failed to write %s: %s", p, e)

PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_API = "https://planetarycomputer.microsoft.com/api/sas/v1"

# Element84 hosts the same Sentinel-2 L2A collection on AWS (sentinel-cogs
# bucket, requester-pays-but-public). Used as a fallback when PC's /search
# endpoint flakes — different infra, different reliability profile, no SAS
# signing required (asset hrefs are public HTTPS S3 URLs).
E84_STAC_API = "https://earth-search.aws.element84.com/v1"

# Map PC's official S2 band names → Element84's friendly aliases for the
# same bands. Element84's STAC catalog uses common-name aliases instead of
# the official ESA "B0X" identifiers, so a translation layer is needed for
# the inference pipeline (which keys off ``Modality.SENTINEL2_L2A.band_order``
# = the official "B0X" names).
_E84_BAND_NAME_MAP = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}

# Windows + httpx async occasionally raises ConnectError on the first call in a
# fresh loop. Retry with exponential backoff on top of the transport retries.
#
# Bumped from 4 → 7 attempts after observing PC's STAC /search at ~50 % TLS
# reset rate from a residential IP. With 4 attempts × 50 % flake, P(all
# fail) = 6 %; with 12 periods that's ~50 % of requests with at least one
# unrecoverable period skip. With 7 attempts P(all fail) drops to 0.8 %,
# making single-period skips rare even on a bad PC window. The cost is
# tail latency: worst-case wait grows from 1.5×(1+2+4+8) = 22 s to
# 1.5×(1+2+4+8+16+32+64) = 190 s — but this only fires when PC is
# struggling, and is bounded by the per-chunk timeout above.
_HTTP_MAX_ATTEMPTS = int(os.environ.get("OE_STAC_HTTP_MAX_ATTEMPTS", "7"))
_HTTP_BACKOFF_SEC = float(os.environ.get("OE_STAC_HTTP_BACKOFF_SEC", "1.5"))

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
    # Route through the same fallback wrapper as the AOI-period path so
    # OE_S2_PROVIDER=element84 takes effect on the legacy single-scene
    # path too. Default (no override) preserves the prior behavior:
    # PC primary, E84 fallback on PC failure.
    scene = await _search_with_fallback(bbox, datetime_range, max_cloud_cover)
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


def stack_to_bhwtc(stack_hwtc: np.ndarray) -> np.ndarray:
    """Reshape a ``(H, W, T, 12)`` temporal stack to ``(1, H, W, T, 12)`` BHWTC."""
    if stack_hwtc.ndim != 4 or stack_hwtc.shape[-1] != 12:
        raise ValueError(f"expected (H, W, T, 12) got {stack_hwtc.shape}")
    return stack_hwtc[np.newaxis, ...].astype(np.float32)


def timestamp_from_iso(iso_str: str) -> tuple[int, int, int]:
    """Convert an ISO datetime like ``"2024-08-22T10:12:03Z"`` to OlmoEarth's
    ``(day 1-31, month 0-11, year)`` tuple."""
    # Accept plain date strings too.
    date_part = iso_str.split("T", 1)[0]
    y, m, d = date_part.split("-")
    return (int(d), int(m) - 1, int(y))


@dataclass(frozen=True)
class SentinelTemporalStack:
    """Result of a successful multi-period S2 fetch — what FT heads trained on
    PER_PERIOD_MOSAIC layers (Ecosystem / AWF / Mangrove) actually need.

    Layout mirrors :class:`SentinelScene` but with an explicit time axis:

      ``stack``       — ``(H, W, T, 12)`` float32, raw DN reflectance, in
                        OlmoEarth's S2 band order. Periods that returned no
                        usable scene are zero-filled and marked in
                        ``period_skipped``.
      ``transform``   — common 10 m/pixel affine in ``crs`` for every period
                        (we choose the first non-empty period's CRS and align
                        every other period to its grid via WarpedVRT).
      ``timestamps``  — ``T``-long list of (day, month-0-indexed, year) tuples;
                        one per period mosaic, taken from the chosen scene's
                        acquisition date. Periods with no scene fall back to
                        the period midpoint so the encoder never sees a zero
                        timestamp.
      ``scene_ids``   — ``T``-long list, ``None`` for skipped periods.
      ``period_skipped`` — bool list of length T; True where the period had
                        no scene matching the cloud / coverage criteria.
    """

    stack: np.ndarray                                 # (H, W, T, 12) float32
    transform: rasterio.Affine
    crs: Any
    timestamps: list[tuple[int, int, int]]            # length T
    scene_ids: list[str | None]                       # length T
    scene_datetimes: list[str | None]                 # length T (ISO-8601)
    cloud_covers: list[float | None]                  # length T
    period_skipped: list[bool]                        # length T
    bbox_wgs84: tuple[float, float, float, float]


async def fetch_s2_temporal_stack(
    bbox: BBox,
    anchor_date: str,
    n_periods: int,
    period_days: int,
    time_offset_days: int = 0,
    max_size_px: int = 256,
    max_cloud_cover: float = 40.0,
) -> SentinelTemporalStack:
    """Fetch ``n_periods`` Sentinel-2 mosaics ending at ``anchor_date``.

    Each period is a ``period_days``-wide window; we pick the single least-
    cloudy scene per window via STAC, then read all 12 bands aligned to a
    common 10 m/pixel grid.

    Period layout (walking backward from the anchor; matches what
    PER_PERIOD_MOSAIC layers in olmoearth_projects expect at inference):

        period[T-1]: [anchor + offset - period_days,
                      anchor + offset]
        period[T-2]: [anchor + offset - 2 * period_days,
                      anchor + offset - period_days]
        ...
        period[0]:   [anchor + offset - T * period_days,
                      anchor + offset - (T - 1) * period_days]

    ``time_offset_days`` lets callers shift the entire window — the rslearn
    training configs declare it relative to the label date; at inference we
    treat the user's anchor as the label proxy.

    Periods that return no scene are kept as zero slots with
    ``period_skipped[t] = True`` so the temporal axis stays length T (the
    encoder's positional / temporal encoding assumes a contiguous T axis;
    silently dropping a period would shift every later mosaic by 30 days
    relative to training). Caller may decide to fall back to single-scene
    fetch if too many periods skip.

    Raises:
        SentinelFetchError: if every period skipped or all bands failed.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >=1, got {n_periods}")
    if period_days < 1:
        raise ValueError(f"period_days must be >=1, got {period_days}")

    from datetime import datetime, timedelta  # noqa: PLC0415

    # Parse anchor_date — accept ISO date or "start/end" range (use end).
    anchor_str = anchor_date.split("/")[-1].strip().split("T", 1)[0]
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except ValueError as e:
        raise ValueError(
            f"anchor_date {anchor_date!r} not parseable as ISO date "
            f"(expected YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD)"
        ) from e
    window_end = anchor + timedelta(days=time_offset_days)

    # Build T contiguous period windows ending at window_end (oldest first).
    period_ranges: list[tuple[datetime, datetime]] = []
    for t in range(n_periods):
        idx_from_end = n_periods - 1 - t  # 0 = most recent
        end_t = window_end - timedelta(days=idx_from_end * period_days)
        start_t = end_t - timedelta(days=period_days)
        period_ranges.append((start_t, end_t))

    # Parallel STAC searches — each call is a single HTTP POST that PC
    # handles well at this concurrency (n_periods <= 12 in practice). Band
    # READS are still sequential because 12 × n_periods concurrent rasterio
    # reads exhaust PC's SAS token rate limits + local file handles. If
    # fetch time becomes the bottleneck again, parallelizing *within-period*
    # band reads (with a bounded semaphore) is the next lever.
    async def _safe_search(dt_range: str) -> dict[str, Any] | None:
        try:
            # Same _search_with_fallback wrapper used by the AOI-period
            # path so OE_S2_PROVIDER=element84 takes effect here too.
            return await _search_with_fallback(bbox, dt_range, max_cloud_cover)
        except SentinelFetchError as e:
            logger.info("temporal_stack: period %s empty (%s)", dt_range, e)
            return None

    period_dt_ranges = [
        f"{start_t.date().isoformat()}/{end_t.date().isoformat()}"
        for start_t, end_t in period_ranges
    ]
    period_scenes: list[dict[str, Any] | None] = list(
        await asyncio.gather(*[_safe_search(dtr) for dtr in period_dt_ranges])
    )

    # Need at least one period to anchor the grid (CRS + transform).
    anchor_period = next(
        ((idx, s) for idx, s in enumerate(period_scenes) if s is not None),
        None,
    )
    if anchor_period is None:
        raise SentinelFetchError(
            f"no Sentinel-2 scenes found in any of {n_periods} periods "
            f"({period_ranges[0][0].date()} → {period_ranges[-1][1].date()}, "
            f"{period_days}d each, cloud<{max_cloud_cover})"
        )

    # Resolve the anchor scene's CRS + a destination grid that matches
    # fetch_s2_composite (so the rest of the pipeline — sampling, CRS-aware
    # tile rendering — stays unchanged).
    _anchor_idx, anchor_scene = anchor_period
    anchor_token = await _get_sas_token(anchor_scene["collection"])
    anchor_assets = anchor_scene["assets"]
    want_bands = list(Modality.SENTINEL2_L2A.band_order)
    missing = [b for b in want_bands if b not in anchor_assets]
    if missing:
        raise SentinelFetchError(f"anchor scene missing bands: {missing}")

    first_href = _sign(anchor_assets[want_bands[0]]["href"], anchor_token)
    with rasterio.open(first_href) as src0:
        scene_crs = src0.crs

    west, south, east, north = transform_bounds(
        "EPSG:4326", scene_crs, bbox.west, bbox.south, bbox.east, bbox.north
    )
    width_m = abs(east - west)
    height_m = abs(north - south)
    native_gsd_m = 10.0
    longest_m = max(width_m, height_m)
    candidate_gsd = longest_m / float(max_size_px)
    gsd_m = max(native_gsd_m, candidate_gsd)
    out_width = max(1, int(round(width_m / gsd_m)))
    out_height = max(1, int(round(height_m / gsd_m)))
    dst_transform = from_bounds(west, south, east, north, out_width, out_height)

    stack = np.zeros((out_height, out_width, n_periods, 12), dtype=np.float32)
    timestamps: list[tuple[int, int, int]] = []
    scene_ids: list[str | None] = []
    scene_datetimes: list[str | None] = []
    cloud_covers: list[float | None] = []
    period_skipped: list[bool] = []

    for t, scene in enumerate(period_scenes):
        start_t, end_t = period_ranges[t]
        if scene is None:
            # Period midpoint — keeps the temporal encoding sensible even
            # though the band slice is zero.
            mid = start_t + (end_t - start_t) / 2
            timestamps.append((mid.day, mid.month - 1, mid.year))
            scene_ids.append(None)
            scene_datetimes.append(None)
            cloud_covers.append(None)
            period_skipped.append(True)
            continue

        token = await _get_sas_token(scene["collection"])
        assets = scene["assets"]
        period_missing = [b for b in want_bands if b not in assets]
        if period_missing:
            logger.warning(
                "temporal_stack: period %d missing bands %s — skipping",
                t, period_missing,
            )
            mid = start_t + (end_t - start_t) / 2
            timestamps.append((mid.day, mid.month - 1, mid.year))
            scene_ids.append(None)
            scene_datetimes.append(None)
            cloud_covers.append(None)
            period_skipped.append(True)
            continue

        period_failed = False
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
                        stack[:, :, t, band_idx] = vrt.read(1).astype(np.float32)
            except (rasterio.RasterioIOError, rasterio.errors.RasterioError) as e:
                logger.warning(
                    "temporal_stack: period %d band %s failed (%s) — skipping period",
                    t, band_name, e,
                )
                period_failed = True
                break

        if period_failed:
            # Zero out anything we may have partially written so the
            # encoder doesn't see a spectrally-mismatched mosaic.
            stack[:, :, t, :] = 0.0
            mid = start_t + (end_t - start_t) / 2
            timestamps.append((mid.day, mid.month - 1, mid.year))
            scene_ids.append(None)
            scene_datetimes.append(None)
            cloud_covers.append(None)
            period_skipped.append(True)
            continue

        ts = timestamp_from_iso(scene["datetime"])
        timestamps.append(ts)
        scene_ids.append(scene["id"])
        scene_datetimes.append(scene["datetime"])
        cloud_covers.append(scene.get("cloud_cover"))
        period_skipped.append(False)

    if all(period_skipped):
        raise SentinelFetchError(
            f"all {n_periods} periods skipped after band-read attempts "
            f"({period_ranges[0][0].date()} → {period_ranges[-1][1].date()})"
        )

    return SentinelTemporalStack(
        stack=stack,
        transform=dst_transform,
        crs=scene_crs,
        timestamps=timestamps,
        scene_ids=scene_ids,
        scene_datetimes=scene_datetimes,
        cloud_covers=cloud_covers,
        period_skipped=period_skipped,
        bbox_wgs84=(bbox.west, bbox.south, bbox.east, bbox.north),
    )


# ---------------------------------------------------------------------------
# Chunked native-resolution fetch — split AOI, share STAC searches, parallel
# band reads, return one chunk's BHWTC stack ready for inference.
# ---------------------------------------------------------------------------


def plan_chunks(bbox: BBox, chunk_size_m: int = 5000) -> list[BBox]:
    """Tile a WGS-84 bbox into a row-major grid of <= ``chunk_size_m`` sub-bboxes.

    Edge chunks may be smaller than the target. Lat/lon → meters via the
    AOI-center cosine factor; good enough for chunks well under 1°.
    """
    mid_lat = (bbox.north + bbox.south) / 2.0
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(mid_lat))
    m_per_deg_lat = 111_000.0
    chunk_deg_lon = chunk_size_m / m_per_deg_lon
    chunk_deg_lat = chunk_size_m / m_per_deg_lat

    n_cols = max(1, math.ceil((bbox.east - bbox.west) / chunk_deg_lon))
    n_rows = max(1, math.ceil((bbox.north - bbox.south) / chunk_deg_lat))
    actual_lon = (bbox.east - bbox.west) / n_cols
    actual_lat = (bbox.north - bbox.south) / n_rows

    chunks: list[BBox] = []
    for r in range(n_rows):
        for c in range(n_cols):
            chunks.append(BBox(
                west=bbox.west + c * actual_lon,
                south=bbox.south + r * actual_lat,
                east=bbox.west + (c + 1) * actual_lon,
                north=bbox.south + (r + 1) * actual_lat,
            ))
    return chunks


@dataclass(frozen=True)
class AoiPeriodScene:
    """One STAC search result valid for every chunk in a given period.

    Reused across chunks to avoid redundant STAC searches — a single S2
    scene is ~110 km wide and typically covers every chunk in a small AOI.
    """
    scene: dict[str, Any] | None      # None when the period had no usable scene
    period_start_iso: str
    period_end_iso: str


async def fetch_aoi_period_scenes(
    bbox: BBox,
    anchor_date: str,
    n_periods: int,
    period_days: int,
    time_offset_days: int = 0,
    max_cloud_cover: float = 40.0,
) -> list[AoiPeriodScene]:
    """One STAC search per period over the WHOLE AOI bbox.

    Periods are walked backward from ``anchor_date + time_offset_days`` in
    ``period_days`` chunks. The returned list has length ``n_periods``,
    aligned with the temporal order the encoder expects.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >=1, got {n_periods}")
    if period_days < 1:
        raise ValueError(f"period_days must be >=1, got {period_days}")

    from datetime import datetime, timedelta  # noqa: PLC0415

    anchor_str = anchor_date.split("/")[-1].strip().split("T", 1)[0]
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except ValueError as e:
        raise ValueError(
            f"anchor_date {anchor_date!r} not parseable as ISO date"
        ) from e
    window_end = anchor + timedelta(days=time_offset_days)

    period_ranges: list[tuple[datetime, datetime]] = []
    for t in range(n_periods):
        idx_from_end = n_periods - 1 - t
        end_t = window_end - timedelta(days=idx_from_end * period_days)
        start_t = end_t - timedelta(days=period_days)
        period_ranges.append((start_t, end_t))

    # Throttle concurrent STAC searches — firing 12 periods in parallel against
    # PC's /search endpoint reliably triggers TLS connection resets from a
    # residential IP (observed 50–80 % per-call flake when N=12 concurrent).
    # Serializing to 3 in-flight cuts the burst rate without significantly
    # hurting wall time (PC search is ~0.5 s when it works). Tunable via
    # OE_STAC_SEARCH_CONCURRENCY env var; ``0`` keeps the legacy unbounded
    # behaviour (only safe on intra-Azure deployments).
    _stac_search_sem = asyncio.Semaphore(
        max(1, int(os.environ.get("OE_STAC_SEARCH_CONCURRENCY", "3")))
    )

    async def _safe_search(dt_range: str) -> dict[str, Any] | None:
        async with _stac_search_sem:
            try:
                # _search_with_fallback tries PC first, then Element84 — so a
                # PC TLS reset cluster doesn't doom this period.
                return await _search_with_fallback(bbox, dt_range, max_cloud_cover)
            except SentinelFetchError as e:
                logger.info("aoi_period_scenes: period %s empty (%s)", dt_range, e)
                return None
            except Exception as e:
                # _http_retrying_request raises httpx exceptions after exhausting
                # its retry budget on BOTH PC and Element84. Catch them here so
                # one bad period doesn't take down the whole AOI search via gather().
                logger.warning(
                    "aoi_period_scenes: period %s failed after HTTP retries (%s)",
                    dt_range, e,
                )
                return None

    period_dt_ranges = [
        f"{s.date().isoformat()}/{e.date().isoformat()}"
        for s, e in period_ranges
    ]
    scenes = list(await asyncio.gather(*[_safe_search(dtr) for dtr in period_dt_ranges]))
    return [
        AoiPeriodScene(
            scene=scene,
            period_start_iso=s.date().isoformat(),
            period_end_iso=e.date().isoformat(),
        )
        for scene, (s, e) in zip(scenes, period_ranges)
    ]


async def fetch_s2_pre_post_pair(
    bbox: BBox,
    event_date: str,
    n_pre: int = 4,
    n_post: int = 4,
    pre_offset_days: int = 300,
    post_offset_days: int = 7,
    period_days: int = 30,
    max_cloud_cover: float = 40.0,
) -> tuple[list[AoiPeriodScene], list[AoiPeriodScene]]:
    """Fetch pre-event + post-event Sentinel-2 scene pairs for change-detection FT heads.

    Mirrors ForestLossDriver's training-time data spec — a pre group of
    scenes ~``pre_offset_days`` BEFORE the event and a post group ~``post_offset_days``
    AFTER it. Each group uses the same per-period least-cloudy search as
    :func:`fetch_aoi_period_scenes`, just with two different anchor offsets:

      * pre: anchor = event_date, time_offset_days = -pre_offset_days
      * post: anchor = event_date, time_offset_days = +post_offset_days

    Both lists are AOI-scoped (one search per period across the whole bbox);
    per-chunk reads use :func:`fetch_s2_chunk_stack` against each list
    independently and the caller concatenates encoder outputs along the
    feature dim.
    """
    pre = await fetch_aoi_period_scenes(
        bbox=bbox,
        anchor_date=event_date,
        n_periods=n_pre,
        period_days=period_days,
        time_offset_days=-pre_offset_days,
        max_cloud_cover=max_cloud_cover,
    )
    post = await fetch_aoi_period_scenes(
        bbox=bbox,
        anchor_date=event_date,
        n_periods=n_post,
        period_days=period_days,
        time_offset_days=post_offset_days,
        max_cloud_cover=max_cloud_cover,
    )
    return pre, post


def _read_one_band_window(
    href: str, scene_crs: Any, dst_transform: Any, out_h: int, out_w: int
) -> tuple[bool, np.ndarray]:
    """Blocking rasterio read of ONE band, aligned to (dst_transform, scene_crs).

    Wraps the ``rasterio.open`` call in ``rasterio.Env(**_GDAL_DEFAULTS)``
    so the GDAL/CURL timeouts + HTTP/2 knobs apply even if the env-var
    setting at module import didn't propagate (rasterio versions differ
    in how strictly they snapshot GDAL config). Without this wrapper the
    chunks sat on dead PC sockets for ~300 s each — observed repeatedly.

    Returns ``(ok, data)``. On failure, ``ok=False`` and ``data`` is a zero
    array — the caller decides whether to skip the period or proceed.
    Designed to run inside ``asyncio.to_thread`` so multiple bands fetch in
    parallel without the GIL (rasterio releases it during IO).

    Note: a Python-level retry-with-backoff was tried here but caused an
    unhandled exception path under live load (route returned 500 + a stuck
    worker). Reverted in favour of GDAL's native ``GDAL_HTTP_MAX_RETRY=5``
    (set at module top). PC-flake mitigation needs to land at a different
    surface — tracked separately.
    """
    try:
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=scene_crs,
                transform=dst_transform,
                width=out_w,
                height=out_h,
                resampling=Resampling.bilinear,
            ) as vrt:
                return True, vrt.read(1).astype(np.float32)
    except (rasterio.RasterioIOError, rasterio.errors.RasterioError) as e:
        short = href.rsplit("/", 1)[-1].split("?")[0][:50]
        logger.warning("band read failed [%s]: %s", short, e)
        return False, np.zeros((out_h, out_w), dtype=np.float32)


async def fetch_s2_chunk_stack(
    chunk_bbox: BBox,
    period_scenes: list[AoiPeriodScene],
    target_gsd_m: float = 10.0,
    pinned_crs: Any | None = None,
) -> SentinelTemporalStack:
    """Read one chunk's worth of every period from pre-fetched scene metadata.

    Optimized for the chunked-AOI orchestrator:
      * STAC searches were ALREADY done at AOI scope by ``fetch_aoi_period_scenes``
      * Each band is a windowed read of just this chunk's extent
      * All 12 bands of a period fetch in parallel via ``asyncio.gather`` +
        ``asyncio.to_thread`` (rasterio releases the GIL during IO, so the
        thread pool is the right tool here)
      * ``pinned_crs`` lets the orchestrator force every chunk onto the same
        UTM zone for clean stitching at AOI edges that straddle a zone

    Output shape mirrors :class:`SentinelTemporalStack` so this is a drop-in
    for the existing inference path. When all periods skip, raises
    ``SentinelFetchError`` so the caller can mark the chunk as nodata.
    """
    n_periods = len(period_scenes)
    if n_periods == 0:
        raise ValueError("period_scenes must be non-empty")

    want_bands = list(Modality.SENTINEL2_L2A.band_order)

    # Pick the anchor scene + CRS. Use ``pinned_crs`` when supplied; otherwise
    # take the first non-empty period's native CRS (matches the legacy
    # fetch_s2_temporal_stack contract).
    anchor_period_idx: int | None = None
    for idx, ps in enumerate(period_scenes):
        if ps.scene is not None and all(b in ps.scene["assets"] for b in want_bands):
            anchor_period_idx = idx
            break
    if anchor_period_idx is None:
        raise SentinelFetchError(
            f"chunk {chunk_bbox.model_dump()}: no period had a usable scene "
            f"with all 12 S2 bands"
        )
    anchor_ps = period_scenes[anchor_period_idx]
    assert anchor_ps.scene is not None
    anchor_first_href = await _signed_href_for_scene(
        anchor_ps.scene, anchor_ps.scene["assets"][want_bands[0]]["href"],
    )
    if pinned_crs is None:
        with rasterio.open(anchor_first_href) as src0:
            scene_crs = src0.crs
    else:
        scene_crs = pinned_crs

    # Compute the chunk's destination grid at the requested GSD in scene_crs.
    west, south, east, north = transform_bounds(
        "EPSG:4326", scene_crs,
        chunk_bbox.west, chunk_bbox.south, chunk_bbox.east, chunk_bbox.north,
    )
    width_m = abs(east - west)
    height_m = abs(north - south)
    out_w = max(1, int(round(width_m / target_gsd_m)))
    out_h = max(1, int(round(height_m / target_gsd_m)))
    dst_transform = from_bounds(west, south, east, north, out_w, out_h)

    stack = np.zeros((out_h, out_w, n_periods, 12), dtype=np.float32)
    from datetime import datetime  # noqa: PLC0415

    def _skip_record(ps: AoiPeriodScene) -> dict[str, Any]:
        ps_start = datetime.fromisoformat(ps.period_start_iso)
        ps_end = datetime.fromisoformat(ps.period_end_iso)
        mid = ps_start + (ps_end - ps_start) / 2
        return {
            "ts": (mid.day, mid.month - 1, mid.year),
            "scene_id": None,
            "scene_datetime": None,
            "cloud_cover": None,
            "skipped": True,
            "bands": None,
        }

    async def _fetch_one_period(ps: AoiPeriodScene) -> dict[str, Any]:
        scene = ps.scene
        if scene is None or any(b not in scene.get("assets", {}) for b in want_bands):
            return _skip_record(ps)

        # Check the local cache FIRST per-band. Each hit skips a full
        # rasterio.open + WarpedVRT + HTTP round-trip to PC — typically
        # 1–3 seconds saved per band. A fully cached period pays zero
        # network cost.
        cached_bands: dict[int, np.ndarray] = {}
        need_fetch: list[tuple[int, str]] = []
        for band_idx, band_name in enumerate(want_bands):
            hit = _s2_cache_get(scene["id"], chunk_bbox, band_name, target_gsd_m)
            if hit is not None and hit.shape == (out_h, out_w):
                cached_bands[band_idx] = hit
            else:
                need_fetch.append((band_idx, band_name))

        if need_fetch:
            # Sign once per scene (PC) or no-op (Element84 — public S3 URLs).
            signed_hrefs = [
                await _signed_href_for_scene(scene, scene["assets"][bn]["href"])
                for _, bn in need_fetch
            ]

            if _S2_SEQUENTIAL:
                # Safety mode — one read at a time. Matches the Ai2
                # tutorial notebook's pattern. Memory footprint per band
                # is bounded by one GDAL block + one numpy array (~tens
                # of MB) regardless of chunk/period count. Slower but
                # OOM-proof.
                results: list[tuple[bool, np.ndarray]] = []
                for href in signed_hrefs:
                    results.append(
                        await asyncio.to_thread(
                            _read_one_band_window,
                            href, scene_crs, dst_transform, out_h, out_w,
                        )
                    )
            else:
                # Parallel reads only for cache misses. Chunk-level sem
                # (4) + period-parallel (6) already caps concurrency at
                # ~4×6×12 = 288 reads, which PC tolerates in practice.
                results = await asyncio.gather(*[
                    asyncio.to_thread(
                        _read_one_band_window, href, scene_crs, dst_transform, out_h, out_w,
                    )
                    for href in signed_hrefs
                ])
            # Second-pass retry on the bands that failed in pass 1. PC's
            # /vsicurl/ COG asset reads occasionally drop TLS connections
            # (~20 % per-request observed) — when 12 bands run in parallel,
            # one transient failure used to skip the WHOLE period (line
            # ``if any(not ok ...)`` below). With 12 bands, P(period skip)
            # under a 20 % per-band rate is 1 - 0.8¹² ≈ 93 %. A single
            # extra pass on JUST the failed bands cuts joint failure to
            # ~7 % per band → P(period skip) ≈ 1 - 0.95¹² ≈ 46 % — and
            # since most "failures" are transient TLS resets, in practice
            # the retry rescues most periods.
            #
            # Why per-period and not inside ``_read_one_band_window``: the
            # latter approach (with time.sleep + broad OSError catch) tied
            # up the asyncio thread pool and caused live PCA requests to
            # 500 with stuck workers. Doing the retry one level up keeps
            # the existing exception envelope (only RasterioIOError /
            # RasterioError surface as failures) and re-uses the same
            # asyncio.to_thread fan-out the parallel pass already proved
            # safe. No sleep — GDAL's own ``GDAL_HTTP_MAX_RETRY=5`` already
            # adds spacing at the libcurl layer.
            failed_pass1 = [
                (i, signed_hrefs[i]) for i, (ok, _) in enumerate(results) if not ok
            ]
            if failed_pass1 and not _S2_SEQUENTIAL:
                logger.info(
                    "scene %s period: retrying %d/%d band reads that failed pass 1",
                    scene["id"], len(failed_pass1), len(signed_hrefs),
                )
                retry_results = await asyncio.gather(*[
                    asyncio.to_thread(
                        _read_one_band_window, href, scene_crs, dst_transform, out_h, out_w,
                    )
                    for _, href in failed_pass1
                ])
                for (orig_idx, _), retry_outcome in zip(failed_pass1, retry_results):
                    results[orig_idx] = retry_outcome
            if any(not ok for ok, _ in results):
                return _skip_record(ps)
            for (band_idx, band_name), (_, arr) in zip(need_fetch, results):
                cached_bands[band_idx] = arr
                _s2_cache_put(scene["id"], chunk_bbox, band_name, target_gsd_m, arr)

        if len(cached_bands) != len(want_bands):
            # Defensive — should be impossible with the checks above.
            logger.warning(
                "s2_cache: period missing bands after fetch (%d/%d) — skipping",
                len(cached_bands), len(want_bands),
            )
            return _skip_record(ps)

        if cached_bands and not need_fetch:
            logger.info(
                "s2_cache: scene %s fully cached (%d bands, 0 network)",
                scene["id"], len(want_bands),
            )
        elif cached_bands and need_fetch:
            logger.info(
                "s2_cache: scene %s partial (%d cached, %d fetched)",
                scene["id"], len(cached_bands) - len(need_fetch), len(need_fetch),
            )

        band_arrays = [cached_bands[i] for i in range(len(want_bands))]
        return {
            "ts": timestamp_from_iso(scene["datetime"]),
            "scene_id": scene["id"],
            "scene_datetime": scene["datetime"],
            "cloud_cover": scene.get("cloud_cover"),
            "skipped": False,
            "bands": band_arrays,
        }

    # ALL periods in parallel × 12 bands per period = up to 72 concurrent
    # rasterio reads per chunk. Combined with the chunk-level semaphore
    # (4 chunks max), the worst-case fan-out is 4 × 72 = 288 connections —
    # high but PC tolerates it because each individual read is small.
    # Without this period-level parallelism, a 1-chunk small AOI was paying
    # the latency of 6 sequential network round-trips even though the
    # actual bytes-per-period are tiny.
    period_results = await asyncio.gather(*[
        _fetch_one_period(ps) for ps in period_scenes
    ])

    timestamps: list[tuple[int, int, int]] = []
    scene_ids: list[str | None] = []
    scene_datetimes: list[str | None] = []
    cloud_covers: list[float | None] = []
    period_skipped: list[bool] = []

    for t, rec in enumerate(period_results):
        timestamps.append(rec["ts"])
        scene_ids.append(rec["scene_id"])
        scene_datetimes.append(rec["scene_datetime"])
        cloud_covers.append(rec["cloud_cover"])
        period_skipped.append(rec["skipped"])
        if not rec["skipped"]:
            for band_idx, arr in enumerate(rec["bands"]):
                stack[:, :, t, band_idx] = arr

    if all(period_skipped):
        raise SentinelFetchError(
            f"chunk {chunk_bbox.model_dump()}: all {n_periods} periods skipped"
        )

    return SentinelTemporalStack(
        stack=stack,
        transform=dst_transform,
        crs=scene_crs,
        timestamps=timestamps,
        scene_ids=scene_ids,
        scene_datetimes=scene_datetimes,
        cloud_covers=cloud_covers,
        period_skipped=period_skipped,
        bbox_wgs84=(chunk_bbox.west, chunk_bbox.south, chunk_bbox.east, chunk_bbox.north),
    )


async def resolve_aoi_grid(
    bbox: BBox,
    period_scenes: list[AoiPeriodScene],
    target_gsd_m: float = 10.0,
) -> tuple[Any, Any, int, int]:
    """Pick a single CRS for every chunk + build the global pixel grid.

    Reads the first non-empty period's anchor band to get the native UTM CRS,
    then projects the AOI bbox into it and computes a clean global affine at
    ``target_gsd_m``. The chunked orchestrator pins every chunk to this CRS
    so per-chunk outputs paste cleanly into the global raster without
    reprojection seams.

    Returns ``(crs, global_transform, global_h, global_w)``.
    """
    want_bands = list(Modality.SENTINEL2_L2A.band_order)
    first_ps = next(
        (ps for ps in period_scenes
         if ps.scene is not None
         and all(b in ps.scene.get("assets", {}) for b in want_bands)),
        None,
    )
    if first_ps is None:
        raise SentinelFetchError(
            f"resolve_aoi_grid: no period had a usable scene with all 12 S2 bands"
        )
    assert first_ps.scene is not None
    first_href = await _signed_href_for_scene(
        first_ps.scene, first_ps.scene["assets"][want_bands[0]]["href"],
    )
    with rasterio.open(first_href) as src0:
        pinned_crs = src0.crs

    g_west, g_south, g_east, g_north = transform_bounds(
        "EPSG:4326", pinned_crs, bbox.west, bbox.south, bbox.east, bbox.north
    )
    global_w = max(1, int(round(abs(g_east - g_west) / target_gsd_m)))
    global_h = max(1, int(round(abs(g_north - g_south) / target_gsd_m)))
    global_transform = from_bounds(g_west, g_south, g_east, g_north, global_w, global_h)
    return pinned_crs, global_transform, global_h, global_w


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
        # Provider tag — drives downstream signing logic. PC needs SAS
        # tokens appended to asset hrefs; Element84 (below) does not.
        "_provider": "pc",
    }


async def _search_element84_least_cloudy(
    bbox: BBox, datetime_range: str, max_cloud_cover: float
) -> dict[str, Any]:
    """Same shape as ``_search_least_cloudy`` but hits Element84's AWS-hosted
    Sentinel-2 STAC catalog. Used as a fallback when PC is flaking.

    Two normalizations applied so the returned scene plugs directly into
    the existing chunk-fetch path:

      1. Datetime ranges are converted to RFC3339 (Element84 rejects
         ``YYYY-MM-DD/YYYY-MM-DD``).
      2. Asset keys are remapped from Element84's friendly aliases
         (``coastal`` / ``blue`` / ``swir16`` / etc.) to PC's official
         ``B0X`` identifiers, since downstream code keys off
         ``Modality.SENTINEL2_L2A.band_order``.

    Asset hrefs from Element84 are public S3 HTTPS URLs (no SAS signing
    required); the ``"_provider": "element84"`` tag tells the chunk
    fetcher to skip ``_sign()`` for these scenes.
    """
    # Element84 wants RFC3339 datetimes — pad bare YYYY-MM-DD with T00:00:00Z.
    def _to_rfc3339(piece: str) -> str:
        piece = piece.strip()
        if "T" in piece:
            return piece if piece.endswith("Z") else piece + "Z"
        return piece + "T00:00:00Z"
    if "/" in datetime_range:
        a, b = datetime_range.split("/", 1)
        rfc = f"{_to_rfc3339(a)}/{_to_rfc3339(b)}"
    else:
        rfc = _to_rfc3339(datetime_range)

    body = {
        "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
        "datetime": rfc,
        "collections": ["sentinel-2-l2a"],
        "query": {"eo:cloud_cover": {"lt": float(max_cloud_cover)}},
        # E84's STAC backend (Elasticsearch) needs the full property path
        # for sortby; the bare key works on PC but 400s here with
        # "No mapping found for [eo:cloud_cover] in order to sort on".
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 5,
    }
    r = await _http_retrying_request("POST", f"{E84_STAC_API}/search", json=body)
    data = r.json()
    features = data.get("features") or []
    if not features:
        raise SentinelFetchError(
            f"element84: no Sentinel-2 scenes for bbox={bbox.model_dump()} "
            f"range={datetime_range} cloud<{max_cloud_cover}"
        )
    feat = features[0]
    props = feat.get("properties") or {}
    raw_assets = feat.get("assets") or {}

    # Remap E84's aliases → PC's B0X keys so the rest of the pipeline doesn't
    # need to know which provider produced the scene. Skip bands E84 doesn't
    # expose — leaving them out causes the per-chunk anchor check to drop
    # this scene rather than crash.
    assets: dict[str, Any] = {}
    for pc_name, e84_name in _E84_BAND_NAME_MAP.items():
        a = raw_assets.get(e84_name)
        if a and "href" in a:
            assets[pc_name] = a

    return {
        "id": feat["id"],
        "collection": feat.get("collection", "sentinel-2-l2a"),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "assets": assets,
        "_provider": "element84",
    }


def _provider_override() -> str | None:
    """Operator-side switch to force a specific Sentinel-2 provider.

    Set ``OE_S2_PROVIDER=element84`` to bypass PC entirely — search +
    asset URLs both come from Element84's AWS-hosted catalog. Useful
    when:
      * PC is degraded for your network (we observed band-read crawl
        from a North-American residential IP on 2026-04-27 even though
        STAC reachability was fine — E84's S3 path was healthy).
      * You're on AWS infrastructure where S3 traffic is intra-region
        free and faster.

    ``OE_S2_PROVIDER=pc`` (or unset, the default) keeps the previous
    behavior: PC primary, E84 fallback only when PC's STAC fails.

    Anything else logs a warning and falls back to default. Read fresh
    on every call so the operator can toggle without restarting (env
    var changes still need a uvicorn reload, but the flag itself is
    cheap to consult).
    """
    raw = os.environ.get("OE_S2_PROVIDER", "").strip().lower()
    if raw in ("element84", "e84", "aws"):
        return "element84"
    if raw in ("", "pc", "planetary_computer", "default"):
        return None
    logger.warning(
        "OE_S2_PROVIDER=%r not recognized; valid values are 'pc' or 'element84'. "
        "Falling back to default (PC primary, E84 fallback).",
        raw,
    )
    return None


async def _search_with_fallback(
    bbox: BBox, datetime_range: str, max_cloud_cover: float
) -> dict[str, Any]:
    """Try PC first; on any failure (HTTP retry exhaustion, 0-results, TLS
    reset cluster), fall through to Element84. Returns whichever finds a
    usable scene; raises only when BOTH providers come up empty.

    PC remains primary because it serves data co-located with our compute
    on Azure (when we eventually deploy there) and has SAS-signed CDN
    URLs which are slightly faster from arbitrary IPs. Element84 is the
    safety net for the "PC is having a bad day" case observed in dev
    where ~50 % of /search calls return TLS resets.

    When ``OE_S2_PROVIDER=element84`` is set, PC is skipped entirely.
    Useful when PC's signed Azure blob band reads crawl on the user's
    network (observed 2026-04-27) — E84 serves the same scenes from
    public S3 URLs on a different network path.
    """
    if _provider_override() == "element84":
        # Skip PC. E84 indexes the same ESA catalog so the scene
        # selection is equivalent (modulo race conditions on freshly-
        # ingested scenes), but the asset URLs come from a public S3
        # bucket — different network infrastructure, different
        # reliability profile.
        return await _search_element84_least_cloudy(
            bbox, datetime_range, max_cloud_cover,
        )
    try:
        return await _search_least_cloudy(bbox, datetime_range, max_cloud_cover)
    except SentinelFetchError:
        # PC said "no scenes here" — Element84 indexes the same catalog,
        # so it'll likely say the same. Skip the fallback to avoid a
        # pointless extra round-trip.
        raise
    except Exception as pc_err:
        logger.info(
            "PC search failed (%s) — falling back to Element84 for %s",
            type(pc_err).__name__, datetime_range,
        )
        try:
            return await _search_element84_least_cloudy(
                bbox, datetime_range, max_cloud_cover,
            )
        except SentinelFetchError:
            raise
        except Exception as e84_err:
            logger.warning(
                "BOTH STAC providers failed for %s: pc=%s, element84=%s",
                datetime_range, type(pc_err).__name__, type(e84_err).__name__,
            )
            # Re-raise PC's error since it was the primary attempt — the
            # caller's catch only knows about SentinelFetchError + Exception
            # so the type doesn't change observable behaviour.
            raise pc_err


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


async def _signed_href_for_scene(scene: dict[str, Any], href: str) -> str:
    """Provider-aware signing. PC needs a SAS token appended; Element84
    URLs are public S3 HTTPS — no signing required. Routing on the
    ``_provider`` tag keeps the chunk-fetch path provider-agnostic."""
    if scene.get("_provider") == "element84":
        return href
    token = await _get_sas_token(scene["collection"])
    return _sign(href, token)
