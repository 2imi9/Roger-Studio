"""GeoAgentOrchestrator — the single choke point where every geo framework
feeds evidence into Gemma 4 and Gemma's reasoning flows back into the GeoJSON.

Supported evidence sources (auto-detected, best-effort — missing sources are
simply omitted from the prompt):

    TIPSv2 polygons       -> classification + confidence + confidence_map
    SamGeo masks          -> mask quality, boundary crispness
    Spectral K-means      -> cluster ID + spectral signature
    Raster patch          -> cropped RGB from the source GeoTIFF
    Elevation (Open-Meteo)-> mean/min/max elevation over the polygon bbox
    Weather (Open-Meteo)  -> temp/humidity/solar/wind at polygon centroid
    OlmoEarth (future)    -> embedding + decoded class

Every polygon returned carries a `validation` block:

    {
      "validated_class": str,
      "original_class": str,
      "action": "accept" | "reclassify" | "split" | "reject",
      "confidence": float,
      "reasoning": str,
      "evidence_chain": [...],  # which sources contributed
    }
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services import gemma_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior remote-sensing geoscientist helping validate
land-cover labels produced by automated pipelines (TIPSv2 zero-shot, SAM
segmentation, or spectral K-means). You receive:

 - a cropped satellite image patch of the polygon
 - the proposed class and its confidence
 - surrounding context (elevation, weather, neighbor classes)

Your job: decide whether the proposed label is correct. If not, suggest the
correct class, or mark the polygon for human review. Be decisive but honest
about uncertainty.

Always respond with a single JSON object, no prose:
{
  "validated_class": "string matching one of the candidate classes OR a new one you propose",
  "action": "accept" | "reclassify" | "split" | "reject",
  "confidence": 0.0-1.0,
  "reasoning": "one or two sentences citing visible evidence in the patch",
  "evidence_used": ["patch", "elevation", "weather", "neighbors"]
}

Rules:
 - "accept": visible evidence supports the proposed class
 - "reclassify": patch clearly shows a different class; set validated_class
 - "split": patch contains 2+ distinct classes that should be separated
 - "reject": patch is cloud, shadow, NoData, or unanalyzable
 - Use terrain context: a 60° slope is not agriculture; sub-zero year-round is not tropical forest
"""


USER_PROMPT_TEMPLATE = """Validate this polygon.

Proposed class: {proposed_class}
Pipeline: {pipeline}
Pipeline confidence: {proposed_confidence}
Candidate classes: {candidates}

Geographic context:
  Centroid: ({lat:.4f}, {lon:.4f})
  Area: {area_km2:.3f} km^2
  Elevation: {elevation_str}
  Slope: {slope_str}

Environmental context (at centroid):
  Temperature: {temp_str}
  Humidity: {humidity_str}
  Solar irradiance: {solar_str}

Neighbor polygons (within 1km):
{neighbors_str}

Attached: 1 RGB patch cropped to the polygon's bounding box.

Return the JSON verdict.
"""


# ---------------------------------------------------------------------------
# Evidence gatherers — each can fail gracefully
# ---------------------------------------------------------------------------


def _polygon_centroid(geometry: dict) -> tuple[float, float]:
    """Crude centroid of a GeoJSON polygon. Returns (lon, lat)."""
    coords = geometry.get("coordinates", [[]])
    if not coords or not coords[0]:
        return (0.0, 0.0)
    ring = coords[0]
    lon = sum(pt[0] for pt in ring) / len(ring)
    lat = sum(pt[1] for pt in ring) / len(ring)
    return (lon, lat)


def _polygon_bbox(geometry: dict) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat)."""
    coords = geometry.get("coordinates", [[]])[0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _crop_raster_patch(raster_path: Path, bbox: tuple[float, float, float, float], size: int = 512):
    """Crop a square RGB patch from a GeoTIFF to the polygon bbox. Returns PIL.Image or None."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        from PIL import Image
    except ImportError:
        logger.warning("rasterio/PIL missing — patch cropping disabled")
        return None

    try:
        with rasterio.open(raster_path) as src:
            # Transform WGS84 bbox into raster CRS
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *bbox, densify_pts=21
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            data = src.read(
                indexes=[1, 2, 3][: min(3, src.count)],
                window=window,
                out_shape=(min(3, src.count), size, size),
                boundless=True,
                fill_value=0,
            )
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        # Normalize to 0-255
        p2, p98 = np.percentile(data, (2, 98))
        if p98 > p2:
            data = np.clip((data - p2) * 255 / (p98 - p2), 0, 255)
        arr = data.astype(np.uint8).transpose(1, 2, 0)
        return Image.fromarray(arr)
    except Exception as e:
        logger.warning(f"Patch crop failed: {e}")
        return None


async def _fetch_elevation(lat: float, lon: float) -> float | None:
    """Fetch single-point elevation from Open-Meteo. Best-effort; returns None on failure."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/elevation",
                params={"latitude": lat, "longitude": lon},
            )
            r.raise_for_status()
            return r.json()["elevation"][0]
    except Exception as e:
        logger.debug(f"Elevation fetch failed: {e}")
        return None


async def _fetch_weather(lat: float, lon: float) -> dict | None:
    """Fetch current weather at centroid from Open-Meteo."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,shortwave_radiation",
                },
            )
            r.raise_for_status()
            return r.json().get("current", {})
    except Exception as e:
        logger.debug(f"Weather fetch failed: {e}")
        return None


def _compute_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough planar area approx for sanity; not a projected calc."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # 1 deg lat ~ 111 km, 1 deg lon ~ 111*cos(lat) km
    avg_lat = (min_lat + max_lat) / 2
    height_km = (max_lat - min_lat) * 111.0
    width_km = (max_lon - min_lon) * 111.0 * np.cos(np.radians(avg_lat))
    return float(abs(height_km * width_km))


def _find_neighbors(
    target_feature: dict,
    all_features: list[dict],
    radius_km: float = 1.0,
    limit: int = 5,
) -> list[dict]:
    """Return up to `limit` neighbor features within radius_km of centroid."""
    t_cent = _polygon_centroid(target_feature["geometry"])
    neighbors = []
    for f in all_features:
        if f is target_feature:
            continue
        c = _polygon_centroid(f["geometry"])
        dist_km = np.sqrt(
            ((c[0] - t_cent[0]) * 111.0 * np.cos(np.radians(t_cent[1]))) ** 2
            + ((c[1] - t_cent[1]) * 111.0) ** 2
        )
        if dist_km <= radius_km:
            neighbors.append((dist_km, f))
    neighbors.sort(key=lambda x: x[0])
    return [n[1] for n in neighbors[:limit]]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def validate_feature(
    feature: dict,
    raster_path: Path | None,
    all_features: list[dict],
    candidate_classes: list[str],
    pipeline: str,
    include_weather: bool = True,
    include_elevation: bool = True,
) -> dict:
    """Run the full validation chain for one polygon.

    Returns the feature with a `validation` block attached to its properties.
    """
    props = feature.get("properties", {})
    geom = feature["geometry"]

    proposed_class = props.get("class_name") or props.get("class") or "unknown"
    proposed_conf = props.get("confidence", 0.0)

    bbox = _polygon_bbox(geom)
    lon, lat = _polygon_centroid(geom)
    area_km2 = _compute_area_km2(bbox)

    # Evidence gathering — fire-and-gather in parallel where possible
    evidence_chain: list[str] = []

    patch = None
    if raster_path is not None:
        patch = _crop_raster_patch(raster_path, bbox)
        if patch is not None:
            evidence_chain.append("patch")

    elev_task = _fetch_elevation(lat, lon) if include_elevation else asyncio.sleep(0, result=None)
    wx_task = _fetch_weather(lat, lon) if include_weather else asyncio.sleep(0, result=None)
    elevation, weather = await asyncio.gather(elev_task, wx_task)
    if elevation is not None:
        evidence_chain.append("elevation")
    if weather:
        evidence_chain.append("weather")

    neighbors = _find_neighbors(feature, all_features)
    if neighbors:
        evidence_chain.append("neighbors")

    # Render neighbor summary
    if neighbors:
        neighbors_str = "\n".join(
            f"  - {n['properties'].get('class_name', n['properties'].get('class', '?'))} "
            f"(conf {n['properties'].get('confidence', 0):.2f})"
            for n in neighbors
        )
    else:
        neighbors_str = "  (none within 1km)"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        proposed_class=proposed_class,
        pipeline=pipeline,
        proposed_confidence=f"{proposed_conf:.3f}",
        candidates=", ".join(candidate_classes),
        lat=lat,
        lon=lon,
        area_km2=area_km2,
        elevation_str=f"{elevation:.1f} m" if elevation is not None else "unknown",
        slope_str="derived from DEM (not computed in v1)",
        temp_str=f"{weather.get('temperature_2m')}°C" if weather else "unknown",
        humidity_str=f"{weather.get('relative_humidity_2m')}%" if weather else "unknown",
        solar_str=f"{weather.get('shortwave_radiation')} W/m²" if weather else "unknown",
        neighbors_str=neighbors_str,
    )

    images = [patch] if patch is not None else []

    try:
        verdict = await gemma_client.chat_with_vision(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=images,
        )
    except gemma_client.GemmaUnavailable as e:
        logger.error(f"Gemma unavailable during validate: {e}")
        feature["properties"]["validation"] = {
            "error": str(e),
            "evidence_chain": evidence_chain,
        }
        return feature

    feature["properties"]["validation"] = {
        "validated_class": verdict.get("validated_class", proposed_class),
        "original_class": proposed_class,
        "action": verdict.get("action", "accept"),
        "confidence": verdict.get("confidence", proposed_conf),
        "reasoning": verdict.get("reasoning", ""),
        "evidence_chain": evidence_chain,
        "evidence_used_by_agent": verdict.get("evidence_used", []),
        "pipeline": pipeline,
    }
    return feature


async def validate_geojson(
    geojson: dict,
    raster_path: Path | None,
    pipeline: str = "tipsv2",
    only_low_confidence: bool = True,
    low_conf_threshold: float = 0.6,
    max_concurrent: int = 4,
) -> dict:
    """Run validation across every polygon in a FeatureCollection.

    Args:
        geojson: a GeoJSON FeatureCollection (from TIPSv2 / SamGeo / Spectral).
        raster_path: path to the source GeoTIFF (for patch cropping). Optional.
        pipeline: which upstream labeler produced these features.
        only_low_confidence: if True, only validate polygons below threshold.
        low_conf_threshold: confidence cutoff.
        max_concurrent: how many polygons to run through Gemma in parallel
            (vLLM batches internally but we cap client-side too).

    Returns a new FeatureCollection with validation blocks and a summary.
    """
    features = geojson.get("features", [])
    if not features:
        return geojson

    # Derive candidate class set from the batch
    candidate_classes = sorted({
        f["properties"].get("class_name") or f["properties"].get("class") or "unknown"
        for f in features
    })

    # Pick features to validate
    def _needs(f):
        if not only_low_confidence:
            return True
        conf = f["properties"].get("confidence", 1.0)
        return conf < low_conf_threshold or f["properties"].get("needs_review", False)

    targets = [f for f in features if _needs(f)]
    skipped = len(features) - len(targets)
    logger.info(f"Validating {len(targets)}/{len(features)} features via Gemma ({skipped} passed through)")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(f):
        async with semaphore:
            return await validate_feature(f, raster_path, features, candidate_classes, pipeline)

    await asyncio.gather(*(_bounded(f) for f in targets))

    # Assemble summary
    validated = [f for f in features if "validation" in f["properties"]]
    actions = [f["properties"]["validation"].get("action") for f in validated]
    action_counts = {a: actions.count(a) for a in set(actions) if a}

    geojson.setdefault("properties", {})
    geojson["properties"]["gemma_validation"] = {
        "model": gemma_client.GEMMA_MODEL,
        "pipeline": pipeline,
        "validated_count": len(validated),
        "skipped_count": skipped,
        "action_counts": action_counts,
        "low_conf_threshold": low_conf_threshold,
    }
    return geojson
