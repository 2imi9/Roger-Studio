import asyncio
import logging
import math
import random
from datetime import datetime, timezone

from fastapi import APIRouter

from app.models.schemas import AnalysisRequest, AnalysisResult, BBox, LandCoverClass
from app.services import olmoearth_datasets, worldcover
from app.services.worldcover import WorldCoverError

logger = logging.getLogger(__name__)

router = APIRouter()


def estimate_area_km2(bbox: BBox) -> float:
    lat_mid = math.radians((bbox.north + bbox.south) / 2)
    width_km = abs(bbox.east - bbox.west) * 111.32 * math.cos(lat_mid)
    height_km = abs(bbox.north - bbox.south) * 110.574
    return width_km * height_km


async def land_cover_from_worldcover(bbox: BBox) -> list[LandCoverClass]:
    """Real land cover via ESA WorldCover 10 m/pixel COG on Planetary Computer."""
    result = await worldcover.classify_land_cover(bbox, year=2021)
    return [
        LandCoverClass(
            id=entry["id"],
            name=entry["name"],
            color=entry["color"],
            percentage=entry["percentage"],
        )
        for entry in result.as_percentages()
    ]


def land_cover_heuristic(bbox: BBox) -> list[LandCoverClass]:
    """Legacy latitude-based heuristic. Used only when WorldCover fails — e.g.
    offline dev, PC outage, or a bbox in the handful of tiny regions where
    the COG lookup returns zero valid pixels."""
    lat = (bbox.north + bbox.south) / 2
    lng = (bbox.east + bbox.west) / 2
    abs_lat = abs(lat)

    # Seed from coordinates so same area gives consistent results
    rng = random.Random(int(lat * 100) ^ int(lng * 100))

    # Latitude-based biomes
    if abs_lat > 60:
        # Polar/boreal
        base = {"Forest": 20, "Grassland": 10, "Barren": 25, "Snow/Ice": 30, "Water": 10, "Wetland": 5}
    elif abs_lat > 40:
        # Temperate
        base = {"Forest": 35, "Cropland": 25, "Grassland": 15, "Urban": 12, "Water": 8, "Barren": 5}
    elif abs_lat > 23:
        # Subtropical
        base = {"Forest": 20, "Cropland": 30, "Grassland": 20, "Urban": 10, "Water": 5, "Barren": 15}
    else:
        # Tropical
        base = {"Forest": 45, "Cropland": 15, "Grassland": 10, "Urban": 8, "Water": 12, "Wetland": 10}

    # Coastal adjustment: if near known coastlines (very rough)
    if abs(lng) > 170 or (-130 < lng < -115 and 30 < lat < 50):
        base["Water"] = base.get("Water", 5) + 15
        base["Urban"] = base.get("Urban", 5) + 5

    # Add noise
    classes = []
    total = 0.0
    for i, (name, pct) in enumerate(base.items()):
        noisy = max(1, pct + rng.uniform(-8, 8))
        total += noisy
        colors = {
            "Forest": "#228b22", "Cropland": "#f0e68c", "Grassland": "#90ee90",
            "Urban": "#808080", "Water": "#1e90ff", "Barren": "#d2b48c",
            "Wetland": "#5f9ea0", "Snow/Ice": "#e0f0ff",
        }
        classes.append(LandCoverClass(id=i + 1, name=name, color=colors.get(name, "#999"), percentage=noisy))

    # Normalize to 100%
    for c in classes:
        c.percentage = round(c.percentage / total * 100, 1)

    classes.sort(key=lambda c: c.percentage, reverse=True)
    return classes


def compute_suitability(bbox: BBox, land_cover: list[LandCoverClass]) -> dict[str, float]:
    """Compute suitability scores based on location and land cover."""
    lat = (bbox.north + bbox.south) / 2
    abs_lat = abs(lat)
    lc_map = {c.name: c.percentage for c in land_cover}

    # Solar: best near equator, peaks at ~25° (sun belt)
    solar_base = max(0, 1.0 - abs(abs_lat - 25) / 50)
    solar = min(1.0, solar_base + lc_map.get("Barren", 0) / 200)

    # Wind: better at higher latitudes and coasts
    wind = min(1.0, 0.3 + abs_lat / 100 + lc_map.get("Grassland", 0) / 150)

    # Agriculture: temperate + existing cropland
    ag_base = max(0, 1.0 - abs(abs_lat - 40) / 40)
    agriculture = min(1.0, ag_base * 0.6 + lc_map.get("Cropland", 0) / 100 * 0.4)

    # Construction: inversely related to forest/water, positively to urban
    urban_frac = lc_map.get("Urban", 0) / 100
    forest_frac = lc_map.get("Forest", 0) / 100
    water_frac = lc_map.get("Water", 0) / 100
    construction = min(1.0, max(0.1, 0.5 + urban_frac * 0.3 - forest_frac * 0.2 - water_frac * 0.3))

    return {
        "solar_energy": round(solar, 2),
        "wind_energy": round(wind, 2),
        "agriculture": round(agriculture, 2),
        "construction": round(construction, 2),
    }


def _summarise_olmoearth(summary: dict) -> dict:
    """Trim ``catalog_summary`` output to the fields the Analysis panel renders.

    Full dataset/model catalogs are also returned (the frontend may hide them
    behind a disclosure) but the top-level ``highlight`` block is what drives
    the main card: which project regions overlap, which single model to try,
    and any caveats from the adapter.
    """
    coverage = summary.get("project_coverage") or []
    datasets = summary.get("datasets") or []
    models = summary.get("models") or []
    # catalog_summary already computes a live-catalog recommendation; fall back
    # to the static recommender only if the live fetch didn't produce one.
    recommended = summary.get("recommended_model") or olmoearth_datasets.recommend_model(
        task=(coverage[0]["dataset"].get("task") if coverage else None),
        size_preference="medium",
    )

    return {
        "highlight": {
            "project_coverage": [
                {"repo_id": c["repo_id"], "task": c["dataset"].get("task")}
                for c in coverage
            ],
            "recommended_model": recommended,
            "notes": summary.get("notes") or [],
        },
        "datasets": datasets,
        "models": models,
    }


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_area(req: AnalysisRequest) -> AnalysisResult:
    area_km2 = estimate_area_km2(req.area)

    # Fire the two independent network lookups in parallel. They were
    # previously serialized, so the endpoint's latency was
    # ``WorldCover STAC + COG reads (~15 s)`` + ``catalog_summary HF +
    # PC calls (~15 s)`` = easily >30 s on cold cache — which blew past
    # the client's 30 s default timeout and surfaced as
    # "API timeout after 30000 ms: /analyze" in the UI. Running them
    # under ``asyncio.gather`` halves wall-clock latency for the common
    # case and keeps the fallback path (WorldCover error → heuristic)
    # intact via ``return_exceptions=True``.
    lc_task = asyncio.create_task(land_cover_from_worldcover(req.area))
    oe_task = asyncio.create_task(olmoearth_datasets.catalog_summary(bbox=req.area))
    lc_result, oe_result = await asyncio.gather(
        lc_task, oe_task, return_exceptions=True
    )

    if isinstance(lc_result, BaseException):
        logger.warning(
            "WorldCover failed for bbox=%s: %s — using heuristic",
            req.area.model_dump(),
            lc_result,
        )
        land_cover = land_cover_heuristic(req.area)
        land_cover_source = "latitude-heuristic-fallback"
    else:
        land_cover = lc_result
        land_cover_source = "worldcover-2021"

    if isinstance(oe_result, BaseException):
        # catalog_summary never raised before this refactor (it swallows
        # network errors internally and returns a sparse payload); if it
        # ever does, a catalog-less response is still shippable — the
        # Analysis card just hides the OlmoEarth section.
        logger.warning(
            "catalog_summary failed for bbox=%s: %s", req.area.model_dump(), oe_result
        )
        oe_summary: dict = {"project_coverage": [], "datasets": [], "models": []}
    else:
        oe_summary = oe_result

    olmoearth_block = _summarise_olmoearth(oe_summary)
    olmoearth_block["land_cover_source"] = land_cover_source

    # Suitability scores dropped — they were latitude-based heuristics, not
    # real remote-sensing signals. Keep the schema field (nullable) so older
    # clients don't break; just stop populating it.
    return AnalysisResult(
        land_cover=land_cover,
        bbox=req.area,
        area_km2=area_km2,
        timestamp=datetime.now(timezone.utc).isoformat(),
        suitability_scores=None,
        olmoearth=olmoearth_block,
    )
