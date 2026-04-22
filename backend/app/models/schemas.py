from enum import Enum
from typing import Any

from pydantic import BaseModel


class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float


class AnalysisRequest(BaseModel):
    area: BBox
    data_sources: list[str] | None = None


class LandCoverClass(BaseModel):
    id: int
    name: str
    color: str
    percentage: float


class AnalysisResult(BaseModel):
    land_cover: list[LandCoverClass]
    bbox: BBox
    area_km2: float
    timestamp: str
    suitability_scores: dict[str, float] | None = None
    olmoearth: dict[str, Any] | None = None


class EnvDataResult(BaseModel):
    wind: dict[str, float] | None = None
    temperature: float | None = None
    solar_irradiance: float | None = None
    humidity: float | None = None
    # Set when the upstream Open-Meteo call couldn't complete (request
    # error, non-200 status, timeout). Allows the frontend to show "data
    # unavailable — upstream failure" rather than silently treating
    # missing fields as genuine absence. Stays ``None`` on success so
    # existing clients that don't read it are unaffected.
    error: str | None = None


class ElevationStats(BaseModel):
    """Summary statistics over the elevation grid.

    Values are in meters above mean sea level (Open-Meteo SRTM source).
    All four fields are always populated, even if every sample was null —
    in that degenerate case they all read 0.0 (see ``terrain.py``).
    """

    min: float
    max: float
    mean: float
    range: float


class ElevationResult(BaseModel):
    """Elevation grid + summary over a bbox.

    Shape matches the frontend ``ElevationResult`` interface byte-for-byte
    (``frontend/src/types/index.ts``). Added as a pydantic model so the
    ``/api/reconstruct`` and ``/api/elevation`` routes can advertise a
    ``response_model`` and catch drift at router-return time rather than
    letting a crash propagate into React.

    ``elevations`` cells are optional — Open-Meteo occasionally omits
    samples for coastal / offshore bboxes. The frontend must render a
    graceful fallback for null cells rather than assume a dense grid.
    """

    lats: list[float]
    lons: list[float]
    # Rows × cols of elevation samples, meters. ``None`` where Open-Meteo
    # had no SRTM coverage (e.g. ocean interior, polar extremes).
    elevations: list[list[float | None]]
    stats: ElevationStats
    resolution: int
    bbox: BBox


class ReconstructResponse(BaseModel):
    """Envelope returned by ``POST /api/reconstruct`` (and now
    ``GET /api/elevation``, aligned to the same shape).

    ``status`` is currently always ``"completed"`` — reserved so we can
    add ``"partial"`` / ``"cached"`` without breaking the contract. The
    frontend's ``client.ts::getElevation`` unwraps to return
    ``response.terrain`` so consumers see the raw ``ElevationResult``.
    """

    status: str
    terrain: ElevationResult


# --- Data format support ---


class DataFormat(str, Enum):
    GEOTIFF = "geotiff"
    COG = "cog"
    NETCDF = "netcdf"
    ZARR = "zarr"
    GEOPACKAGE = "geopackage"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    LAS = "las"
    LAZ = "laz"
    PARQUET = "parquet"
    GEOPARQUET = "geoparquet"
    CSV = "csv"
    UNKNOWN = "unknown"


class CRS(BaseModel):
    epsg: int | None = None
    wkt: str | None = None
    proj4: str | None = None


class RasterInfo(BaseModel):
    width: int
    height: int
    bands: int
    dtype: str
    nodata: float | None = None
    crs: CRS | None = None
    bbox: BBox | None = None
    resolution: tuple[float, float] | None = None
    band_names: list[str] | None = None


class VectorInfo(BaseModel):
    geometry_type: str
    feature_count: int
    crs: CRS | None = None
    bbox: BBox | None = None
    columns: list[str]
    sample_properties: dict[str, Any] | None = None


class PointCloudInfo(BaseModel):
    point_count: int
    bbox: BBox | None = None
    has_color: bool = False
    has_intensity: bool = False
    has_classification: bool = False
    point_format: int | None = None


class MultidimInfo(BaseModel):
    dimensions: dict[str, int]
    variables: list[str]
    coords: list[str]
    crs: CRS | None = None
    bbox: BBox | None = None
    time_range: tuple[str, str] | None = None
    attrs: dict[str, Any] | None = None


class DatasetInfo(BaseModel):
    filename: str
    format: DataFormat
    size_bytes: int
    raster: RasterInfo | None = None
    vector: VectorInfo | None = None
    point_cloud: PointCloudInfo | None = None
    multidim: MultidimInfo | None = None
    preview_geojson: dict | None = None  # Lightweight GeoJSON for map preview


# -----------------------------------------------------------------------------
# Response shapes for previously-untyped dict-returning routes.
#
# The audit caught that several POST routes declared ``-> dict`` without a
# ``response_model``, so FastAPI emitted them as schemaless ``object`` in the
# OpenAPI spec. The frontend was carrying its own hand-written TypeScript
# interfaces (``client.ts``) and trusting them to match reality. When a
# backend change touched a returned field name, the frontend still
# type-checked — shape drift only surfaced at runtime as undefined reads.
# Mirroring the client.ts shapes here gives us a single source of truth and
# lets FastAPI validate-on-egress.
# -----------------------------------------------------------------------------


class PolygonElevation(BaseModel):
    min_m: float
    max_m: float
    mean_m: float
    median_m: float
    range_m: float
    source: str


class Centroid(BaseModel):
    lat: float
    lon: float


class PolygonStatsResponse(BaseModel):
    perimeter_km: float
    area_km2: float
    centroid: Centroid
    bbox: BBox
    vertex_count: int
    elevation_sample_count: int
    elevation: PolygonElevation | None = None
    # Populated when elevation fetching fails — matches env_data's pattern
    # of prefixed error strings (timeout_... / unreachable_... / etc.).
    error: str | None = None


class StacItem(BaseModel):
    id: str
    collection: str
    datetime: str
    cloud_cover: float | None = None
    bbox: list[float]
    assets: list[str]
    thumbnail_url: str | None = None


class StacSearchResponse(BaseModel):
    count: int
    matched: int | None = None
    items: list[StacItem]
    error: str | None = None


class CompositeTileResponse(BaseModel):
    tile_url: str
    tilejson_url: str
    search_id: str
    collection: str
    assets: list[str]
    datetime_range: str
    bbox: BBox
    notes: list[str] | None = None
    error: str | None = None


class OlmoEarthLegendClass(BaseModel):
    name: str
    color: str


class OlmoEarthLegendColormap(BaseModel):
    kind: str  # "stops" | "classes" — frontend narrows on this tag
    note: str | None = None
    stops: list[Any] | None = None  # [["#hex", 0.0], ...]
    classes: list[OlmoEarthLegendClass] | None = None


class OlmoEarthInferenceResult(BaseModel):
    job_id: str
    tile_url: str
    legend: dict[str, Any] | None = None
    colormap: str
    kind: str  # "stub" | "pytorch"
    status: str  # "ready" | "running"
    model_repo_id: str
    bbox: BBox
    notes: list[str] | None = None
    # FT + pytorch paths fill these in. All optional on stub responses.
    task_type: str | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None
    class_names_tentative: bool | None = None
    class_probs: list[float] | None = None
    # Class ids that actually appear in the rendered raster (for
    # segmentation). Drives the frontend's "show only present colors"
    # legend so a 110-class head doesn't blast 110 swatches when the
    # user's AOI only contains ~5 classes.
    present_class_ids: list[int] | None = None
    prediction_value: float | None = None
    units: str | None = None
    decoder_key: str | None = None
    embedding_dim: int | None = None
    patch_size: int | None = None
    sliding_window: bool | None = None
    window_size: int | None = None
    scene_id: str | None = None
    scene_datetime: str | None = None
    scene_cloud_cover: float | None = None
    stub_reason: str | None = None
    # Stub-only: concrete retry suggestions the LLM can relay to the user.
    # Each entry is {description, params} where `params` maps to the
    # run_olmoearth_inference tool arguments — so the agent can literally
    # copy the params dict into its next tool call instead of inventing
    # vague "want to retry?" questions.
    suggested_retries: list[dict[str, Any]] | None = None
    # Set when the backend auto-retried internally after a stubbed first
    # attempt and the retry produced a real forward pass. Carries the
    # override params that were applied (subset of {date_range,
    # max_size_px, sliding_window, window_size}) so the LLM can mention
    # which knob was adjusted in its explanation to the user.
    auto_retry_applied: dict[str, Any] | None = None


class ProjectState(BaseModel):
    """Free-form session state persisted inside a Project. Kept as a
    loose dict on the wire so the frontend can add new keys without
    requiring a schema migration on the backend. Typical contents:
    ``selectedArea``, ``imageryLayers``, ``labeledFeatures``,
    ``customTags``, ``datasets``, ``projectName``, ``labelMode``."""

    model_config = {"extra": "allow"}


class ProjectWrite(BaseModel):
    """Payload for create/update. ``id`` only used on update — the POST
    create endpoint generates a fresh uuid and ignores any id the caller
    supplies so two tabs can't race to create the same project id."""
    name: str
    description: str | None = None
    state: dict[str, Any] = {}


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str  # ISO 8601 UTC
    updated_at: str
    state: dict[str, Any]


class ProjectSearchRequest(BaseModel):
    """OE Studio-style search payload. Minimal today — one LIKE filter on
    name — but keeps the POST /search resource pattern so adding more
    filter operators later (``{eq, inc, ne}`` à la OE Studio) doesn't
    break existing callers."""
    name_contains: str | None = None
    limit: int = 50
    offset: int = 0


class OlmoEarthDemoSpecInference(BaseModel):
    bbox: BBox
    model_repo_id: str
    date_range: str
    max_size_px: int
    sliding_window: bool
    window_size: int | None = None


class OlmoEarthLoadedModelsResponse(BaseModel):
    """In-memory warm-cache snapshot for the frontend's ready-timer badges.

    Distinct from ``/olmoearth/cache-status`` which reports *disk* cache
    state. This endpoint reports *process* cache state — which FT heads
    have already been through ``load_encoder()`` and so skip the 2–10 s
    safetensors re-read on the next inference call. The frontend uses it
    to paint a "warm" vs "cold" pill on each compare-demo button so users
    can predict the click cost.
    """
    loaded: list[str]


class OlmoEarthLegendHint(BaseModel):
    """Static colormap hint for a demo side, resolvable before inference runs.

    Distinct from ``OlmoEarthLegendColormap`` (which is populated by the
    real inference pipeline and can include discovered class lists). This
    shape carries only what's known up-front per FT repo: the colormap
    key, the human-readable axis label, an honesty note, and the gradient
    stops the frontend draws. Populated by ``olmoearth_demos._static_legend_for``.
    """
    colormap: str
    label: str | None = None
    note: str | None = None
    stops: list[list[Any]] = []  # [["#hex", 0.0], ["#hex", 0.5], ...]
    # Semantic anchors for the low/high ends of the gradient so the UI
    # doesn't just say "low / high" (useless for a mangrove softmax score).
    low_label: str | None = None
    high_label: str | None = None


class OlmoEarthDemoSide(BaseModel):
    id: str
    label: str
    tile_url: str
    job_id: str
    spec: OlmoEarthDemoSpecInference
    # Optional so FT heads without a registered colormap stay compatible —
    # the frontend just hides the legend block when absent.
    legend_hint: OlmoEarthLegendHint | None = None


class OlmoEarthDemoPair(BaseModel):
    id: str
    title: str
    blurb: str
    fit_bbox: BBox
    a: OlmoEarthDemoSide
    b: OlmoEarthDemoSide


class OlmoEarthDemoPairsResponse(BaseModel):
    pairs: list[OlmoEarthDemoPair]


class OlmoEarthDemoPrebakeScheduled(BaseModel):
    label: str
    job_id: str
    status: str


class OlmoEarthDemoPrebakeResponse(BaseModel):
    scheduled: list[OlmoEarthDemoPrebakeScheduled]


# Auto-label returns a FeatureCollection-shaped dict plus method/class metadata.
# The GeoJSON features are genuinely dynamic (count, geometry, properties
# depend on the raster + method), so we keep ``features`` as ``list[dict]``
# rather than forcing a strict pydantic GeoJSON schema — the frontend
# consumes this shape with ``as`` casts and doesn't care about feature
# internals beyond having a coordinates array.
class AutoLabelResponse(BaseModel):
    type: str  # always "FeatureCollection"
    features: list[dict[str, Any]]
    method: str | None = None
    classes: list[dict[str, Any]] | None = None
    # Pipelines differ in what extra summary fields they emit (tipsv2
    # publishes `model`, spectral publishes `n_classes`, samgeo publishes
    # `segmentation_params`), so we allow arbitrary extras via model_config.
    model_config = {"extra": "allow"}
