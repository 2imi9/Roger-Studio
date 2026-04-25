"""OlmoEarth catalog + loader + inference endpoints."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.models.schemas import (
    BBox,
    OlmoEarthDemoPairsResponse,
    OlmoEarthDemoPrebakeResponse,
    OlmoEarthInferenceResult,
    OlmoEarthLoadedModelsResponse,
)
from app.services import (
    olmoearth_datasets,
    olmoearth_demos,
    olmoearth_inference,
    olmoearth_loader,
    olmoearth_model,
)
from app.services.sentinel2_fetch import SentinelFetchError
from app.services.system_health import (
    AOISizeExceededError,
    CircuitBreakerTrippedError,
    ClientDisconnectedError,
    InsufficientMemoryError,
    measure_memory,
)


# --- Embedding export request schema (kept local to the router since it
# isn't shared with other modules; hoist to schemas.py if it becomes a
# public API contract or picks up frontend consumers).
class EmbeddingSimilarityRequest(BaseModel):
    """Parameters for the cosine-similarity embedding tool.

    ``query_lon`` / ``query_lat`` default to ``None`` so a quick demo
    (no pixel picking) just uses the AOI center — matches Ai2 tutorial's
    "show me what looks like this region" first-step flow. Once the
    frontend has a click-to-pick UI, callers will supply explicit
    coordinates."""

    bbox: BBox
    model_repo_id: str = Field(default="allenai/OlmoEarth-v1-Tiny")
    date_range: str = Field(default="2024-04-01/2024-10-01")
    n_periods: int = Field(default=12, ge=1, le=12)
    period_days: int = Field(default=30, ge=7, le=90)
    time_offset_days: int = Field(default=0)
    target_gsd_m: float = Field(default=10.0, ge=10.0, le=80.0)
    patch_size: int = Field(default=4, ge=1, le=8)
    chunk_size_m: int = Field(default=5000, ge=1000, le=20000)
    query_lon: float | None = Field(
        default=None,
        description="WGS-84 longitude of the query pixel. Defaults to AOI center.",
    )
    query_lat: float | None = Field(
        default=None,
        description="WGS-84 latitude of the query pixel. Defaults to AOI center.",
    )
    window_px: int = Field(
        default=1, ge=1, le=15,
        description=(
            "Mean-pool the query embedding over a window_px × window_px "
            "patch area. 1 = single pixel (matches Ai2 tutorial); larger "
            "windows trade spatial precision for noise robustness."
        ),
    )


class EmbeddingPCARgbRequest(BaseModel):
    """Parameters for the PCA false-color embedding tool — first of the
    in-UI embedding workflows. Mirrors EmbeddingExportRequest's defaults
    so the same UI controls drive both."""

    bbox: BBox
    model_repo_id: str = Field(default="allenai/OlmoEarth-v1-Tiny")
    date_range: str = Field(default="2024-04-01/2024-10-01")
    n_periods: int = Field(default=12, ge=1, le=12)
    period_days: int = Field(default=30, ge=7, le=90)
    time_offset_days: int = Field(default=0)
    target_gsd_m: float = Field(default=10.0, ge=10.0, le=80.0)
    patch_size: int = Field(default=4, ge=1, le=8)
    chunk_size_m: int = Field(default=5000, ge=1000, le=20000)


class EmbeddingExportRequest(BaseModel):
    """Parameters for a chunked OlmoEarth embedding export."""

    bbox: BBox
    model_repo_id: str = Field(
        default="allenai/OlmoEarth-v1-Tiny",
        description=(
            "Base encoder to use. Only base encoders (Nano/Tiny/Base/Large) "
            "are supported — FT heads produce task-specific outputs, not "
            "raw embeddings."
        ),
    )
    date_range: str = Field(
        default="2024-04-01/2024-10-01",
        description="RFC-3339 date range; used as the anchor end for the temporal stack.",
    )
    n_periods: int = Field(default=12, ge=1, le=12)
    period_days: int = Field(default=30, ge=7, le=90)
    time_offset_days: int = Field(default=0)
    target_gsd_m: float = Field(
        default=10.0, ge=10.0, le=80.0,
        description="Input pixel size in meters (10/20/40/80). Output embedding pixel = gsd × patch_size.",
    )
    patch_size: int = Field(default=4, ge=1, le=8)
    chunk_size_m: int = Field(default=5000, ge=1000, le=20000)


class EmbeddingFewShotRequest(BaseModel):
    """Few-shot semantic segmentation via nearest-prototype matching.

    The user clicks a few example points per class (the picker UI lifts
    the same primitive used by the similarity tool). Each class's
    embeddings are averaged → prototype; then every AOI pixel is
    assigned its nearest prototype by cosine similarity.

    No fine-tuning happens — this is purely embedding-space classification
    + a colour map, so it works on any base encoder for any region.
    """

    class _Point(BaseModel):
        lon: float = Field(..., ge=-180.0, le=180.0)
        lat: float = Field(..., ge=-90.0, le=90.0)

    class _Class(BaseModel):
        name: str = Field(..., min_length=1, max_length=64)
        color: str = Field(
            default="#888888",
            description="CSS hex colour (#RRGGBB) used in the response legend.",
        )
        points: list["EmbeddingFewShotRequest._Point"] = Field(..., min_length=1)

    bbox: BBox
    model_repo_id: str = Field(default="allenai/OlmoEarth-v1-Tiny")
    date_range: str = Field(default="2024-04-01/2024-10-01")
    n_periods: int = Field(default=3, ge=1, le=12)
    period_days: int = Field(default=30, ge=7, le=90)
    time_offset_days: int = Field(default=0)
    target_gsd_m: float = Field(default=10.0, ge=10.0, le=80.0)
    patch_size: int = Field(default=4, ge=1, le=8)
    chunk_size_m: int = Field(default=5000, ge=1000, le=20000)
    classes: list[_Class] = Field(
        ...,
        min_length=2,
        description=(
            "Two or more classes, each with a name + colour + at least one "
            "labelled point. The class index in the response is the index "
            "in this list; index 0 maps to the first class, etc."
        ),
    )


class FtClassificationGeoJsonRequest(BaseModel):
    """Parameters for vectorising an FT classification raster as GeoJSON.

    The pipeline reuses the same ``start_inference`` path that powers the
    map tile — so identical AOI + model + date_range hits the cached job
    and skips re-running inference. The request body matches ``/infer``'s
    minimum so the frontend can hand-off the same payload.
    """

    bbox: BBox
    model_repo_id: str = Field(
        ...,
        description=(
            "FT classification head, e.g. allenai/OlmoEarth-v1-FT-Mangrove-Base. "
            "Regression heads (LFMC) and embedding-only base encoders are rejected."
        ),
    )
    date_range: str | None = Field(
        default=None,
        description=(
            "Optional date range; defaults to the FT head's published recommendation "
            "(see olmoearth_ft.recommended_date_range)."
        ),
    )
    min_pixels: int = Field(
        default=4, ge=0, le=10_000,
        description=(
            "Drop polygons under this pixel count to suppress speckle. "
            "Default 4 = ~160 m² at 10 m GSD. Set to 0 to disable filtering."
        ),
    )
    simplify_tolerance_m: float = Field(
        default=5.0, ge=0.0, le=200.0,
        description=(
            "Douglas–Peucker tolerance in meters. 0 disables. Default 5 m is "
            "half S2 GSD — visually identical, often 5–10× smaller GeoJSON."
        ),
    )
    event_date: str | None = Field(
        default=None,
        description=(
            "ISO date for pre/post change-detection heads (ForestLossDriver). "
            "Plumbed into the underlying ``start_inference`` call so the cache "
            "key matches the map-tile run with the same event."
        ),
    )


_BASE_ENCODER_REPO_IDS = {
    "allenai/OlmoEarth-v1-Nano",
    "allenai/OlmoEarth-v1-Tiny",
    "allenai/OlmoEarth-v1-Base",
    "allenai/OlmoEarth-v1-Large",
}

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/olmoearth/catalog")
async def olmoearth_catalog(
    west: float | None = Query(None),
    south: float | None = Query(None),
    east: float | None = Query(None),
    north: float | None = Query(None),
    force: bool = Query(False, description="Bypass the 10-min TTL cache."),
) -> dict:
    """Return the live OlmoEarth catalog (models + datasets).

    Optional bbox adds project-coverage + a recommended model for that area.
    """
    if force:
        await olmoearth_datasets.refresh_live_catalog(force=True)

    bbox: BBox | None = None
    if all(v is not None for v in (west, south, east, north)):
        bbox = BBox(west=west, south=south, east=east, north=north)
    return await olmoearth_datasets.catalog_summary(bbox=bbox)


@router.get("/olmoearth/cache-status")
async def olmoearth_cache_status() -> dict:
    """Which OlmoEarth repos are cached on disk or currently loading."""
    return olmoearth_loader.status_snapshot()


@router.get("/olmoearth/loaded-models", response_model=OlmoEarthLoadedModelsResponse)
async def olmoearth_loaded_models() -> dict:
    """Which FT heads are resident in the process's in-memory model cache.

    Used by the compare-demo UI to predict click cost: a repo_id present
    here means the next inference skips the 2–10 s safetensors re-read
    (warm ~3 s total) vs. cold (~30 s). Cheap O(n=3–5) read under a lock.
    """
    return {"loaded": olmoearth_model.loaded_repo_ids()}


@router.post("/olmoearth/load")
async def olmoearth_load(
    payload: dict = Body(
        ...,
        description="{repo_id, repo_type: 'model'|'dataset', hf_token?}",
    ),
) -> dict:
    """Kick off a background `snapshot_download` for a repo and return its
    initial task state. The UI polls ``/olmoearth/cache-status`` for progress.
    """
    repo_id = payload.get("repo_id")
    repo_type = payload.get("repo_type", "model")
    hf_token = payload.get("hf_token")
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    if repo_type not in ("model", "dataset"):
        raise HTTPException(400, "repo_type must be 'model' or 'dataset'")
    return await olmoearth_loader.start_load(
        repo_id=repo_id, repo_type=repo_type, hf_token=hf_token,
    )


@router.post("/olmoearth/unload")
async def olmoearth_unload(payload: dict = Body(...)) -> dict:
    """Delete a cached repo from `~/.cache/huggingface/hub/`."""
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    result = olmoearth_loader.unload_repo(repo_id)
    if not result.get("removed") and result.get("error") != "not in cache":
        raise HTTPException(500, result.get("error", "unload failed"))
    return result


@router.get("/olmoearth/system-health")
async def olmoearth_system_health() -> dict:
    """Snapshot of host memory + whether a chunked job would be allowed
    right now. Cheap (one syscall); frontend can poll before clicking
    Run to show the user "free 1.2 GB before continuing" warnings
    instead of hitting a 503 after the request."""
    status = measure_memory()
    return {
        "total_gb": round(status.total_gb, 2),
        "available_gb": round(status.available_gb, 2),
        "used_gb": round(status.used_gb, 2),
        "percent_used": round(status.percent, 1),
        "threshold_gb": round(status.threshold_gb, 2),
        "ok": status.ok(),
    }


@router.post("/olmoearth/infer", response_model=OlmoEarthInferenceResult)
async def olmoearth_infer(request: Request, payload: dict = Body(...)) -> dict:
    """Register an inference job and return its XYZ tile URL.

    Returns **503** when the system-health precheck refuses the job for
    low free RAM — surfaces as a user-actionable error ("free N GB") in
    the UI instead of letting the chunked pipeline OOM-crash the host.

    Polls ``request.is_disconnected()`` from inside the chunked
    orchestrator — abandoning the tab cancels in-flight chunk fetches
    instead of letting them grind for the full 25-minute timeout window.
    """
    try:
        bbox = BBox(**payload["bbox"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"bbox is required: {e}") from e
    model_repo_id = payload.get("model_repo_id")
    if not model_repo_id:
        raise HTTPException(400, "model_repo_id is required")
    # Forward optional inference knobs so HTTP callers match the LLM tool's
    # parity. Silently dropping these meant sliding-window requests over HTTP
    # ran as single-window (bug), and suggested_retries couldn't key off the
    # real sliding_window state (follow-on bug in the stub-retry helper).
    try:
        return await olmoearth_inference.start_inference(
            bbox=bbox,
            model_repo_id=model_repo_id,
            date_range=payload.get("date_range"),
            max_size_px=int(payload.get("max_size_px", 256)),
            sliding_window=bool(payload.get("sliding_window", False)),
            window_size=int(payload.get("window_size", 32)),
            event_date=payload.get("event_date") or None,
            disconnect_check=request.is_disconnected,
        )
    except AOISizeExceededError as e:
        # 413 Payload Too Large is the RFC 7231 match: the resource the
        # client is asking the server to process is larger than the
        # server is willing to handle. Intentionally NO Retry-After —
        # retrying the same AOI won't help; user must shrink the bbox.
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InsufficientMemoryError as e:
        # 503 Service Unavailable with Retry-After so clients know to try
        # again after freeing memory. Body is the descriptive message so
        # the UI can render "close tabs / free X GB" directly.
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        # Network died mid-job. Partial scene cache survives — a retry
        # ~1 minute later will pick up from disk. Retry-After=60 hints
        # that at clients / the UI auto-retry layer.
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "60"},
        ) from e
    except ClientDisconnectedError as e:
        # 499 Client Closed Request — nginx convention. The tab/curl is
        # already gone; the body won't be read. Logging the cancellation
        # at info-level avoids polluting the error stream with what is
        # essentially "user changed their mind".
        logger.info("inference cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e


@router.get("/olmoearth/demo-pairs", response_model=OlmoEarthDemoPairsResponse)
async def olmoearth_demo_pairs() -> dict:
    """Return the curated compare-demo registry.

    Each entry carries a pre-computed ``job_id`` + tile URL for both sides
    (A and B). Tiles are rendered lazily — the first request to a tile
    kicks off inference via the regular ``/olmoearth/infer-tile`` route,
    which calls ``render_tile`` → ``start_inference`` under the hood.
    Clients that want eager warm-up can call
    ``POST /olmoearth/demo-pairs/prebake`` instead.
    """
    return {"pairs": olmoearth_demos.describe_pairs()}


@router.post("/olmoearth/demo-pairs/prebake", response_model=OlmoEarthDemoPrebakeResponse)
async def olmoearth_demo_prebake(pair_id: str | None = Query(None)) -> dict:
    """Fire-and-forget warm-up for demo inference runs.

    Schedules one ``start_inference`` task per demo spec as a background
    task and returns IMMEDIATELY with the list of job_ids. Inference runs
    asynchronously — subsequent tile requests to ``/olmoearth/infer-tile/
    {job_id}/...`` either (a) serve cached tiles if the run has finished,
    or (b) trigger a fresh inference on the same spec which collapses
    into the in-flight run via ``_make_job_id``'s deterministic hashing.
    Either way: no double-inference, no client-side timeout.

    Pass ``pair_id=<id>`` to limit warm-up to a single demo. Without
    filter, warms up every registered pair.
    """
    pairs = olmoearth_demos.DEMO_PAIRS
    if pair_id is not None:
        pairs = [p for p in pairs if p.id == pair_id]
        if not pairs:
            raise HTTPException(404, f"unknown demo pair: {pair_id}")

    specs: list[tuple[str, object]] = []
    for pair in pairs:
        for side in (pair.a, pair.b):
            specs.append((side.label, side))

    async def _warm(label: str, side: olmoearth_demos.DemoSpec) -> None:
        try:
            await olmoearth_inference.start_inference(
                bbox=BBox(west=side.west, south=side.south, east=side.east, north=side.north),
                model_repo_id=side.model_repo_id,
                date_range=side.date_range,
            )
        except Exception as e:  # noqa: BLE001 — warm-up is best effort
            logger.warning("demo prebake failed for %s: %s", label, e)

    # Schedule each warm-up as a detached task and return immediately.
    # asyncio.create_task means the task runs on the event loop without
    # the HTTP handler awaiting its completion — client sees < 50 ms
    # response, inference cooks in the background.
    scheduled: list[dict] = []
    for label, side in specs:
        asyncio.create_task(_warm(label, side))  # type: ignore[arg-type]
        scheduled.append({"label": label, "job_id": side.job_id, "status": "scheduled"})  # type: ignore[attr-defined]
    return {"scheduled": scheduled}


@router.get("/olmoearth/download/{job_id}.tif")
async def olmoearth_download_geotiff(job_id: str) -> Response:
    """Serve a finished inference raster as a georeferenced GeoTIFF.

    Handy for users who want the prediction outside the browser tiling
    flow — drop it into QGIS / ArcGIS / rasterio for overlay, change-
    detection, zonal stats, etc. Class-raster jobs get a single-band
    int16 with class IDs + ``CLASS_N`` tags on the band metadata so GIS
    tools can label hovers; regression jobs get a float32 scalar raster
    un-normalized to the task's native units (e.g. % LFMC).

    Returns 404 when the job doesn't exist / is still running / is a
    stub (synthetic rasters aren't useful for export).
    """
    result = olmoearth_inference.raster_geotiff_bytes(job_id)
    if result is None:
        raise HTTPException(
            404,
            f"No exportable raster for job {job_id!r}. Job may still be "
            "running, have failed to a preview stub, or be unknown.",
        )
    tif_bytes, filename = result
    return Response(
        content=tif_bytes,
        media_type="image/tiff",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # GeoTIFFs are immutable on disk (spec hash → deterministic
            # output) so cache aggressively.
            "Cache-Control": "public, max-age=3600",
            "X-Roger-Raster-Job-Id": job_id,
        },
    )


@router.post("/olmoearth/embedding-tools/similarity", response_model=OlmoEarthInferenceResult)
async def olmoearth_embedding_similarity(
    request: Request,
    req: EmbeddingSimilarityRequest = Body(...),
) -> dict:
    """Compute embeddings + render cosine-similarity heatmap to a query
    point. Second of the in-UI embedding tools.

    Response is identical to ``/infer`` so the frontend wires the
    returned ``tile_url`` as an ImageryLayer with no special-casing.
    Same safety stack (RAM precheck, breaker, AOI guardrail) inherited
    from the chunked orchestrator.

    Returns 400 on non-base-encoder model_repo_id.
    """
    if req.model_repo_id not in _BASE_ENCODER_REPO_IDS:
        raise HTTPException(
            400,
            (
                f"Similarity search needs a base encoder. "
                f"Got {req.model_repo_id!r}."
            ),
        )
    try:
        return await olmoearth_inference.run_embedding_tool_similarity(
            bbox=req.bbox,
            model_repo_id=req.model_repo_id,
            query_lon=req.query_lon,
            query_lat=req.query_lat,
            window_px=req.window_px,
            date_range=req.date_range,
            n_periods=req.n_periods,
            period_days=req.period_days,
            time_offset_days=req.time_offset_days,
            chunk_size_m=req.chunk_size_m,
            target_gsd_m=req.target_gsd_m,
            patch_size=req.patch_size,
            disconnect_check=request.is_disconnected,
        )
    except AOISizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InsufficientMemoryError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "60"},
        ) from e
    except ClientDisconnectedError as e:
        logger.info("similarity cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e


@router.post("/olmoearth/embedding-tools/pca-rgb", response_model=OlmoEarthInferenceResult)
async def olmoearth_embedding_pca_rgb(
    request: Request,
    req: EmbeddingPCARgbRequest = Body(...),
) -> dict:
    """Compute embeddings + render top-3 PCA components as a false-color
    map layer. First of the in-UI embedding tools (similarity search,
    few-shot classify, change detection follow with the same scaffolding).

    Response is identical in shape to ``/olmoearth/infer`` so the
    frontend treats the result like any other map-producing inference —
    just attach the returned ``tile_url`` as an ImageryLayer.

    Returns 503/413 on the same safety conditions as ``/infer`` (RAM,
    breaker, oversized AOI), 400 on non-base-encoder model_repo_id.
    """
    if req.model_repo_id not in _BASE_ENCODER_REPO_IDS:
        raise HTTPException(
            400,
            (
                f"PCA false-color is an embedding tool — only base "
                f"encoders (Nano/Tiny/Base/Large) produce raw "
                f"embeddings. Got {req.model_repo_id!r}."
            ),
        )
    try:
        return await olmoearth_inference.run_embedding_tool_pca_rgb(
            bbox=req.bbox,
            model_repo_id=req.model_repo_id,
            date_range=req.date_range,
            n_periods=req.n_periods,
            period_days=req.period_days,
            time_offset_days=req.time_offset_days,
            chunk_size_m=req.chunk_size_m,
            target_gsd_m=req.target_gsd_m,
            patch_size=req.patch_size,
            disconnect_check=request.is_disconnected,
        )
    except AOISizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InsufficientMemoryError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "60"},
        ) from e
    except ClientDisconnectedError as e:
        logger.info("PCA cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e


@router.post("/olmoearth/embedding-tools/few-shot", response_model=OlmoEarthInferenceResult)
async def olmoearth_embedding_few_shot(
    request: Request,
    req: EmbeddingFewShotRequest = Body(...),
) -> dict:
    """Few-shot semantic segmentation — user clicks examples per class,
    backend computes nearest-prototype classification over the AOI.

    Wraps :func:`olmoearth_inference.run_embedding_tool_few_shot`. The
    response is identical in shape to /infer, so the frontend treats
    the resulting tile_url + class legend exactly like any other FT
    classification — including the GeoJSON export endpoint
    (/ft-classification/geojson).

    Inherits the same safety stack as /infer (RAM precheck, oversized
    AOI, breaker, disconnect-cancel).
    """
    if req.model_repo_id not in _BASE_ENCODER_REPO_IDS:
        raise HTTPException(
            400,
            (
                f"Few-shot needs a base encoder. Got {req.model_repo_id!r}. "
                f"FT heads produce task-specific outputs, not embeddings."
            ),
        )

    # Pydantic models → plain dicts that the service layer expects
    # (avoids leaking the inner _Class / _Point types into the service).
    classes_payload: list[dict[str, object]] = [
        {
            "name": c.name,
            "color": c.color,
            "points": [
                {"lon": float(p.lon), "lat": float(p.lat)} for p in c.points
            ],
        }
        for c in req.classes
    ]
    try:
        return await olmoearth_inference.run_embedding_tool_few_shot(
            bbox=req.bbox,
            model_repo_id=req.model_repo_id,
            classes=classes_payload,
            date_range=req.date_range,
            n_periods=req.n_periods,
            period_days=req.period_days,
            time_offset_days=req.time_offset_days,
            chunk_size_m=req.chunk_size_m,
            target_gsd_m=req.target_gsd_m,
            patch_size=req.patch_size,
            disconnect_check=request.is_disconnected,
        )
    except ValueError as e:
        # Validation failure inside the service (fewer than 2 classes
        # in the request, non-base encoder slipping through) → 400.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SentinelFetchError as e:
        # Two distinct failure modes share this exception type:
        # (a) the chunked encoder pass returned 0 valid pixels (PC
        #     outage / cloud-locked AOI) → 503 + Retry-After
        # (b) the user clicked points that all fell outside the AOI
        #     or in nodata regions → 400 with the precise message
        # The two are easy to disambiguate: (b)'s message contains
        # "labelled point(s) but none landed on a valid embedding
        # patch", which is a user-input error. Anything else is
        # transient infra and should hint at retry.
        msg = str(e)
        if "labelled point" in msg or "no valid pixels in embedding" in msg:
            raise HTTPException(status_code=400, detail=msg) from e
        raise HTTPException(
            status_code=503, detail=msg, headers={"Retry-After": "30"},
        ) from e
    except AOISizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InsufficientMemoryError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "60"},
        ) from e
    except ClientDisconnectedError as e:
        logger.info("few-shot cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e


@router.post("/olmoearth/export-embedding")
async def olmoearth_export_embedding(
    request: Request,
    req: EmbeddingExportRequest = Body(...),
) -> Response:
    """Compute + export OlmoEarth embeddings for an AOI as an int8 COG.

    Mirrors Ai2 OlmoEarth Studio's "custom embedding exports" feature:
    runs the chosen base encoder over a chunked native-resolution S2
    pipeline, stitches per-patch embeddings, quantizes to int8 using the
    AlphaEarth-compatible scheme from olmoearth_pretrain.evals, and
    streams a COG with one band per embedding dimension.

    The int8 layout is bit-for-bit compatible with Studio's exports — use
    ``dequantize_embeddings`` from olmoearth_pretrain.evals.embedding_transforms
    to recover float vectors. nodata value is -128 (pixels with no scene
    coverage across all periods).

    Returns 400 when the requested model isn't a base encoder, and 500
    with a clear error body when every chunk fails (e.g. PC outage).
    """
    if req.model_repo_id not in _BASE_ENCODER_REPO_IDS:
        raise HTTPException(
            400,
            (
                f"{req.model_repo_id!r} is not a base encoder. Embedding "
                f"export supports only: "
                f"{sorted(_BASE_ENCODER_REPO_IDS)}."
            ),
        )

    # Load the base encoder (cached after first call by olmoearth_model).
    try:
        model, device = await asyncio.to_thread(
            olmoearth_model.load_encoder, req.model_repo_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to load {req.model_repo_id}: {e}") from e

    try:
        result = await olmoearth_inference._run_chunked_embedding_export(
            bbox=req.bbox,
            model=model,
            device=device,
            model_repo_id=req.model_repo_id,
            date_range=req.date_range,
            n_periods=req.n_periods,
            period_days=req.period_days,
            time_offset_days=req.time_offset_days,
            chunk_size_m=req.chunk_size_m,
            target_gsd_m=req.target_gsd_m,
            patch_size=req.patch_size,
            disconnect_check=request.is_disconnected,
        )
    except AOISizeExceededError as e:
        # AOI exceeds the deployment's chunk-count ceiling — refuse at
        # submit rather than stress the host. Same 413 pattern as /infer.
        raise HTTPException(status_code=413, detail=str(e)) from e
    except ClientDisconnectedError as e:
        logger.info("export-embedding cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e
    except InsufficientMemoryError as e:
        # System-health precheck rejected the job — surface as 503 with a
        # Retry-After hint so clients / UI can render "free RAM, retry".
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        raise HTTPException(
            status_code=503, detail=str(e),
            headers={"Retry-After": "60"},
        ) from e
    except Exception as e:
        logger.exception("embedding export failed")
        raise HTTPException(
            500, f"Embedding export failed: {type(e).__name__}: {e}",
        ) from e

    try:
        cog_bytes, filename = await asyncio.to_thread(
            olmoearth_inference.build_embedding_cog_bytes, result,
        )
    except Exception as e:
        logger.exception("COG serialization failed")
        raise HTTPException(
            500, f"COG serialization failed: {type(e).__name__}: {e}",
        ) from e

    return Response(
        content=cog_bytes,
        media_type="image/tiff",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Embedding-Dim": str(result["embedding_dim"]),
            "X-Embedding-Patch-GSD-M": str(
                result["target_gsd_m"] * result["patch_size"]
            ),
            "X-Chunks-Processed": str(result["chunks_processed"]),
            "X-Chunks-Failed": str(result["chunks_failed"]),
        },
    )


@router.post("/olmoearth/ft-classification/geojson")
async def olmoearth_ft_classification_geojson(
    request: Request,
    req: FtClassificationGeoJsonRequest = Body(...),
) -> Response:
    """Vectorise an FT classification raster into a downloadable GeoJSON.

    The Studio frontend renders FT outputs as colored XYZ tiles, which is
    great for in-app exploration but useless if the user wants to take the
    classification into Google Earth, QGIS, ArcGIS, or My Maps. This route
    runs (or reuses cached) inference, polygonises the per-pixel class
    raster, attaches the published per-class colours + areas, and streams
    the result as ``application/geo+json`` ready to drop into any GIS.

    Workflow:
      1. Run / reuse the same ``start_inference`` job the tile renderer uses
         — identical AOI + model + date_range hits the cached _jobs entry,
         no duplicate forward pass.
      2. Reject jobs whose ``task_type`` isn't classification (e.g. LFMC
         regression doesn't fit the polygon model — direct the user at
         Export-as-COG instead).
      3. ``vectorize_classification`` does the rasterio.features.shapes
         polygonisation + WGS84 reprojection + speckle filter.

    Errors:
      400 — model isn't a classification FT head.
      404 — no class raster available (regression task or stub fallback).
      413/503 — inherits the safety stack of /infer (oversize AOI, OOM,
              breaker trip).
    """
    # Step 1: reuse the same path as /infer so caching kicks in.
    try:
        infer_resp = await olmoearth_inference.start_inference(
            bbox=req.bbox,
            model_repo_id=req.model_repo_id,
            date_range=req.date_range,
            event_date=req.event_date,
            disconnect_check=request.is_disconnected,
        )
    except AOISizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InsufficientMemoryError as e:
        raise HTTPException(
            status_code=503, detail=str(e), headers={"Retry-After": "30"},
        ) from e
    except CircuitBreakerTrippedError as e:
        raise HTTPException(
            status_code=503, detail=str(e), headers={"Retry-After": "60"},
        ) from e
    except ClientDisconnectedError as e:
        logger.info("ft-classification cancelled (client disconnect): %s", e)
        raise HTTPException(status_code=499, detail=str(e)) from e

    job_id = infer_resp["job_id"]
    job = olmoearth_inference._jobs.get(job_id)
    if job is None:
        raise HTTPException(500, f"Job {job_id} disappeared after start_inference")

    # Step 2: reject non-classification tasks. The vectoriser only makes
    # sense for discrete-class outputs. Surface stub failures as 503 +
    # the underlying reason so the user sees "PC outage / no scenes /
    # breaker tripped" instead of a confusing "task_type=None" message
    # when the real failure was upstream.
    if job.get("kind") == "stub":
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference for {req.model_repo_id!r} failed and fell back "
                f"to a stub — no real classification raster to vectorise. "
                f"Underlying reason: {job.get('stub_reason') or 'unknown'}. "
                f"Retry when network stabilises (PC outage, all chunks "
                f"failed, or AOI had no cloud-free scenes)."
            ),
            headers={"Retry-After": "60"},
        )
    task_type = job.get("task_type")
    if task_type not in ("classification", "segmentation"):
        raise HTTPException(
            400,
            (
                f"Model {req.model_repo_id!r} returned task_type={task_type!r}, "
                f"not a classification. GeoJSON vectorisation only supports "
                f"classification heads (Mangrove, AWF, Ecosystem, etc.). "
                f"For regression (LFMC) or embeddings, use Export-as-COG."
            ),
        )

    class_raster = job.get("class_raster")
    if class_raster is None:
        # Stub fallback path — no real raster to vectorise.
        raise HTTPException(
            404,
            (
                f"Job {job_id} has no class raster (kind={job.get('kind')!r}, "
                f"stub_reason={job.get('stub_reason')!r}). Real inference "
                f"likely failed; retry when network stabilises."
            ),
        )

    # Step 3: vectorise. Run in a thread because shapes() + transform_geom
    # are CPU-bound and shouldn't block the event loop on big rasters.
    # Per-class colors live on ``legend_override`` (built by _build_ft_legend
    # with published task colors). The plain ``legend`` field is the
    # colormap-level legend which has gradient stops, not per-class colors.
    legend = job.get("legend_override") or job.get("legend") or {}
    legend_classes = legend.get("classes") if isinstance(legend, dict) else None
    target_gsd_m = (job.get("temporal_stack") or {}).get("target_gsd_m") or 10.0
    scene_datetime = (job.get("temporal_stack") or {}).get("scene_datetimes")
    if isinstance(scene_datetime, list) and scene_datetime:
        scene_datetime = next((s for s in scene_datetime if s), None)

    try:
        fc = await asyncio.to_thread(
            olmoearth_inference.vectorize_classification,
            class_raster=class_raster,
            transform=job["raster_transform"],
            crs=job["raster_crs"],
            target_gsd_m=float(target_gsd_m),
            class_names=job.get("class_names") or [],
            legend_classes=legend_classes,
            min_pixels=req.min_pixels,
            simplify_tolerance_m=req.simplify_tolerance_m if req.simplify_tolerance_m > 0 else None,
            extra_properties={
                "model_repo_id": req.model_repo_id,
                "scene_id": job.get("scene_id"),
                "scene_datetime": scene_datetime,
                "task_type": task_type,
                "job_id": job_id,
                "min_pixels": req.min_pixels,
                "simplify_tolerance_m": req.simplify_tolerance_m,
            },
        )
    except Exception as e:
        logger.exception("FT classification vectorisation failed")
        raise HTTPException(
            500,
            f"Vectorisation failed: {type(e).__name__}: {e}",
        ) from e

    # Stream as application/geo+json (RFC 7946) with a sensible filename.
    # ``model_short`` cuts the "allenai/OlmoEarth-v1-FT-" prefix so the
    # filename is human-readable when the user finds it in Downloads.
    import json  # noqa: PLC0415
    short_model = req.model_repo_id.split("/")[-1].replace("OlmoEarth-v1-FT-", "")
    filename = f"{short_model}_{job_id[:8]}.geojson"
    body = json.dumps(fc, separators=(",", ":")).encode("utf-8")
    return Response(
        content=body,
        media_type="application/geo+json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Feature-Count": str(len(fc.get("features", []))),
            "X-Job-Id": job_id,
        },
    )


@router.get("/olmoearth/infer-tile/{job_id}/{z}/{x}/{y}.png")
async def olmoearth_infer_tile(job_id: str, z: int, x: int, y: int) -> Response:
    """Serve one XYZ tile for an inference job.

    Cache-control strategy:
    * While the job is still running (``render_tile`` returns a transparent
      256×256 placeholder), respond with ``no-store`` so the client re-
      fetches on the next map render — otherwise MapLibre caches the
      transparent tile for an hour and the real output never appears even
      after inference completes.
    * Once ``status == "ready"``, switch to long cache (1 h) since the
      raster is immutable on disk (spec hash → deterministic output).
    * For unknown job_ids (not yet registered by ``start_inference``),
      fall back to a transparent tile with ``no-store`` instead of 404 so
      MapLibre keeps retrying — covers the frontend→backend race where
      tile requests beat the prebake POST by a few ms.
    """
    png = olmoearth_inference.render_tile(job_id, z, x, y)
    # Render_tile returns None only if the job_id is completely unknown.
    # Serve a transparent PNG in that case + no-store so clients retry.
    if png is None:
        from PIL import Image  # noqa: PLC0415
        import io as _io  # noqa: PLC0415
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = _io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )
    # Look up the job to decide cache freshness.
    job = olmoearth_inference._jobs.get(job_id)  # noqa: SLF001 — internal sync read
    is_ready = job is not None and job.get("status") == "ready"
    cache_header = "public, max-age=3600" if is_ready else "no-store"
    # Expose the job kind ("pytorch" / "stub" / "pending") + any stub
    # reason via response headers so SplitMap's pollReady (already doing
    # a GET to detect readiness via cache-control) can also surface
    # "this raster is a preview stub, not real inference" to the user.
    # Lowercase header keys to keep cross-browser fetch API access
    # consistent (browsers normalize on read).
    headers = {
        "Cache-Control": cache_header,
        "ETag": f'"{job_id}-{z}-{x}-{y}"',
        "X-Inference-Kind": str(job.get("kind", "unknown")) if job else "unknown",
    }
    if job is not None and job.get("stub_reason"):
        # Truncate defensively — the reason string can carry a full
        # exception repr. The frontend only needs a short summary.
        headers["X-Inference-Stub-Reason"] = str(job["stub_reason"])[:200]
    return Response(content=png, media_type="image/png", headers=headers)
