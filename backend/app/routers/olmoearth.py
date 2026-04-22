"""OlmoEarth catalog + loader + inference endpoints."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, HTTPException, Query, Response

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


@router.post("/olmoearth/infer", response_model=OlmoEarthInferenceResult)
async def olmoearth_infer(payload: dict = Body(...)) -> dict:
    """Register an inference job and return its XYZ tile URL."""
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
    return await olmoearth_inference.start_inference(
        bbox=bbox,
        model_repo_id=model_repo_id,
        date_range=payload.get("date_range"),
        max_size_px=int(payload.get("max_size_px", 256)),
        sliding_window=bool(payload.get("sliding_window", False)),
        window_size=int(payload.get("window_size", 32)),
    )


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
