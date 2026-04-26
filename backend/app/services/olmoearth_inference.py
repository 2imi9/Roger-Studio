"""OlmoEarth inference pipeline — bbox + cached model → XYZ tile layer.

Two execution paths:

  1. **Real PyTorch path** (``kind="pytorch"``) — ``start_inference`` fetches a
     Sentinel-2 L2A composite over the bbox via Planetary Computer, runs the
     OlmoEarth encoder forward pass (``run_s2_inference``), PCA-projects the
     per-patch embedding to a scalar raster, and caches it alongside the
     scene's CRS + affine transform. ``render_tile`` then samples that
     prediction raster for each XYZ tile using a WGS-84 → scene-CRS
     reprojection. No watermark — these pixels ARE the model's view.

  2. **Stub fallback** (``kind="stub"``) — kept for graceful degradation:
     when the S2 fetch fails (no cloud-free scene, network drop) or the
     model forward raises, we still return a cohesive tile layer so the
     frontend doesn't silently lose the "Run on area" affordance. Stub
     tiles carry the translucent PREVIEW watermark so users can tell the
     two paths apart at a glance.

Design notes:
  - Jobs live in an in-memory dict keyed by a short hash of the spec. No
    DB; single-user dev backend.
  - Tile endpoint is HTTP-cacheable (ETag = job_id + z/x/y), so MapLibre's
    tile cache naturally holds warm tiles.
  - The real path blocks on ``asyncio.to_thread`` for the torch forward —
    ``uvicorn`` workers stay responsive while inference runs.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import io
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
import rasterio.warp
from rasterio.crs import CRS

from app.models.schemas import BBox
from app.services import olmoearth_ft, olmoearth_model
from app.services.system_health import (
    AOISizeExceededError,
    CircuitBreakerTrippedError,
    ClientDisconnectedError,
    check_aoi_size_or_raise,
    check_memory_or_raise,
    chunk_ram_ok,
    circuit_breaker_fail_rate_threshold,
    circuit_breaker_min_total_fails,
    circuit_breaker_threshold,
    should_trip_fractional,
)
from app.services.sentinel2_fetch import (
    SentinelFetchError,
    fetch_aoi_period_scenes,
    fetch_s2_chunk_stack,
    fetch_s2_composite,
    fetch_s2_pre_post_pair,
    image_to_bhwtc,
    plan_chunks,
    resolve_aoi_grid,
    stack_to_bhwtc,
    timestamp_from_iso,
)

logger = logging.getLogger(__name__)


# Jobs: job_id -> spec + status. A job is just a named render config; the
# real work happens up-front (real path) or per tile (stub fallback).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


def _set_progress(job_id: str | None, **fields: Any) -> None:
    """Best-effort update of a running job's progress fields.

    Called from within the chunked orchestrators as work proceeds so the
    frontend's InferenceProgressMonitor (polling GET /jobs/{id}/progress)
    can show stage + chunks_done/total + ETA in real time.

    Tolerant of:
      * job_id=None — silently no-op (some callers don't yet thread it)
      * unknown job_id — silently no-op (job evicted or never registered)
      * concurrent calls from multiple chunk coroutines — atomic dict
        mutations are GIL-safe at the field level; we don't need the
        async lock for monotonic counter bumps and the round-trip cost
        of acquiring it on every chunk completion would dominate the
        per-chunk overhead.
    """
    if job_id is None:
        return
    job = _jobs.get(job_id)
    if job is None:
        return
    for k, v in fields.items():
        job[k] = v


def get_job_progress(job_id: str) -> dict[str, Any] | None:
    """Snapshot of a job's progress for the GET /progress endpoint.

    Returns None when the job_id isn't registered (404 territory). Returns
    a flat dict with stage/chunks/timing fields suitable for direct JSON
    serialization. Computed fields (elapsed_ms, est_remaining_ms) are
    derived here so the frontend doesn't have to re-compute every poll.
    """
    job = _jobs.get(job_id)
    if job is None:
        return None
    started = job.get("started_ts") or 0.0
    elapsed_ms = int((time.time() - started) * 1000) if started else 0
    done = int(job.get("progress_chunks_done") or 0)
    total = int(job.get("progress_chunks_total") or 0)
    failed = int(job.get("progress_chunks_failed") or 0)
    # ETA: extrapolate per-chunk wall time × remaining, divided by
    # observed chunk concurrency (4 from chunk_sem). Only meaningful
    # once at least one chunk has completed; before that, return null.
    est_remaining_ms: int | None = None
    if done > 0 and total > done:
        per_chunk_ms = elapsed_ms / done
        remaining = total - done
        # Match the orchestrator's eta heuristic: divide by chunk_sem
        # capacity so 4× parallelism is reflected.
        est_remaining_ms = int((remaining * per_chunk_ms) / 4)
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "kind": job.get("kind", "pending"),
        "stage": job.get("progress_stage", "queued"),
        "message": job.get("progress_message", ""),
        "chunks_total": total,
        "chunks_done": done,
        "chunks_failed": failed,
        "elapsed_ms": elapsed_ms,
        "est_remaining_ms": est_remaining_ms,
        "stub_reason": job.get("stub_reason"),
    }


def preview_job_id(spec: dict[str, Any]) -> str:
    """Public wrapper around _make_job_id for the router's
    /infer/preview-id endpoint. Lets the frontend learn the job_id
    deterministically so it can poll progress while the real /infer
    request is still in flight."""
    return _make_job_id(spec)


# Global concurrency cap on chunked inference jobs (AOI inference + embedding
# export). This is the host-safety knob: chunk_sem(4) inside each orchestrator
# caps fan-out *within* a single request, but two simultaneous PCA requests
# would still pin 8 chunks × ~1.5 GB peak ≈ 12 GB at once. Five overlapping
# requests took a real laptop to 35 GB resident + 100% CPU for 37 hours and
# wedged the desktop — exactly the failure mode this guard prevents. Default
# is 1 (laptop-safe, all chunked work serializes); operators on Azure VMs
# with dedicated RAM can set ``OE_MAX_CONCURRENT_JOBS=2`` (or higher) to
# pipeline jobs.
_GLOBAL_JOB_SEM: asyncio.Semaphore | None = None


def _max_concurrent_jobs() -> int:
    env = os.environ.get("OE_MAX_CONCURRENT_JOBS")
    if env is None:
        return 1
    try:
        value = int(env)
    except ValueError:
        logger.warning(
            "OE_MAX_CONCURRENT_JOBS=%r not parseable as int — falling back to 1",
            env,
        )
        return 1
    return max(1, value)


def _global_job_sem() -> asyncio.Semaphore:
    """Lazy-init the module-level semaphore so it binds to the running event
    loop (asyncio.Semaphore created at module import time before a loop
    exists is fine in 3.10+, but tests that swap loops between cases need a
    fresh instance — see ``_reset_global_job_sem_for_tests``).
    """
    global _GLOBAL_JOB_SEM
    if _GLOBAL_JOB_SEM is None:
        _GLOBAL_JOB_SEM = asyncio.Semaphore(_max_concurrent_jobs())
    return _GLOBAL_JOB_SEM


def _reset_global_job_sem_for_tests() -> None:
    """Drop the cached global semaphore. Tests call this between cases when
    they monkeypatch ``OE_MAX_CONCURRENT_JOBS`` so the next call picks up
    the new value."""
    global _GLOBAL_JOB_SEM
    _GLOBAL_JOB_SEM = None


def _with_global_job_lock(fn):
    """Wrap a chunked-inference coroutine so only N can run at once across
    the whole process. N is set by ``OE_MAX_CONCURRENT_JOBS`` (default 1)."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with _global_job_sem():
            return await fn(*args, **kwargs)

    return wrapper


# Poll cadence for the disconnect watcher. 5 s is a balance: tight enough
# that an abandoned 25-min PCA dies within a fraction of a chunk's wall
# time, loose enough that the awaitable is essentially free in steady-state.
_DISCONNECT_POLL_INTERVAL_S = 5.0


async def _watch_for_disconnect(
    disconnect_check,
    target_task: asyncio.Task,
) -> None:
    """Cancel ``target_task`` as soon as the HTTP client goes away.

    ``disconnect_check`` is the FastAPI ``request.is_disconnected`` coroutine
    function — but we accept any ``Callable[[], Awaitable[bool]]`` so the
    orchestrator stays HTTP-agnostic and unit-testable. Loops until either
    the target finishes naturally (returns) or disconnect_check returns True
    (cancels target, then returns). Auto-cancelled by the orchestrator's
    ``finally`` once gather wakes.
    """
    while not target_task.done():
        try:
            disconnected = await disconnect_check()
        except Exception as e:
            # A broken disconnect_check shouldn't be able to wedge the job.
            # Log once and stop polling — gather still drives to completion.
            logger.warning("disconnect_check raised %s — disabling poll", e)
            return
        if disconnected:
            logger.warning(
                "client disconnected — cancelling in-flight chunked work"
            )
            target_task.cancel()
            return
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)

# Patch size used in both the forward pass and the coarseness of the output
# prediction raster. 4 matches the Inference-Quickstart default; smaller would
# give finer output but multiply GPU/CPU cost.
_PATCH_SIZE = 4

# Clamp on fetched S2 tile size. 256 matches OlmoEarth's 2.56 km pretraining
# tile at 10 m/pixel. On CPU, larger inputs inflate latency without visibly
# improving tile quality.
_DEFAULT_MAX_SIZE_PX = 256


def _make_job_id(spec: dict[str, Any]) -> str:
    """Stable hash of the job spec so identical requests share a job."""
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# Task → colormap. Picked to match OlmoEarth Studio's visual language.
# "probabilistic" maps get viridis-ish (green→yellow→orange), "binary"
# gets a bi-tonal colormap. Extend as real FT heads come online.
_COLORMAPS: dict[str, str] = {
    "allenai/OlmoEarth-v1-FT-LFMC-Base": "flammability",
    "allenai/OlmoEarth-v1-FT-Mangrove-Base": "mangrove",
    "allenai/OlmoEarth-v1-FT-AWF-Base": "landuse",
    "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base": "forestloss",
    "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": "ecosystem",
    "allenai/OlmoEarth-v1-Base": "embedding",
    "allenai/OlmoEarth-v1-Nano": "embedding",
    "allenai/OlmoEarth-v1-Tiny": "embedding",
    "allenai/OlmoEarth-v1-Large": "embedding",
}

#
# Legend copy is deliberately honest about what the numeric values
# represent. Earlier drafts labeled the mangrove output "Mangrove
# probability" which implies a calibrated 0-1 probability — but it's
# actually the raw post-softmax score from an uncalibrated classifier
# head. Two values with the same score don't mean "equally likely" in a
# well-defined Bayesian sense; they just mean the model ranks them the
# same. The audit called this out as a scientific-credibility risk
# (users may interpret "0.7 mangrove probability" as "70% confidence"
# when neither Platt scaling nor isotonic calibration has been applied).
# We now say "softmax score (uncalibrated)" / "class confidence
# (uncalibrated softmax)" / similar on every relevant task. Regression
# units stay unchanged — LFMC is in actual moisture-percent units, etc.
# ``low_label`` / ``high_label`` anchor the gradient with the actual thing
# each end means. Earlier the legend only said "low / high" which is
# useless for mangrove ("low of what?"). Now a viewer sees e.g.
# "non-mangrove → mangrove" and knows cyan = mangrove. For classification
# tasks the gradient is only a fallback visualization — the per-class
# palette comes from the runtime FT legend — but honest anchor copy here
# still helps users reading the static legend hint before inference runs.
_COLORMAP_LEGEND: dict[str, dict[str, Any]] = {
    "flammability": {
        "label": "Live fuel moisture (% — low → high)",
        "stops": [("#dc2626", 0.0), ("#f59e0b", 0.5), ("#16a34a", 1.0)],
        "note": "Regression output — predicted moisture percentage.",
        "low_label": "dry fuel",
        "high_label": "moist fuel",
    },
    "mangrove": {
        "label": "Mangrove softmax score (uncalibrated)",
        "stops": [("#0f172a", 0.0), ("#0891b2", 0.5), ("#67e8f9", 1.0)],
        "note": (
            "Raw post-softmax score — ranks pixels by model confidence "
            "but NOT a calibrated probability. Treat as a relative "
            "ordering, not a 0-1 likelihood."
        ),
        "low_label": "non-mangrove",
        "high_label": "mangrove",
    },
    "landuse": {
        "label": "Southern-Kenya land-use class (argmax)",
        "stops": [("#ca8a04", 0.0), ("#65a30d", 0.5), ("#0891b2", 1.0)],
        "note": (
            "Each pixel shows the argmax class id from an uncalibrated "
            "softmax. Ties resolved by lowest-index class."
        ),
        # The AWF head's class order is (roughly) cropland → rangeland/
        # grassland → water. Anchors here mirror that ordering so the
        # static gradient hint isn't misleading — the real per-class
        # palette still comes from the runtime FT legend.
        "low_label": "cropland",
        "high_label": "water",
    },
    "forestloss": {
        "label": "Forest-loss driver (argmax class)",
        "stops": [("#16a34a", 0.0), ("#facc15", 0.5), ("#dc2626", 1.0)],
        "note": (
            "Argmax class id from an uncalibrated softmax over driver "
            "categories (agriculture / settlements / fire / other)."
        ),
        "low_label": "intact forest",
        "high_label": "loss driver",
    },
    "ecosystem": {
        "label": "Ecosystem type (argmax of 110 classes)",
        "stops": [("#6366f1", 0.0), ("#ec4899", 0.5), ("#f59e0b", 1.0)],
        "note": (
            "Argmax class id from an uncalibrated softmax. With 110 "
            "classes the winning score is often <0.1 — ranking is "
            "informative, absolute confidence isn't."
        ),
        "low_label": "class 0",
        "high_label": "class 109",
    },
    "embedding": {
        "label": "Encoder feature magnitude (PCA 1st component)",
        "stops": [("#0f172a", 0.0), ("#6366f1", 0.5), ("#f59e0b", 1.0)],
        "note": (
            "Unsupervised feature visualization — no class semantics. "
            "Useful for inspecting what the encoder 'sees'; treat as "
            "exploratory, not predictive."
        ),
        "low_label": "low magnitude",
        "high_label": "high magnitude",
    },
    # Similarity heatmap for the ``embedding-tools/similarity`` endpoint.
    # Scalar raster carries cosine similarity rescaled to [0, 1]:
    #   0.0 = anti-correlated, 0.5 = unrelated, 1.0 = identical to query.
    # Gradient picks up rapidly past 0.75 so users can visually spot the
    # "genuinely similar" cluster; purple dominates the mid-band
    # (unrelated) so it reads as "nothing to see here" at a glance.
    "similarity": {
        "label": "Cosine similarity to query (0=opposite, 0.5=unrelated, 1=identical)",
        "stops": [
            ("#0f172a", 0.0),    # near-black for dissimilar / nodata
            ("#312e81", 0.5),    # indigo for unrelated
            ("#f59e0b", 0.75),   # amber where similarity starts to matter
            ("#fef3c7", 1.0),    # light yellow for near-identical
        ],
        "note": (
            "Each pixel's dot product with the query embedding, rescaled "
            "to [0, 1]. Bright = more like your query. Works globally — "
            "no labels required."
        ),
        "low_label": "dissimilar",
        "high_label": "similar",
    },
}


async def start_inference(
    bbox: BBox,
    model_repo_id: str,
    date_range: str | None = None,
    max_size_px: int = _DEFAULT_MAX_SIZE_PX,
    sliding_window: bool = False,
    window_size: int = 32,
    event_date: str | None = None,
    _auto_retry_depth: int = 0,
    disconnect_check=None,
) -> dict[str, Any]:
    """Register a job, run real OlmoEarth forward, cache the prediction raster.

    When ``sliding_window=True`` the S2 composite is tiled into
    ``window_size``-pixel non-overlapping windows and the FT model is run
    per-tile — useful for (a) scene-level tasks (classification / regression)
    that otherwise paint the whole bbox one color, and (b) larger bboxes
    where feeding 256+ pixels in one forward pass moves off-distribution.

    On real-path success (``kind="pytorch"``) the returned job carries a
    scene id + scene datetime so the UI can show what was actually classified.
    On fallback to stub (``kind="stub"``) the response includes a ``stub_reason``
    field explaining why. In either case the XYZ tile URL is identical.
    """
    spec = {
        "bbox": bbox.model_dump(),
        "model_repo_id": model_repo_id,
        "date_range": date_range or "2024-04-01/2024-10-01",
        "max_size_px": max_size_px,
        "sliding_window": sliding_window,
        "window_size": window_size if sliding_window else None,
        # event_date drives the pre/post split for change-detection FT
        # heads (ForestLossDriver). Hashed into job_id so two runs with
        # different events produce distinct cached jobs.
        "event_date": event_date,
    }
    job_id = _make_job_id(spec)

    async with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing is not None:
            if existing.get("status") == "ready" and existing.get("kind") != "stub":
                # Already finished with real pixels — hand back the cached
                # result. (Stub results deliberately skip this branch —
                # the failure that produced them was most likely transient
                # ― PC STAC hiccup, SAS token refresh race, parallel-prebake
                # resource contention — and a retry on the next click is
                # the right behavior. Without this carve-out, the user
                # had no way to recover a stubbed demo side without
                # restarting the backend.)
                return _build_response(existing)
            if existing.get("status") == "ready" and existing.get("kind") == "stub":
                logger.info(
                    "retrying inference for %s — previous attempt stubbed (reason=%s)",
                    job_id, existing.get("stub_reason"),
                )
                # Fall through to the _jobs[job_id] = running block below,
                # which re-registers as running and kicks off a fresh attempt.
            if existing.get("status") == "running":
                # Another caller is already running inference for this
                # exact spec. Return the pending stub so the second caller
                # doesn't kick off a duplicate download + forward pass.
                # Previously this branch fell through and ran inference
                # AGAIN in parallel, which at ~8 concurrent calls pinned
                # the GPU and made every job crawl. Demo-pair prebake fires
                # one POST per user click (plus the polling-loop HEADs),
                # so dedup is essential.
                logger.warning("inference %s already running — returning pending stub (dedup)", job_id)
                return _build_response(existing)
        _jobs[job_id] = {
            "job_id": job_id,
            "spec": spec,
            "status": "running",
            "kind": "pending",
            "colormap": _COLORMAPS.get(model_repo_id, "embedding"),
            "started_ts": time.time(),
            # Progress state — live-updated by the chunked orchestrators
            # below as work proceeds. Read by GET /jobs/{id}/progress so
            # the frontend monitor can show "X / N chunks" + ETA.
            "progress_stage": "queued",
            "progress_chunks_total": 0,
            "progress_chunks_done": 0,
            "progress_chunks_failed": 0,
            "progress_message": "Resolving scenes…",
        }

    try:
        real = await _run_real_inference(
            bbox, model_repo_id, spec["date_range"], max_size_px,
            sliding_window=sliding_window, window_size=window_size,
            event_date=event_date,
            disconnect_check=disconnect_check,
            job_id=job_id,
        )
    except (SentinelFetchError, Exception) as e:
        logger.warning("real inference failed for %s: %s — falling back to stub", job_id, e)
        async with _jobs_lock:
            _jobs[job_id].update(
                kind="stub",
                status="ready",
                stub_reason=f"{type(e).__name__}: {e}"[:500],
                legend=_COLORMAP_LEGEND.get(_jobs[job_id]["colormap"]),
            )
            stub_resp = _build_response(_jobs[job_id])
        # Auto-retry once with the first suggested retry params. Saves the
        # user a round-trip: the LLM otherwise asks "want me to retry with
        # max_size_px=128?" and the user has to say yes. By the time they
        # respond, the PC STAC cache may have evicted the scene list. Cap
        # depth at 1 so we can't recurse indefinitely, and never retry
        # when the user explicitly set non-default params (signal they
        # know what they want and don't want us second-guessing).
        retries = stub_resp.get("suggested_retries") or []
        # Skip auto-retry when the backend is already under inference load —
        # the demo-pair prebake runs 6 jobs concurrently, and doubling that
        # to 12 via auto-retry can throttle PC STAC or saturate CPU. The LLM
        # still sees `suggested_retries` and can offer them to the user;
        # we just don't eagerly fire a second attempt on top of a busy queue.
        running_jobs = sum(
            1 for j in _jobs.values() if j.get("status") == "running"
        )
        if (
            _auto_retry_depth == 0
            and retries
            and retries[0].get("params")
            and running_jobs <= 1  # only this job's own slot
        ):
            # Build a fresh param set by layering the suggestion on top of
            # the current call. Only params the suggestion specifies
            # override; everything else carries through.
            override = retries[0]["params"]
            new_kwargs = {
                "bbox": bbox,
                "model_repo_id": model_repo_id,
                "date_range": override.get("date_range", date_range),
                "max_size_px": int(override.get("max_size_px", max_size_px)),
                "sliding_window": bool(override.get("sliding_window", sliding_window)),
                "window_size": int(override.get("window_size", window_size)),
                "_auto_retry_depth": 1,
            }
            logger.info(
                "auto-retrying inference for %s with overrides=%s",
                job_id, override,
            )
            retry_resp = await start_inference(**new_kwargs)
            # Surface the retry outcome. If the retry succeeded (pytorch),
            # annotate that we auto-retried so the LLM can mention it.
            if retry_resp.get("kind") == "pytorch":
                notes = list(retry_resp.get("notes") or [])
                notes.append(
                    f"Auto-retry applied: first attempt stubbed; retried with "
                    f"{override} and got a real forward pass."
                )
                retry_resp["notes"] = notes
                retry_resp["auto_retry_applied"] = override
            return retry_resp
        return stub_resp

    async with _jobs_lock:
        _jobs[job_id].update(
            kind="pytorch",
            status="ready",
            legend=_COLORMAP_LEGEND.get(_jobs[job_id]["colormap"]),
            **real,
        )
        return _build_response(_jobs[job_id])


@_with_global_job_lock
async def _run_chunked_aoi_inference(
    bbox: BBox,
    model: olmoearth_ft.FTModel,
    device: Any,
    input_spec: dict[str, Any],
    effective_patch: int,
    date_range: str,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    sliding_window: bool = False,
    window_size: int = 64,
    disconnect_check=None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Native-resolution chunked inference for FT heads with ``input_spec``.

    The OlmoEarth viewer's quality advantage comes from running at full 10 m/
    pixel with tiled forward passes that match training distribution. We do
    the same here:

      1. Tile the AOI into ``chunk_size_m`` × ``chunk_size_m`` sub-bboxes
      2. Fire ONE STAC search per period over the WHOLE AOI (not per chunk;
         a single S2 scene is ~110 km wide and covers all chunks anyway)
      3. Pin every chunk to the AOI-wide UTM CRS so per-chunk outputs paste
         cleanly into a global raster without reprojection seams
      4. Per chunk: parallel-read all 12 bands × n_periods, run FT inference,
         upsample to native pixel resolution, paste into global raster
      5. Return the same response shape as the legacy FT branch so the
         caller doesn't need to know which path produced the result

    Latency estimate for the user's typical 20 km AOI at chunk_size_m=5000:
      * ~6 STAC searches (parallel) → ~2 s
      * 12 chunks × 12-band parallel reads × 6 periods → ~30–60 s total
      * 12 forward passes on GPU → ~5–10 s
      * Total: ~50–80 s, comfortably under the HTTP timeout
    """
    # Safety precheck FIRST — refuse to launch if free RAM is already
    # below the threshold. This turns a host-level force-shutdown (the
    # observed symptom when RAM spilled into swap) into a clean raised
    # exception that the router converts to HTTP 503. Cheap: one syscall.
    check_memory_or_raise()

    n_periods = int(input_spec["n_periods"])
    period_days = int(input_spec.get("period_days") or 30)
    time_offset_days = int(input_spec.get("time_offset_days") or 0)

    chunks = plan_chunks(bbox, chunk_size_m=chunk_size_m)
    # AOI size guardrail — refuse oversized AOIs BEFORE firing any STAC
    # or rasterio IO. Approximate AOI area in km² via the same
    # lat-corrected cosine trig ``plan_chunks`` uses internally so the
    # error message shows a number that matches the chunks count.
    _mid_lat = (bbox.north + bbox.south) / 2.0
    _m_per_deg_lon = 111_000.0 * math.cos(math.radians(_mid_lat))
    _aoi_area_km2 = (
        abs(bbox.east - bbox.west) * _m_per_deg_lon
        * abs(bbox.north - bbox.south) * 111_000.0
    ) / 1e6
    check_aoi_size_or_raise(
        chunks=len(chunks),
        chunk_size_m=chunk_size_m,
        aoi_area_km2=_aoi_area_km2,
    )
    logger.info(
        "chunked inference for %s: %d chunks (%d m each), n_periods=%d",
        model.repo_id, len(chunks), chunk_size_m, n_periods,
    )
    _set_progress(
        job_id,
        progress_chunks_total=len(chunks),
        progress_stage="resolving_scenes",
        progress_message=f"Searching Sentinel-2 scenes for {n_periods} period(s)…",
    )

    period_scenes = await fetch_aoi_period_scenes(
        bbox=bbox,
        anchor_date=date_range,
        n_periods=n_periods,
        period_days=period_days,
        time_offset_days=time_offset_days,
        max_cloud_cover=40.0,
    )
    if all(ps.scene is None for ps in period_scenes):
        raise SentinelFetchError(
            f"chunked inference: AOI-wide search returned 0 usable scenes "
            f"across {n_periods} periods"
        )

    pinned_crs, global_transform, global_h, global_w = await resolve_aoi_grid(
        bbox, period_scenes, target_gsd_m=target_gsd_m,
    )
    _set_progress(
        job_id,
        progress_stage="processing_chunks",
        progress_message=f"Fetching Sentinel-2 bands · 0 / {len(chunks)} chunks",
    )

    # Native-pixel-res output rasters. int32 for class indices matches
    # downstream expectations (raster_class_histogram + GeoTIFF export).
    global_class_raster = np.zeros((global_h, global_w), dtype=np.int32)
    global_scalar_raster = np.zeros((global_h, global_w), dtype=np.float32)

    # Chunk-level parallelism: cap at 4 concurrent fetches so 4 × 12 = 48
    # simultaneous PC connections, well under typical per-IP limits. GPU
    # forwards are serialized inside PyTorch's CUDA stream regardless of
    # how many we kick off, so concurrency here is a fetch-bound win, not
    # a compute-bound one. Net effect for a 12-chunk job: fetches overlap
    # 4-wide so total fetch wall time ≈ ceil(12/4) × per-chunk-fetch ≈
    # 3 × 15 s instead of 12 × 15 s.
    chunk_sem = asyncio.Semaphore(4)
    # Hard timeout per chunk, measured FROM THE MOMENT the chunk actually
    # acquires a sem slot — NOT from when asyncio.gather() kicked it off.
    # The previous version wrapped ``asyncio.wait_for(_process_chunk)``
    # around the coroutine INCLUDING its ``async with chunk_sem`` wait,
    # so a 25-chunk job produced the visible "all 25 timed out at t=180s"
    # pattern even when chunks 1–4 were legitimately working: chunks 5–25
    # sat in the sem queue for 180 s and then got cancelled before ever
    # trying to do network work. The fix is ``wait_for`` INSIDE the
    # semaphore in ``_process_chunk_bounded`` below. Bumping to 300 s too
    # — cached + small AOI is ~5 s, cold + large AOI can hit 120+.
    _CHUNK_TIMEOUT_S = 300.0
    # Sentinel value carried in result tuples for failed chunks. Stitching
    # below filters them out so a single-chunk failure doesn't poison the
    # whole AOI raster.
    _CHUNK_FAIL = object()

    # Circuit breaker: abort the job once the network is clearly not
    # recoverable. Two independent trip rules, each tunable via env:
    #   (a) consecutive — N chunks in a row fail (catches bursty drops)
    #   (b) fractional  — M+ total failures AND failure_rate > P
    #                     (catches the slow-burn 50 % flake rate where
    #                     every other chunk succeeds so the consecutive
    #                     counter keeps resetting but half the work is
    #                     wasted)
    _BREAKER_THRESHOLD = circuit_breaker_threshold()
    _BREAKER_MIN_TOTAL = circuit_breaker_min_total_fails()
    _BREAKER_RATE = circuit_breaker_fail_rate_threshold()
    breaker_state = {
        "consecutive_fails": 0,
        "tripped": False,
        "successes": 0,
        "failures": 0,
    }

    def _record_chunk_outcome(success: bool) -> None:
        """Update breaker state after a chunk completes. Call this from
        BOTH the inner _process_chunk (success/fail returns) AND the
        outer _process_chunk_bounded (timeout). One call per chunk."""
        if success:
            breaker_state["successes"] += 1
            breaker_state["consecutive_fails"] = 0
        else:
            breaker_state["failures"] += 1
            breaker_state["consecutive_fails"] += 1
        # Live-update the public progress fields the frontend monitor
        # polls. Done count includes BOTH successes and failures so the
        # progress bar always advances; failed count is a separate
        # rendering hint so the UI can flag partial-coverage early.
        _done = breaker_state["successes"] + breaker_state["failures"]
        _total = len(chunks)
        _set_progress(
            job_id,
            progress_chunks_done=_done,
            progress_chunks_failed=breaker_state["failures"],
            progress_message=(
                f"Fetching Sentinel-2 bands · {_done} / {_total} chunks"
                + (f" · {breaker_state['failures']} failed" if breaker_state["failures"] else "")
            ),
        )
        if success:
            return
        if breaker_state["tripped"]:
            return
        # Rule (a): consecutive trip.
        if (
            _BREAKER_THRESHOLD > 0
            and breaker_state["consecutive_fails"] >= _BREAKER_THRESHOLD
        ):
            breaker_state["tripped"] = True
            logger.warning(
                "circuit breaker TRIPPED (consecutive) — %d chunks failed "
                "in a row, aborting remaining work",
                _BREAKER_THRESHOLD,
            )
            return
        # Rule (b): fractional-rate trip.
        if should_trip_fractional(
            failures=breaker_state["failures"],
            successes=breaker_state["successes"],
            min_total_fails=_BREAKER_MIN_TOTAL,
            rate_threshold=_BREAKER_RATE,
        ):
            total = breaker_state["successes"] + breaker_state["failures"]
            breaker_state["tripped"] = True
            logger.warning(
                "circuit breaker TRIPPED (fractional) — %d failures out "
                "of %d chunks (%.0f%% fail rate > %.0f%% threshold)",
                breaker_state["failures"], total,
                100 * breaker_state["failures"] / total,
                100 * _BREAKER_RATE,
            )

    chunk_t0 = time.time()
    chunks_done_counter = {"n": 0}  # mutable shared counter for log lines

    async def _process_chunk(ch_idx: int, chunk_bbox: BBox) -> Any:
        """Do one chunk's fetch + inference + paste. No semaphore here —
        semaphore + timeout are layered in ``_process_chunk_bounded``."""
        t_chunk_start = time.time()
        try:
            stack_result = await fetch_s2_chunk_stack(
                chunk_bbox=chunk_bbox,
                period_scenes=period_scenes,
                target_gsd_m=target_gsd_m,
                pinned_crs=pinned_crs,
            )
        except SentinelFetchError as e:
            chunks_done_counter["n"] += 1
            elapsed = time.time() - chunk_t0
            logger.warning(
                "chunk %d/%d skipped (fetch, %.1fs): %s [done=%d/%d, elapsed=%.0fs]",
                ch_idx + 1, len(chunks), time.time() - t_chunk_start, e,
                chunks_done_counter["n"], len(chunks), elapsed,
            )
            return _CHUNK_FAIL
        t_after_fetch = time.time()

        ch, cw, _, _ = stack_result.stack.shape
        ch4 = (ch // effective_patch) * effective_patch
        cw4 = (cw // effective_patch) * effective_patch
        if ch4 == 0 or cw4 == 0:
            logger.warning(
                "chunk %d/%d too small for patch=%d (shape=%s)",
                ch_idx + 1, len(chunks), effective_patch, stack_result.stack.shape,
            )
            return _CHUNK_FAIL

        chunk_image = stack_to_bhwtc(stack_result.stack[:ch4, :cw4, :, :])
        try:
            # Forward releases GIL during CUDA dispatch; multiple chunks
            # may queue here concurrently but PyTorch serializes on the
            # GPU stream — concurrency above is fetch-bound, not compute.
            #
            # When sliding_window is on, each chunk runs the head over a
            # grid of ``window_size``-pixel windows instead of one
            # scene-level forward pass. For classification heads this
            # turns a chunk's "one class everywhere" output into one
            # class per ~64 px window — matches the head's training-time
            # ``predict_window_px`` and gives meaningful spatial detail
            # within each chunk. The output shape contract is identical
            # (FTInferenceResult with patch-resolution scalar +
            # class_raster) so the stitch + upsample paths below need
            # no changes.
            if sliding_window:
                chunk_result = await asyncio.to_thread(
                    olmoearth_model.run_ft_tiled_inference,
                    model,
                    chunk_image,
                    stack_result.timestamps,
                    window_size,
                    effective_patch,
                    device,
                )
            else:
                chunk_result = await asyncio.to_thread(
                    olmoearth_model.run_ft_inference,
                    model,
                    chunk_image,
                    stack_result.timestamps,
                    effective_patch,
                    device,
                )
        except Exception as e:
            logger.warning("chunk %d/%d inference failed: %s", ch_idx + 1, len(chunks), e)
            return _CHUNK_FAIL

        chunk_scalar_hi = np.repeat(
            np.repeat(chunk_result.scalar, effective_patch, axis=0),
            effective_patch, axis=1,
        )
        chunk_class_hi: np.ndarray | None = None
        if chunk_result.class_raster is not None:
            chunk_class_hi = np.repeat(
                np.repeat(chunk_result.class_raster, effective_patch, axis=0),
                effective_patch, axis=1,
            )

        chunk_west = stack_result.transform.c
        chunk_north = stack_result.transform.f
        col_offset = int(round((chunk_west - global_transform.c) / target_gsd_m))
        row_offset = int(round((global_transform.f - chunk_north) / target_gsd_m))

        if row_offset < 0 or col_offset < 0:
            logger.warning(
                "chunk %d/%d negative offset (row=%d col=%d) — skipping",
                ch_idx + 1, len(chunks), row_offset, col_offset,
            )
            return _CHUNK_FAIL
        # Paste-window dimensions come from the actual produced chunk
        # raster, not from ch4/cw4 — sliding-window inference trims
        # edge windows that don't fit a full ``window_size``, so the
        # tiled output may be SMALLER than the chunk's native footprint
        # (e.g. a 500-px chunk with window=64 emits 7×16=112 patches
        # = 448 native px; the bottom + right 52 px get dropped).
        chunk_native_h, chunk_native_w = chunk_scalar_hi.shape[:2]
        h_to_paste = min(chunk_native_h, global_h - row_offset)
        w_to_paste = min(chunk_native_w, global_w - col_offset)
        if h_to_paste <= 0 or w_to_paste <= 0:
            logger.warning(
                "chunk %d/%d zero-sized paste window — skipping",
                ch_idx + 1, len(chunks),
            )
            return _CHUNK_FAIL

        chunks_done_counter["n"] += 1
        elapsed = time.time() - chunk_t0
        t_total = time.time() - t_chunk_start
        t_fetch = t_after_fetch - t_chunk_start
        t_fwd = t_total - t_fetch
        done = chunks_done_counter["n"]
        # ETA = (avg time per completed chunk) × remaining chunks.
        avg_per = max(0.5, elapsed / max(1, done))
        remaining = len(chunks) - done
        eta_s = (remaining * avg_per) / 4  # 4 = chunk_sem capacity
        logger.info(
            "chunk %d/%d done in %.1fs (fetch=%.1fs, fwd=%.1fs) "
            "[done=%d/%d, elapsed=%.0fs, eta=%.0fs]",
            ch_idx + 1, len(chunks), t_total, t_fetch, t_fwd,
            done, len(chunks), elapsed, eta_s,
        )
        return {
            "scalar_hi": chunk_scalar_hi,
            "class_hi": chunk_class_hi,
            "row_offset": row_offset,
            "col_offset": col_offset,
            "h_to_paste": h_to_paste,
            "w_to_paste": w_to_paste,
            "result": chunk_result,
        }

    async def _process_chunk_bounded(ch_idx: int, chunk_bbox: BBox) -> Any:
        """Acquire a chunk_sem slot, then apply the timeout ONLY to actual
        work. Previously the timeout wrapped semaphore acquisition too,
        which made chunks 5–25 of a big gather fire their 180s timer
        while waiting for a slot and never actually running — "all chunks
        timed out at elapsed=180s" in the observed bug.

        Also short-circuits when the circuit breaker has tripped: once
        N consecutive chunks have failed, queued chunks return
        _CHUNK_FAIL immediately instead of waiting their turn to try
        the same dead network. This is what turns a 35-min grind into a
        15-min bail.
        """
        if breaker_state["tripped"]:
            # Don't even bother acquiring the sem — the breaker already
            # decided this job is doomed. Don't call _record_chunk_outcome
            # either; we're not consuming a "real" retry slot.
            return _CHUNK_FAIL
        async with chunk_sem:
            # Re-check inside the sem in case the breaker tripped while
            # we were waiting for a slot.
            if breaker_state["tripped"]:
                return _CHUNK_FAIL
            # Per-chunk RAM gate. The submit-time precheck is a one-shot
            # at job entry; this catches the case where free RAM has fallen
            # below the OS's swap-thrash floor while earlier chunks were
            # running (e.g. user opened more browser tabs, another ML
            # process grew). Failing fast here is much cheaper than
            # letting the next fetch + forward push the host into paging.
            ram_ok, ram_status = await asyncio.to_thread(chunk_ram_ok)
            if not ram_ok:
                chunks_done_counter["n"] += 1
                logger.warning(
                    "chunk %d/%d skipped — per-chunk RAM gate failed (%s)",
                    ch_idx + 1, len(chunks), ram_status.describe(),
                )
                _record_chunk_outcome(success=False)
                return _CHUNK_FAIL
            try:
                result = await asyncio.wait_for(
                    _process_chunk(ch_idx, chunk_bbox), timeout=_CHUNK_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                chunks_done_counter["n"] += 1
                elapsed = time.time() - chunk_t0
                logger.warning(
                    "chunk %d/%d TIMED OUT after %.0fs (network stuck?) "
                    "[done=%d/%d, elapsed=%.0fs]",
                    ch_idx + 1, len(chunks), _CHUNK_TIMEOUT_S,
                    chunks_done_counter["n"], len(chunks), elapsed,
                )
                _record_chunk_outcome(success=False)
                return _CHUNK_FAIL
            else:
                _record_chunk_outcome(success=result is not _CHUNK_FAIL)
                return result

    async def _heartbeat() -> None:
        """Emit a log line every 30 s so stuck-vs-slow is obvious from
        stdout. Auto-cancelled once gather() returns."""
        while True:
            await asyncio.sleep(30.0)
            done = chunks_done_counter["n"]
            elapsed = time.time() - chunk_t0
            remaining = len(chunks) - done
            # Estimate remaining wall time based on observed per-chunk rate.
            if done > 0:
                avg_per = elapsed / done
                eta_s = (remaining * avg_per) / 4  # chunk_sem capacity
                logger.info(
                    "heartbeat: %d/%d chunks done, %.0fs elapsed, ~%.0fs remaining",
                    done, len(chunks), elapsed, eta_s,
                )
            else:
                logger.info(
                    "heartbeat: 0/%d chunks done, %.0fs elapsed "
                    "(still on first batch — slow network?)",
                    len(chunks), elapsed,
                )

    # Fan out all chunks; semaphore caps actual concurrency. asyncio.gather
    # collects results in input order so stitching below is deterministic.
    # Wrapping gather in a task lets the disconnect watcher cancel it when
    # the HTTP client closes the connection — see _watch_for_disconnect.
    heartbeat_task = asyncio.create_task(_heartbeat())

    async def _gather_chunks():
        return await asyncio.gather(
            *[_process_chunk_bounded(i, c) for i, c in enumerate(chunks)]
        )

    gather_task = asyncio.create_task(_gather_chunks())
    disconnect_task = (
        asyncio.create_task(_watch_for_disconnect(disconnect_check, gather_task))
        if disconnect_check is not None else None
    )
    try:
        try:
            chunk_records = await gather_task
        except asyncio.CancelledError:
            # If the watcher cancelled us, surface as ClientDisconnectedError
            # so the route can return 499 instead of an opaque 500.
            if (
                disconnect_task is not None
                and disconnect_task.done()
                and not disconnect_task.cancelled()
            ):
                raise ClientDisconnectedError(
                    "client closed connection during chunked inference; "
                    f"in-flight chunks cancelled "
                    f"(processed={breaker_state['successes']}, "
                    f"failed={breaker_state['failures']}, total={len(chunks)})"
                )
            raise
    finally:
        heartbeat_task.cancel()
        if disconnect_task is not None:
            disconnect_task.cancel()
        for _t in (heartbeat_task, disconnect_task):
            if _t is None:
                continue
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass

    # Circuit breaker fired — raise BEFORE stitching so the caller (router)
    # can surface a clear "retry later" 503 with stats. Scene-cache writes
    # from the chunks that DID succeed are already on disk, so a retry
    # resumes much faster than starting from scratch.
    if breaker_state["tripped"]:
        raise CircuitBreakerTrippedError(
            processed=breaker_state["successes"],
            failed=breaker_state["failures"],
            total=len(chunks),
            threshold=_BREAKER_THRESHOLD,
        )

    _set_progress(
        job_id,
        progress_stage="stitching",
        progress_message="Stitching tiles…",
    )

    # Stitch: serialize the numpy paste step. Chunks are non-overlapping by
    # construction (plan_chunks) so concurrent paste would be safe in
    # principle, but doing it sequentially after gather keeps the diff
    # local and readable.
    last_chunk_result = None
    chunks_processed = 0
    chunks_failed = 0
    for record in chunk_records:
        if record is _CHUNK_FAIL:
            chunks_failed += 1
            continue
        global_scalar_raster[
            record["row_offset"]:record["row_offset"] + record["h_to_paste"],
            record["col_offset"]:record["col_offset"] + record["w_to_paste"],
        ] = record["scalar_hi"][:record["h_to_paste"], :record["w_to_paste"]]
        if record["class_hi"] is not None:
            global_class_raster[
                record["row_offset"]:record["row_offset"] + record["h_to_paste"],
                record["col_offset"]:record["col_offset"] + record["w_to_paste"],
            ] = record["class_hi"][:record["h_to_paste"], :record["w_to_paste"]]
        last_chunk_result = record["result"]
        chunks_processed += 1

    if chunks_processed == 0:
        raise SentinelFetchError(
            f"all {len(chunks)} chunks failed during chunked inference"
        )
    assert last_chunk_result is not None  # implied by chunks_processed > 0

    final_class_raster = global_class_raster if (
        last_chunk_result.class_raster is not None
    ) else None
    colormap = last_chunk_result.colormap
    legend = _build_ft_legend(last_chunk_result)

    present_class_ids: list[int] | None = None
    if final_class_raster is not None:
        uniq = np.unique(final_class_raster).astype(int)
        uniq = uniq[(uniq >= 0) & (uniq < last_chunk_result.num_classes)]
        present_class_ids = uniq.tolist()[:256]

    # Surface the most recent non-skipped period's scene metadata in the
    # response — matches the legacy temporal-stack contract.
    recent_period_idx = next(
        (i for i in range(n_periods - 1, -1, -1)
         if period_scenes[i].scene is not None),
        n_periods - 1,
    )
    recent_scene = period_scenes[recent_period_idx].scene or {}

    return {
        "raster_transform": global_transform,
        "raster_crs": pinned_crs,
        "raster_height": int(global_h),
        "raster_width": int(global_w),
        "scene_id": recent_scene.get("id"),
        "scene_datetime": recent_scene.get("datetime"),
        "scene_cloud_cover": recent_scene.get("eo:cloud_cover")
            or recent_scene.get("cloud_cover"),
        "patch_size": effective_patch,
        "sliding_window": False,
        "window_size": None,
        "temporal_stack": {
            "n_periods": n_periods,
            "periods_used": int(sum(1 for ps in period_scenes if ps.scene is not None)),
            "periods_skipped": int(sum(1 for ps in period_scenes if ps.scene is None)),
            "scene_ids": [
                ps.scene.get("id") if ps.scene else None for ps in period_scenes
            ],
            "scene_datetimes": [
                ps.scene.get("datetime") if ps.scene else None for ps in period_scenes
            ],
            "chunked": True,
            "chunk_size_m": chunk_size_m,
            "chunks_total": len(chunks),
            "chunks_processed": chunks_processed,
            "chunks_failed": chunks_failed,
            "target_gsd_m": target_gsd_m,
        },
        "scalar_raster": global_scalar_raster,
        "task_type": last_chunk_result.task_type,
        "num_classes": last_chunk_result.num_classes,
        "class_names": last_chunk_result.class_names,
        "class_names_tentative": last_chunk_result.class_names_tentative,
        "class_raster": final_class_raster,
        "class_probs": (
            last_chunk_result.class_probs.tolist()
            if last_chunk_result.class_probs is not None else None
        ),
        "present_class_ids": present_class_ids,
        "prediction_value": last_chunk_result.prediction_value,
        "units": last_chunk_result.units,
        "decoder_key": last_chunk_result.decoder_key,
        "colormap_override": colormap,
        "legend_override": legend,
    }


@_with_global_job_lock
async def _run_chunked_pre_post_inference(
    bbox: BBox,
    model: olmoearth_ft.FTModel,
    device: Any,
    input_spec: dict[str, Any],
    effective_patch: int,
    event_date: str,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    sliding_window: bool = False,
    window_size: int = 64,
    disconnect_check=None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Native-resolution chunked inference for pre/post change-detection FT heads.

    Mirrors :func:`_run_chunked_aoi_inference` (same chunking, safety stack,
    circuit breaker, RAM gates) but fetches TWO scene groups per chunk —
    pre-event and post-event — and concatenates encoder outputs along the
    feature dim before running the head. Required by ForestLossDriver,
    whose conv-pool-fc decoder expects 1536-channel input
    (``2 × 768 = pre + post``).

    Each chunk does roughly twice the work of the single-stack path
    (fetch + encoder forward both run twice), so wall time is ~2× the
    temporal-stack pipeline at the same chunk count.
    """
    check_memory_or_raise()

    n_pre = int(input_spec.get("n_periods", 8) // 2 or 4)
    n_post = int(input_spec.get("n_periods", 8) - n_pre or 4)
    pre_offset_days = int(-(input_spec.get("time_offset_days") or -300)) or 300
    post_offset_days = 7
    period_days = 30  # CONTAINS-spec metadata says 0; we use 30-day windows for least-cloudy search

    chunks = plan_chunks(bbox, chunk_size_m=chunk_size_m)
    _mid_lat = (bbox.north + bbox.south) / 2.0
    _m_per_deg_lon = 111_000.0 * math.cos(math.radians(_mid_lat))
    _aoi_area_km2 = (
        abs(bbox.east - bbox.west) * _m_per_deg_lon
        * abs(bbox.north - bbox.south) * 111_000.0
    ) / 1e6
    check_aoi_size_or_raise(
        chunks=len(chunks),
        chunk_size_m=chunk_size_m,
        aoi_area_km2=_aoi_area_km2,
    )
    logger.info(
        "chunked pre/post inference for %s: %d chunks, n_pre=%d, n_post=%d, event=%s",
        model.repo_id, len(chunks), n_pre, n_post, event_date,
    )
    _set_progress(
        job_id,
        progress_chunks_total=len(chunks),
        progress_stage="resolving_scenes",
        progress_message=f"Searching pre/post scene pairs around {event_date}…",
    )

    pre_scenes, post_scenes = await fetch_s2_pre_post_pair(
        bbox=bbox,
        event_date=event_date,
        n_pre=n_pre,
        n_post=n_post,
        pre_offset_days=pre_offset_days,
        post_offset_days=post_offset_days,
        period_days=period_days,
        max_cloud_cover=40.0,
    )
    if all(ps.scene is None for ps in pre_scenes):
        raise SentinelFetchError(
            f"chunked pre/post: pre-event search returned 0 usable scenes "
            f"across {n_pre} periods at {event_date} - {pre_offset_days}d"
        )
    if all(ps.scene is None for ps in post_scenes):
        raise SentinelFetchError(
            f"chunked pre/post: post-event search returned 0 usable scenes "
            f"across {n_post} periods at {event_date} + {post_offset_days}d"
        )

    # Use the union of pre + post scenes to resolve a single AOI grid so
    # both groups read the same chunk extents — the head requires identical
    # H, W on both sides of the concat.
    pinned_crs, global_transform, global_h, global_w = await resolve_aoi_grid(
        bbox, list(pre_scenes) + list(post_scenes), target_gsd_m=target_gsd_m,
    )
    _set_progress(
        job_id,
        progress_stage="processing_chunks",
        progress_message=f"Fetching pre + post Sentinel-2 bands · 0 / {len(chunks)} chunks",
    )

    global_class_raster = np.zeros((global_h, global_w), dtype=np.int32)
    global_scalar_raster = np.zeros((global_h, global_w), dtype=np.float32)

    chunk_sem = asyncio.Semaphore(4)
    _CHUNK_TIMEOUT_S = 300.0
    _CHUNK_FAIL = object()

    _BREAKER_THRESHOLD = circuit_breaker_threshold()
    _BREAKER_MIN_TOTAL = circuit_breaker_min_total_fails()
    _BREAKER_RATE = circuit_breaker_fail_rate_threshold()
    breaker_state = {
        "consecutive_fails": 0,
        "tripped": False,
        "successes": 0,
        "failures": 0,
    }

    def _record_chunk_outcome(success: bool) -> None:
        if success:
            breaker_state["successes"] += 1
            breaker_state["consecutive_fails"] = 0
        else:
            breaker_state["failures"] += 1
            breaker_state["consecutive_fails"] += 1
        # Live-update the public progress fields the frontend monitor polls.
        _done = breaker_state["successes"] + breaker_state["failures"]
        _total = len(chunks)
        _set_progress(
            job_id,
            progress_chunks_done=_done,
            progress_chunks_failed=breaker_state["failures"],
            progress_message=(
                f"Fetching pre + post Sentinel-2 bands · {_done} / {_total} chunks"
                + (f" · {breaker_state['failures']} failed" if breaker_state["failures"] else "")
            ),
        )
        if success:
            return
        if breaker_state["tripped"]:
            return
        if (
            _BREAKER_THRESHOLD > 0
            and breaker_state["consecutive_fails"] >= _BREAKER_THRESHOLD
        ):
            breaker_state["tripped"] = True
            logger.warning(
                "circuit breaker TRIPPED (consecutive) — pre/post chunked",
            )
            return
        if should_trip_fractional(
            failures=breaker_state["failures"],
            successes=breaker_state["successes"],
            min_total_fails=_BREAKER_MIN_TOTAL,
            rate_threshold=_BREAKER_RATE,
        ):
            breaker_state["tripped"] = True
            logger.warning("circuit breaker TRIPPED (fractional) — pre/post chunked")

    chunk_t0 = time.time()
    chunks_done_counter = {"n": 0}

    async def _process_chunk(ch_idx: int, chunk_bbox: BBox) -> Any:
        t_chunk_start = time.time()
        # Fetch pre + post stacks in parallel so both halves of the chunk's
        # network bill complete in roughly one stack's wall time, not two.
        try:
            pre_stack, post_stack = await asyncio.gather(
                fetch_s2_chunk_stack(
                    chunk_bbox=chunk_bbox,
                    period_scenes=pre_scenes,
                    target_gsd_m=target_gsd_m,
                    pinned_crs=pinned_crs,
                ),
                fetch_s2_chunk_stack(
                    chunk_bbox=chunk_bbox,
                    period_scenes=post_scenes,
                    target_gsd_m=target_gsd_m,
                    pinned_crs=pinned_crs,
                ),
            )
        except SentinelFetchError as e:
            chunks_done_counter["n"] += 1
            logger.warning(
                "pre/post chunk %d/%d skipped (fetch): %s",
                ch_idx + 1, len(chunks), e,
            )
            return _CHUNK_FAIL
        t_after_fetch = time.time()

        # Pre and post may differ in H, W by 1 px due to floating-point
        # bounds rounding. Crop both to the common extent before stacking.
        ph, pw = pre_stack.stack.shape[:2]
        qh, qw = post_stack.stack.shape[:2]
        ch = min(ph, qh)
        cw = min(pw, qw)
        ch4 = (ch // effective_patch) * effective_patch
        cw4 = (cw // effective_patch) * effective_patch
        if ch4 == 0 or cw4 == 0:
            logger.warning(
                "pre/post chunk %d/%d too small for patch=%d (pre=%s post=%s)",
                ch_idx + 1, len(chunks), effective_patch,
                pre_stack.stack.shape, post_stack.stack.shape,
            )
            return _CHUNK_FAIL

        pre_image = stack_to_bhwtc(pre_stack.stack[:ch4, :cw4, :, :])
        post_image = stack_to_bhwtc(post_stack.stack[:ch4, :cw4, :, :])

        try:
            # Sliding-window mode: tile the pre+post pair into
            # ``window_size`` chunks and run the conv-pool-fc head per
            # window. Turns ForestLossDriver's "one driver class per
            # 5km chunk" into "one class per ~64 px window" — matches
            # the head's training-time predict_window_px and gives
            # users meaningful spatial detail across the AOI.
            if sliding_window:
                chunk_result = await asyncio.to_thread(
                    olmoearth_model.run_ft_pre_post_tiled_inference,
                    model,
                    pre_image,
                    post_image,
                    pre_stack.timestamps,
                    post_stack.timestamps,
                    window_size,
                    effective_patch,
                    device,
                )
            else:
                chunk_result = await asyncio.to_thread(
                    olmoearth_model.run_ft_pre_post_inference,
                    model,
                    pre_image,
                    post_image,
                    pre_stack.timestamps,
                    post_stack.timestamps,
                    effective_patch,
                    device,
                )
        except Exception as e:
            logger.warning("pre/post chunk %d/%d inference failed: %s", ch_idx + 1, len(chunks), e)
            return _CHUNK_FAIL

        chunk_scalar_hi = np.repeat(
            np.repeat(chunk_result.scalar, effective_patch, axis=0),
            effective_patch, axis=1,
        )
        chunk_class_hi: np.ndarray | None = None
        if chunk_result.class_raster is not None:
            chunk_class_hi = np.repeat(
                np.repeat(chunk_result.class_raster, effective_patch, axis=0),
                effective_patch, axis=1,
            )

        chunk_west = pre_stack.transform.c
        chunk_north = pre_stack.transform.f
        col_offset = int(round((chunk_west - global_transform.c) / target_gsd_m))
        row_offset = int(round((global_transform.f - chunk_north) / target_gsd_m))

        if row_offset < 0 or col_offset < 0:
            logger.warning(
                "pre/post chunk %d/%d negative offset (row=%d col=%d) — skipping",
                ch_idx + 1, len(chunks), row_offset, col_offset,
            )
            return _CHUNK_FAIL
        # Same paste-window adjustment as the single-stack chunked path:
        # sliding-window may emit a smaller class raster than the chunk's
        # native footprint when window_size doesn't divide the chunk evenly.
        chunk_native_h, chunk_native_w = chunk_scalar_hi.shape[:2]
        h_to_paste = min(chunk_native_h, global_h - row_offset)
        w_to_paste = min(chunk_native_w, global_w - col_offset)
        if h_to_paste <= 0 or w_to_paste <= 0:
            return _CHUNK_FAIL

        chunks_done_counter["n"] += 1
        t_total = time.time() - t_chunk_start
        t_fetch = t_after_fetch - t_chunk_start
        logger.info(
            "pre/post chunk %d/%d done in %.1fs (fetch=%.1fs, fwd=%.1fs)",
            ch_idx + 1, len(chunks), t_total, t_fetch, t_total - t_fetch,
        )
        return {
            "scalar_hi": chunk_scalar_hi,
            "class_hi": chunk_class_hi,
            "row_offset": row_offset,
            "col_offset": col_offset,
            "h_to_paste": h_to_paste,
            "w_to_paste": w_to_paste,
            "result": chunk_result,
        }

    async def _process_chunk_bounded(ch_idx: int, chunk_bbox: BBox) -> Any:
        if breaker_state["tripped"]:
            return _CHUNK_FAIL
        async with chunk_sem:
            if breaker_state["tripped"]:
                return _CHUNK_FAIL
            ram_ok, ram_status = await asyncio.to_thread(chunk_ram_ok)
            if not ram_ok:
                chunks_done_counter["n"] += 1
                logger.warning(
                    "pre/post chunk %d/%d skipped — RAM gate (%s)",
                    ch_idx + 1, len(chunks), ram_status.describe(),
                )
                _record_chunk_outcome(success=False)
                return _CHUNK_FAIL
            try:
                result = await asyncio.wait_for(
                    _process_chunk(ch_idx, chunk_bbox), timeout=_CHUNK_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                chunks_done_counter["n"] += 1
                logger.warning("pre/post chunk %d/%d TIMED OUT", ch_idx + 1, len(chunks))
                _record_chunk_outcome(success=False)
                return _CHUNK_FAIL
            else:
                _record_chunk_outcome(success=result is not _CHUNK_FAIL)
                return result

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(30.0)
            done = chunks_done_counter["n"]
            elapsed = time.time() - chunk_t0
            logger.info(
                "pre/post heartbeat: %d/%d chunks done, %.0fs elapsed",
                done, len(chunks), elapsed,
            )

    heartbeat_task = asyncio.create_task(_heartbeat())

    async def _gather_chunks():
        return await asyncio.gather(
            *[_process_chunk_bounded(i, c) for i, c in enumerate(chunks)]
        )

    gather_task = asyncio.create_task(_gather_chunks())
    disconnect_task = (
        asyncio.create_task(_watch_for_disconnect(disconnect_check, gather_task))
        if disconnect_check is not None else None
    )
    try:
        try:
            chunk_records = await gather_task
        except asyncio.CancelledError:
            if (
                disconnect_task is not None
                and disconnect_task.done()
                and not disconnect_task.cancelled()
            ):
                raise ClientDisconnectedError(
                    "client closed connection during pre/post inference; "
                    f"in-flight chunks cancelled "
                    f"(processed={breaker_state['successes']}, "
                    f"failed={breaker_state['failures']}, total={len(chunks)})"
                )
            raise
    finally:
        heartbeat_task.cancel()
        if disconnect_task is not None:
            disconnect_task.cancel()
        for _t in (heartbeat_task, disconnect_task):
            if _t is None:
                continue
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass

    if breaker_state["tripped"]:
        raise CircuitBreakerTrippedError(
            processed=breaker_state["successes"],
            failed=breaker_state["failures"],
            total=len(chunks),
            threshold=_BREAKER_THRESHOLD,
        )

    last_chunk_result = None
    chunks_processed = 0
    chunks_failed = 0
    for record in chunk_records:
        if record is _CHUNK_FAIL:
            chunks_failed += 1
            continue
        global_scalar_raster[
            record["row_offset"]:record["row_offset"] + record["h_to_paste"],
            record["col_offset"]:record["col_offset"] + record["w_to_paste"],
        ] = record["scalar_hi"][:record["h_to_paste"], :record["w_to_paste"]]
        if record["class_hi"] is not None:
            global_class_raster[
                record["row_offset"]:record["row_offset"] + record["h_to_paste"],
                record["col_offset"]:record["col_offset"] + record["w_to_paste"],
            ] = record["class_hi"][:record["h_to_paste"], :record["w_to_paste"]]
        last_chunk_result = record["result"]
        chunks_processed += 1

    if chunks_processed == 0:
        raise SentinelFetchError(
            f"all {len(chunks)} pre/post chunks failed during inference"
        )
    assert last_chunk_result is not None

    final_class_raster = global_class_raster if (
        last_chunk_result.class_raster is not None
    ) else None
    colormap = last_chunk_result.colormap
    legend = _build_ft_legend(last_chunk_result)

    present_class_ids: list[int] | None = None
    if final_class_raster is not None:
        uniq = np.unique(final_class_raster).astype(int)
        uniq = uniq[(uniq >= 0) & (uniq < last_chunk_result.num_classes)]
        present_class_ids = uniq.tolist()[:256]

    # Most recent post-event scene as the "primary" scene metadata (matches
    # the user's mental model — "what does the post-event imagery show").
    recent_post_idx = next(
        (i for i in range(n_post - 1, -1, -1) if post_scenes[i].scene is not None),
        n_post - 1,
    )
    recent_scene = post_scenes[recent_post_idx].scene or {}

    return {
        "raster_transform": global_transform,
        "raster_crs": pinned_crs,
        "raster_height": int(global_h),
        "raster_width": int(global_w),
        "scene_id": recent_scene.get("id"),
        "scene_datetime": recent_scene.get("datetime"),
        "scene_cloud_cover": recent_scene.get("eo:cloud_cover")
            or recent_scene.get("cloud_cover"),
        "patch_size": effective_patch,
        "sliding_window": False,
        "window_size": None,
        "temporal_stack": {
            "n_periods": n_pre + n_post,
            "n_pre": n_pre,
            "n_post": n_post,
            "event_date": event_date,
            "periods_used": int(
                sum(1 for ps in list(pre_scenes) + list(post_scenes) if ps.scene is not None)
            ),
            "periods_skipped": int(
                sum(1 for ps in list(pre_scenes) + list(post_scenes) if ps.scene is None)
            ),
            "pre_scene_ids": [
                ps.scene.get("id") if ps.scene else None for ps in pre_scenes
            ],
            "pre_scene_datetimes": [
                ps.scene.get("datetime") if ps.scene else None for ps in pre_scenes
            ],
            "post_scene_ids": [
                ps.scene.get("id") if ps.scene else None for ps in post_scenes
            ],
            "post_scene_datetimes": [
                ps.scene.get("datetime") if ps.scene else None for ps in post_scenes
            ],
            "chunked": True,
            "chunk_size_m": chunk_size_m,
            "chunks_total": len(chunks),
            "chunks_processed": chunks_processed,
            "chunks_failed": chunks_failed,
            "target_gsd_m": target_gsd_m,
            "pre_post_split": True,
        },
        "scalar_raster": global_scalar_raster,
        "task_type": last_chunk_result.task_type,
        "num_classes": last_chunk_result.num_classes,
        "class_names": last_chunk_result.class_names,
        "class_names_tentative": last_chunk_result.class_names_tentative,
        "class_raster": final_class_raster,
        "class_probs": (
            last_chunk_result.class_probs.tolist()
            if last_chunk_result.class_probs is not None else None
        ),
        "present_class_ids": present_class_ids,
        "prediction_value": last_chunk_result.prediction_value,
        "units": last_chunk_result.units,
        "decoder_key": last_chunk_result.decoder_key,
        "colormap_override": colormap,
        "legend_override": legend,
    }


# ---------------------------------------------------------------------------
# Embedding export — matches the Ai2 OlmoEarth Studio "custom embedding
# exports" feature. Runs the BASE encoder (Nano/Tiny/Base/Large) over the
# chunked AOI pipeline, stitches per-patch embeddings into a single global
# raster, quantizes float32 → int8 using olmoearth_pretrain's AlphaEarth-
# compatible scheme, and writes a COG with one band per embedding
# dimension.
#
# Output is a drop-in replacement for Studio's embeddings COG: same int8
# layout, same nodata convention (-128), same quantize/dequantize
# roundtrip via olmoearth_pretrain.evals.embedding_transforms.
# ---------------------------------------------------------------------------


@_with_global_job_lock
async def _run_chunked_embedding_export(
    bbox: BBox,
    model: Any,                  # torch.nn.Module — Any to avoid module-level import
    device: Any,
    model_repo_id: str,
    date_range: str,
    n_periods: int = 12,
    period_days: int = 30,
    time_offset_days: int = 0,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    patch_size: int = 4,
    modality: str = "sentinel2_l2a",
    disconnect_check=None,
    return_float: bool = False,
) -> dict[str, Any]:
    """Chunked embedding export. Mirrors _run_chunked_aoi_inference but:

      * Runs the BASE encoder (no FT head) per chunk and keeps the full
        per-patch embedding tensor instead of the PCA-reduced scalar.
      * Stitches at **patch resolution** (global_h/patch × global_w/patch)
        since the encoder collapses every ``patch_size`` input pixels to
        one token.
      * Default path: quantizes the final float32 tensor to int8 via the
        AlphaEarth-compatible scheme from olmoearth_pretrain. Returns
        ``{embedding_int8, transform, crs, ...}`` for COG serialization.
      * ``return_float=True``: skip the quantize step and return
        ``{embedding_float32, nodata_mask, transform, crs, ...}``. Used
        by in-process embedding tools (PCA false-color, similarity
        search, few-shot classify) that need full precision and don't
        need to ship a COG to the client.
    """
    # Safety precheck FIRST — same rationale as _run_chunked_aoi_inference.
    # Embedding export peaks at a similar memory footprint (chunked fetch
    # + stitched global raster + int8 quantization buffer), so it gets
    # the same guard.
    check_memory_or_raise()

    # Reuse the same AOI planning + scene search + grid-resolution
    # machinery as the FT pipeline so cache hits, parallelism, timeouts,
    # heartbeats all apply uniformly.
    chunks = plan_chunks(bbox, chunk_size_m=chunk_size_m)
    # AOI size guardrail — same rationale as _run_chunked_aoi_inference.
    _mid_lat = (bbox.north + bbox.south) / 2.0
    _m_per_deg_lon = 111_000.0 * math.cos(math.radians(_mid_lat))
    _aoi_area_km2 = (
        abs(bbox.east - bbox.west) * _m_per_deg_lon
        * abs(bbox.north - bbox.south) * 111_000.0
    ) / 1e6
    check_aoi_size_or_raise(
        chunks=len(chunks),
        chunk_size_m=chunk_size_m,
        aoi_area_km2=_aoi_area_km2,
    )
    logger.info(
        "chunked embedding export for %s: %d chunks (%d m each), n_periods=%d, patch_size=%d",
        model_repo_id, len(chunks), chunk_size_m, n_periods, patch_size,
    )

    period_scenes = await fetch_aoi_period_scenes(
        bbox=bbox,
        anchor_date=date_range,
        n_periods=n_periods,
        period_days=period_days,
        time_offset_days=time_offset_days,
        max_cloud_cover=40.0,
    )
    if all(ps.scene is None for ps in period_scenes):
        raise SentinelFetchError(
            f"embedding export: AOI-wide search returned 0 usable scenes "
            f"across {n_periods} periods"
        )

    pinned_crs, global_transform, global_h, global_w = await resolve_aoi_grid(
        bbox, period_scenes, target_gsd_m=target_gsd_m,
    )

    # Snap global dimensions to patch_size so every chunk's embedding
    # tile fits into the patch-resolution global grid without off-by-one.
    global_h = (global_h // patch_size) * patch_size
    global_w = (global_w // patch_size) * patch_size
    if global_h == 0 or global_w == 0:
        raise ValueError(
            f"embedding export: AOI too small for patch_size={patch_size} "
            f"(global_h={global_h}, global_w={global_w})"
        )
    global_h_patch = global_h // patch_size
    global_w_patch = global_w // patch_size

    # Global embedding buffer at patch resolution. D is unknown until the
    # first successful chunk returns — allocate lazily on first hit.
    global_embedding: np.ndarray | None = None
    embedding_dim: int | None = None

    chunk_t0 = time.time()
    chunks_done_counter = {"n": 0}
    # Bumped 180 → 300 s. With slow PC days (some band reads take 30 s
    # before failing over to the per-period second-pass retry) and 4
    # chunks competing for the asyncio thread pool, 180 s was too tight
    # — chunks all hit the wall before any could finish a full
    # fetch + encoder forward. 300 s gives a real cold-start budget
    # while still bailing fast enough that the breaker can trip on
    # genuinely dead networks.
    _CHUNK_TIMEOUT_S = 300.0
    _CHUNK_FAIL = object()
    chunk_sem = asyncio.Semaphore(4)

    # Circuit breaker — mirrors _run_chunked_aoi_inference. Two trip rules:
    # (a) consecutive fails (bursty drops) and (b) fractional fail rate
    # (slow-burn 50 % flake networks).
    _BREAKER_THRESHOLD = circuit_breaker_threshold()
    _BREAKER_MIN_TOTAL = circuit_breaker_min_total_fails()
    _BREAKER_RATE = circuit_breaker_fail_rate_threshold()
    breaker_state = {
        "consecutive_fails": 0,
        "tripped": False,
        "successes": 0,
        "failures": 0,
    }

    def _record_chunk_outcome(success: bool) -> None:
        if success:
            breaker_state["successes"] += 1
            breaker_state["consecutive_fails"] = 0
            return
        breaker_state["failures"] += 1
        breaker_state["consecutive_fails"] += 1
        if breaker_state["tripped"]:
            return
        if (
            _BREAKER_THRESHOLD > 0
            and breaker_state["consecutive_fails"] >= _BREAKER_THRESHOLD
        ):
            breaker_state["tripped"] = True
            logger.warning(
                "circuit breaker TRIPPED (embedding export, consecutive) — "
                "%d chunks failed in a row, aborting remaining work",
                _BREAKER_THRESHOLD,
            )
            return
        if should_trip_fractional(
            failures=breaker_state["failures"],
            successes=breaker_state["successes"],
            min_total_fails=_BREAKER_MIN_TOTAL,
            rate_threshold=_BREAKER_RATE,
        ):
            total = breaker_state["successes"] + breaker_state["failures"]
            breaker_state["tripped"] = True
            logger.warning(
                "circuit breaker TRIPPED (embedding export, fractional) — "
                "%d failures out of %d chunks (%.0f%% > %.0f%% threshold)",
                breaker_state["failures"], total,
                100 * breaker_state["failures"] / total,
                100 * _BREAKER_RATE,
            )

    async def _process_chunk(ch_idx: int, chunk_bbox: BBox) -> Any:
        async with chunk_sem:
            t_chunk_start = time.time()
            try:
                stack_result = await fetch_s2_chunk_stack(
                    chunk_bbox=chunk_bbox,
                    period_scenes=period_scenes,
                    target_gsd_m=target_gsd_m,
                    pinned_crs=pinned_crs,
                )
            except SentinelFetchError as e:
                chunks_done_counter["n"] += 1
                elapsed = time.time() - chunk_t0
                logger.warning(
                    "chunk %d/%d skipped (fetch, %.1fs): %s [done=%d/%d, elapsed=%.0fs]",
                    ch_idx + 1, len(chunks), time.time() - t_chunk_start, e,
                    chunks_done_counter["n"], len(chunks), elapsed,
                )
                return _CHUNK_FAIL

            ch, cw, _, _ = stack_result.stack.shape
            ch4 = (ch // patch_size) * patch_size
            cw4 = (cw // patch_size) * patch_size
            if ch4 == 0 or cw4 == 0:
                chunks_done_counter["n"] += 1
                return _CHUNK_FAIL

            chunk_image = stack_to_bhwtc(stack_result.stack[:ch4, :cw4, :, :])
            try:
                # BASE encoder — returns the full per-patch embedding.
                result = await asyncio.to_thread(
                    olmoearth_model.run_s2_inference,
                    model, chunk_image, stack_result.timestamps,
                    patch_size, device,
                )
            except Exception as e:
                chunks_done_counter["n"] += 1
                logger.warning(
                    "chunk %d/%d inference failed: %s", ch_idx + 1, len(chunks), e,
                )
                return _CHUNK_FAIL

            # Paste offset in the global PATCH grid (not pixel grid).
            chunk_west = stack_result.transform.c
            chunk_north = stack_result.transform.f
            px_col_offset = int(round((chunk_west - global_transform.c) / target_gsd_m))
            px_row_offset = int(round((global_transform.f - chunk_north) / target_gsd_m))
            patch_col_offset = px_col_offset // patch_size
            patch_row_offset = px_row_offset // patch_size

            emb_h, emb_w, emb_d = result.embedding.shape
            h_to_paste = min(emb_h, global_h_patch - patch_row_offset)
            w_to_paste = min(emb_w, global_w_patch - patch_col_offset)
            if (
                patch_row_offset < 0 or patch_col_offset < 0
                or h_to_paste <= 0 or w_to_paste <= 0
            ):
                chunks_done_counter["n"] += 1
                return _CHUNK_FAIL

            chunks_done_counter["n"] += 1
            elapsed = time.time() - chunk_t0
            done = chunks_done_counter["n"]
            t_total = time.time() - t_chunk_start
            logger.info(
                "chunk %d/%d done in %.1fs (emb=%s) [done=%d/%d, elapsed=%.0fs]",
                ch_idx + 1, len(chunks), t_total, (emb_h, emb_w, emb_d),
                done, len(chunks), elapsed,
            )
            return {
                "embedding": result.embedding,       # (h, w, D) float32
                "patch_row_offset": patch_row_offset,
                "patch_col_offset": patch_col_offset,
                "h_to_paste": h_to_paste,
                "w_to_paste": w_to_paste,
                "embedding_dim": emb_d,
            }

    async def _process_chunk_bounded(ch_idx: int, chunk_bbox: BBox) -> Any:
        # Short-circuit once the breaker trips — same pattern as the AOI
        # orchestrator. Queued chunks exit without trying.
        if breaker_state["tripped"]:
            return _CHUNK_FAIL
        # Per-chunk RAM gate (mirrors the AOI orchestrator). Stops the next
        # fetch + encoder forward when free RAM has dropped below the OS's
        # swap-thrash floor since the submit-time precheck. Cheap psutil
        # call wrapped in to_thread because it's a syscall and we don't
        # want to block the event loop even briefly under contention.
        ram_ok, ram_status = await asyncio.to_thread(chunk_ram_ok)
        if not ram_ok:
            chunks_done_counter["n"] += 1
            logger.warning(
                "chunk %d/%d skipped — per-chunk RAM gate failed (%s)",
                ch_idx + 1, len(chunks), ram_status.describe(),
            )
            _record_chunk_outcome(success=False)
            return _CHUNK_FAIL
        try:
            result = await asyncio.wait_for(
                _process_chunk(ch_idx, chunk_bbox), timeout=_CHUNK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            chunks_done_counter["n"] += 1
            logger.warning(
                "chunk %d/%d TIMED OUT after %.0fs (embedding export)",
                ch_idx + 1, len(chunks), _CHUNK_TIMEOUT_S,
            )
            _record_chunk_outcome(success=False)
            return _CHUNK_FAIL
        _record_chunk_outcome(success=result is not _CHUNK_FAIL)
        return result

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(30.0)
            done = chunks_done_counter["n"]
            elapsed = time.time() - chunk_t0
            logger.info(
                "heartbeat(embed): %d/%d chunks done, %.0fs elapsed",
                done, len(chunks), elapsed,
            )

    heartbeat_task = asyncio.create_task(_heartbeat())

    async def _gather_chunks():
        return await asyncio.gather(
            *[_process_chunk_bounded(i, c) for i, c in enumerate(chunks)]
        )

    gather_task = asyncio.create_task(_gather_chunks())
    disconnect_task = (
        asyncio.create_task(_watch_for_disconnect(disconnect_check, gather_task))
        if disconnect_check is not None else None
    )
    try:
        try:
            chunk_records = await gather_task
        except asyncio.CancelledError:
            if (
                disconnect_task is not None
                and disconnect_task.done()
                and not disconnect_task.cancelled()
            ):
                raise ClientDisconnectedError(
                    "client closed connection during embedding export; "
                    f"in-flight chunks cancelled "
                    f"(processed={breaker_state['successes']}, "
                    f"failed={breaker_state['failures']}, total={len(chunks)})"
                )
            raise
    finally:
        heartbeat_task.cancel()
        if disconnect_task is not None:
            disconnect_task.cancel()
        for _t in (heartbeat_task, disconnect_task):
            if _t is None:
                continue
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass

    # Breaker tripped — raise before stitching so the router surfaces a
    # clean 503 instead of a misleading "all chunks failed" error.
    if breaker_state["tripped"]:
        raise CircuitBreakerTrippedError(
            processed=breaker_state["successes"],
            failed=breaker_state["failures"],
            total=len(chunks),
            threshold=_BREAKER_THRESHOLD,
        )

    # Stitch: allocate global buffer lazily on first successful chunk so
    # we only pay memory for the actual embedding_dim (Nano=128, Tiny=192,
    # Base=768, Large=1024).
    chunks_processed = 0
    chunks_failed = 0
    for record in chunk_records:
        if record is _CHUNK_FAIL:
            chunks_failed += 1
            continue
        if global_embedding is None:
            embedding_dim = record["embedding_dim"]
            global_embedding = np.zeros(
                (global_h_patch, global_w_patch, embedding_dim), dtype=np.float32,
            )
        r0 = record["patch_row_offset"]
        c0 = record["patch_col_offset"]
        h = record["h_to_paste"]
        w = record["w_to_paste"]
        global_embedding[r0:r0 + h, c0:c0 + w, :] = record["embedding"][:h, :w, :]
        chunks_processed += 1

    if chunks_processed == 0 or global_embedding is None:
        raise SentinelFetchError(
            f"embedding export: all {len(chunks)} chunks failed"
        )

    # Track the fill mask BEFORE quantization so downstream tools know
    # which pixels have no data. A zero vector after int8 quantization is
    # genuinely 0 (middle of int8 range), indistinguishable from the
    # untouched-chunk sentinel; we keep it as a separate bool array.
    nodata_mask = ~np.any(global_embedding != 0, axis=-1)

    # Patch-resolution transform: pixel size = target_gsd_m * patch_size.
    import rasterio as _rio_mod  # noqa: PLC0415
    patch_transform = global_transform * _rio_mod.Affine.scale(patch_size, patch_size)

    # Shared metadata returned regardless of quantization mode.
    shared = {
        "embedding_dim": int(embedding_dim),
        "transform": patch_transform,
        "crs": pinned_crs,
        "patch_size": patch_size,
        "target_gsd_m": target_gsd_m,
        "chunks_processed": chunks_processed,
        "chunks_failed": chunks_failed,
        "chunks_total": len(chunks),
        "n_periods": n_periods,
        "period_days": period_days,
        "modality": modality,
        "model_repo_id": model_repo_id,
        "scene_ids": [
            ps.scene.get("id") if ps.scene else None for ps in period_scenes
        ],
        "scene_datetimes": [
            ps.scene.get("datetime") if ps.scene else None for ps in period_scenes
        ],
    }

    if return_float:
        # Float32 path — downstream tools (PCA-RGB, similarity search,
        # few-shot classify) need the full-precision embedding + nodata
        # mask. The COG exporter is NOT used on this path.
        logger.info(
            "embedding computed (float32): %d/%d chunks ok, shape=%s, dim=%d, patch_gsd=%sm",
            chunks_processed, len(chunks),
            global_embedding.shape, embedding_dim, target_gsd_m * patch_size,
        )
        return {
            **shared,
            "embedding_float32": global_embedding,  # (H_patch, W_patch, D) float32
            "nodata_mask": nodata_mask,              # (H_patch, W_patch) bool
        }

    # AlphaEarth-compatible int8 quantization (sqrt + scale to ±127, -128 = nodata)
    # Reserve -128 for patches nobody filled (empty chunks, edges).
    import torch as _torch  # noqa: PLC0415
    from olmoearth_pretrain.evals.embedding_transforms import (  # noqa: PLC0415
        quantize_embeddings,
    )
    emb_tensor = _torch.from_numpy(global_embedding)
    quantized = quantize_embeddings(emb_tensor).numpy().astype(np.int8)
    if nodata_mask.any():
        # Broadcast (-128) across all bands for nodata pixels.
        quantized[nodata_mask, :] = -128

    logger.info(
        "embedding export: %d/%d chunks ok, shape=%s, dim=%d, patch_gsd=%sm",
        chunks_processed, len(chunks),
        quantized.shape, embedding_dim, target_gsd_m * patch_size,
    )

    return {
        **shared,
        "embedding_int8": quantized,       # (H_patch, W_patch, D) int8
    }


async def run_embedding_tool_pca_rgb(
    bbox: BBox,
    model_repo_id: str,
    date_range: str = "2024-04-01/2024-10-01",
    # Default reduced from 12 → 3 periods for the embedding tools: PCA
    # false-color visualisation just needs enough temporal context to
    # extract spatial structure, not a full year. 3 periods × 30 days =
    # ~90 days of S2 history. Cuts band fetches 4× (3×12=36 reads/chunk
    # vs 12×12=144) and encoder forward time roughly proportionally,
    # turning a 3-min PCA into ~45-60 s. Operators can still pass
    # higher n_periods explicitly when temporal richness matters.
    n_periods: int = 3,
    period_days: int = 30,
    time_offset_days: int = 0,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    patch_size: int = 4,
    disconnect_check=None,
) -> dict[str, Any]:
    """Compute embeddings for an AOI + render as a PCA false-color map layer.

    End-to-end pipeline for the first embedding tool exposed in the UI:

      1. Chunked fetch + base encoder forward (reuses
         ``_run_chunked_embedding_export(return_float=True)``)
      2. Project to top-3 principal components via
         ``olmoearth_model.pca_to_rgb`` → ``(H, W, 3)`` uint8
      3. Register as a pytorch tile job so the existing XYZ tile route +
         ImageryLayer flow on the frontend Just Works.

    Returns the same shape as ``start_inference`` (job_id + tile_url +
    legend + metadata) so the frontend treats it identically to any
    other map-producing inference call.
    """
    if model_repo_id not in {
        "allenai/OlmoEarth-v1-Nano",
        "allenai/OlmoEarth-v1-Tiny",
        "allenai/OlmoEarth-v1-Base",
        "allenai/OlmoEarth-v1-Large",
    }:
        raise ValueError(
            f"PCA false-color is an embedding tool — only base encoders "
            f"(Nano/Tiny/Base/Large) produce raw embeddings. Got {model_repo_id!r}."
        )

    # Spec mirrors start_inference's shape so job_id stays stable across
    # the same AOI+model+date combo — reruns hit the _jobs dedupe path.
    spec = {
        "bbox": bbox.model_dump(),
        "model_repo_id": model_repo_id,
        "date_range": date_range,
        "tool": "embedding_pca_rgb",
        "n_periods": n_periods,
        "period_days": period_days,
        "target_gsd_m": target_gsd_m,
        "patch_size": patch_size,
    }
    job_id = _make_job_id(spec)

    async with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing is not None and existing.get("status") == "ready" and existing.get("kind") == "pytorch":
            return _build_response(existing)
        if existing is not None and existing.get("status") == "running":
            logger.warning(
                "PCA-RGB %s already running — returning pending stub (dedup)", job_id,
            )
            return _build_response(existing)
        _jobs[job_id] = {
            "job_id": job_id,
            "spec": spec,
            "status": "running",
            "kind": "pending",
            "colormap": "pca_rgb",
            "started_ts": time.time(),
        }

    # Load encoder lazily — cached across requests by olmoearth_model.
    model, device = await asyncio.to_thread(olmoearth_model.load_encoder, model_repo_id)

    try:
        export_result = await _run_chunked_embedding_export(
            bbox=bbox,
            model=model,
            device=device,
            model_repo_id=model_repo_id,
            date_range=date_range,
            n_periods=n_periods,
            period_days=period_days,
            time_offset_days=time_offset_days,
            chunk_size_m=chunk_size_m,
            target_gsd_m=target_gsd_m,
            patch_size=patch_size,
            return_float=True,     # keep float32 for PCA
            disconnect_check=disconnect_check,
        )
    except Exception as e:
        async with _jobs_lock:
            _jobs[job_id].update(
                status="ready", kind="stub",
                stub_reason=f"{type(e).__name__}: {e}"[:500],
            )
        raise

    # Run PCA on CPU — SVD is cheap for typical N=H×W <= 16k. Spin into a
    # thread so the event loop keeps breathing for other endpoints.
    global_embedding = export_result["embedding_float32"]
    rgb_raster = await asyncio.to_thread(olmoearth_model.pca_to_rgb, global_embedding)

    # Register the job shape the tile renderer expects. rgb_raster triggers
    # the new RGB path in _render_pytorch_tile (skips colormap).
    # Surface a recent-scene marker for the UI "scene id" chip.
    scene_ids = export_result.get("scene_ids") or []
    scene_datetimes = export_result.get("scene_datetimes") or []
    recent_scene_id = next((s for s in reversed(scene_ids) if s is not None), None)
    recent_scene_dt = next((s for s in reversed(scene_datetimes) if s is not None), None)

    async with _jobs_lock:
        _jobs[job_id].update(
            status="ready",
            kind="pytorch",
            task_type="embedding_pca_rgb",
            rgb_raster=rgb_raster,
            # scalar_raster kept for any downstream code that expects it —
            # the tile renderer's RGB path takes precedence when rgb_raster
            # is present.
            scalar_raster=np.zeros(rgb_raster.shape[:2], dtype=np.float32),
            raster_transform=export_result["transform"],
            raster_crs=export_result["crs"],
            raster_height=int(rgb_raster.shape[0]),
            raster_width=int(rgb_raster.shape[1]),
            scene_id=recent_scene_id,
            scene_datetime=recent_scene_dt,
            patch_size=export_result["patch_size"],
            embedding_dim=export_result["embedding_dim"],
            legend={
                "kind": "rgb",
                "label": "OlmoEarth embedding PCA false-color",
                "note": (
                    "Top-3 principal components of the per-patch embedding "
                    "mapped to RGB. Similar embeddings get similar colors — "
                    "works globally, no labels required."
                ),
            },
            temporal_stack={
                "n_periods": n_periods,
                "chunks_total": export_result["chunks_total"],
                "chunks_processed": export_result["chunks_processed"],
                "chunks_failed": export_result["chunks_failed"],
                "scene_ids": scene_ids,
                "scene_datetimes": scene_datetimes,
            },
        )
        return _build_response(_jobs[job_id])


async def run_embedding_tool_similarity(
    bbox: BBox,
    model_repo_id: str,
    query_lon: float | None = None,
    query_lat: float | None = None,
    window_px: int = 1,
    date_range: str = "2024-04-01/2024-10-01",
    # Same rationale as the PCA tool — 3 periods is enough to compute a
    # meaningful similarity signal without burning 4× the network.
    n_periods: int = 3,
    period_days: int = 30,
    time_offset_days: int = 0,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    patch_size: int = 4,
    disconnect_check=None,
) -> dict[str, Any]:
    """Compute embeddings + render cosine-similarity heatmap vs. a query point.

    Second in-UI embedding tool. Same chunked pipeline as PCA false-color
    — the difference is post-processing:

      1. Chunked fetch + base encoder forward (``return_float=True``)
      2. Resolve query lon/lat → global patch grid (row, col)
      3. Extract query vector (mean over ``window_px`` × ``window_px``
         window for noise robustness — single-pixel queries are spiky)
      4. Cosine similarity for every patch → scalar raster in [0, 1]
      5. Register as pytorch tile job with the "similarity" colormap

    When ``query_lon`` / ``query_lat`` are ``None`` the AOI center is
    used — gives users a one-click "what else looks like the middle of
    this area" demo without needing a pixel-picking UI yet.
    """
    if model_repo_id not in {
        "allenai/OlmoEarth-v1-Nano",
        "allenai/OlmoEarth-v1-Tiny",
        "allenai/OlmoEarth-v1-Base",
        "allenai/OlmoEarth-v1-Large",
    }:
        raise ValueError(
            f"Similarity search needs a base encoder. Got {model_repo_id!r}."
        )

    # Default query = AOI center. Works well as a smoke-test / first-
    # click demo; dedicated pixel-picker UX can override these later.
    if query_lon is None:
        query_lon = (bbox.west + bbox.east) / 2.0
    if query_lat is None:
        query_lat = (bbox.south + bbox.north) / 2.0

    spec = {
        "bbox": bbox.model_dump(),
        "model_repo_id": model_repo_id,
        "date_range": date_range,
        "tool": "embedding_similarity",
        "query_lon": round(float(query_lon), 6),
        "query_lat": round(float(query_lat), 6),
        "window_px": window_px,
        "n_periods": n_periods,
        "period_days": period_days,
        "target_gsd_m": target_gsd_m,
        "patch_size": patch_size,
    }
    job_id = _make_job_id(spec)

    async with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing is not None and existing.get("status") == "ready" and existing.get("kind") == "pytorch":
            return _build_response(existing)
        if existing is not None and existing.get("status") == "running":
            logger.warning(
                "similarity %s already running — returning pending stub (dedup)", job_id,
            )
            return _build_response(existing)
        _jobs[job_id] = {
            "job_id": job_id,
            "spec": spec,
            "status": "running",
            "kind": "pending",
            "colormap": "similarity",
            "started_ts": time.time(),
        }

    model, device = await asyncio.to_thread(olmoearth_model.load_encoder, model_repo_id)

    try:
        export_result = await _run_chunked_embedding_export(
            bbox=bbox,
            model=model,
            device=device,
            model_repo_id=model_repo_id,
            date_range=date_range,
            n_periods=n_periods,
            period_days=period_days,
            time_offset_days=time_offset_days,
            chunk_size_m=chunk_size_m,
            target_gsd_m=target_gsd_m,
            patch_size=patch_size,
            return_float=True,
            disconnect_check=disconnect_check,
        )
    except Exception as e:
        async with _jobs_lock:
            _jobs[job_id].update(
                status="ready", kind="stub",
                stub_reason=f"{type(e).__name__}: {e}"[:500],
            )
        raise

    global_embedding = export_result["embedding_float32"]  # (H_patch, W_patch, D)
    patch_transform = export_result["transform"]
    pinned_crs = export_result["crs"]
    h_patch, w_patch, _ = global_embedding.shape

    # Project the WGS-84 query into the patch grid's CRS, then map to
    # (row, col) via the inverse affine.
    from rasterio.warp import transform as _transform_coords  # noqa: PLC0415
    xs, ys = _transform_coords(
        CRS.from_string("EPSG:4326"), pinned_crs, [query_lon], [query_lat],
    )
    qx, qy = float(xs[0]), float(ys[0])
    inv = ~patch_transform
    col_f, row_f = inv * (qx, qy)
    q_col = int(round(col_f))
    q_row = int(round(row_f))
    # Clamp to the grid + optionally average a small window for
    # robustness. ``window_px=1`` (default) gives a single-pixel query
    # which matches the Ai2 tutorial's basic flow.
    q_col = max(0, min(w_patch - 1, q_col))
    q_row = max(0, min(h_patch - 1, q_row))
    hw = max(1, int(window_px)) // 2
    r0 = max(0, q_row - hw)
    r1 = min(h_patch, q_row + hw + 1)
    c0 = max(0, q_col - hw)
    c1 = min(w_patch, q_col + hw + 1)
    query_vec = global_embedding[r0:r1, c0:c1].reshape(-1, global_embedding.shape[-1])
    # Drop nodata rows (all-zero) before averaging so a query near the
    # AOI edge doesn't get diluted by untouched patches.
    valid_rows = np.any(query_vec != 0, axis=-1)
    if not valid_rows.any():
        # User clicked on nodata. Fall back to the global mean so the
        # heatmap is still meaningful ("what's most average about this
        # AOI") rather than raising.
        flat = global_embedding.reshape(-1, global_embedding.shape[-1])
        valid_flat = flat[np.any(flat != 0, axis=-1)]
        if valid_flat.shape[0] == 0:
            raise SentinelFetchError(
                "similarity: no valid pixels in embedding — every chunk "
                "returned nodata"
            )
        query_vec = valid_flat.mean(axis=0)
    else:
        query_vec = query_vec[valid_rows].mean(axis=0)

    # Scalar raster in [0, 1].
    sim_raster = await asyncio.to_thread(
        olmoearth_model.cosine_similarity_map, global_embedding, query_vec,
    )

    scene_ids = export_result.get("scene_ids") or []
    scene_datetimes = export_result.get("scene_datetimes") or []
    recent_scene_id = next((s for s in reversed(scene_ids) if s is not None), None)
    recent_scene_dt = next((s for s in reversed(scene_datetimes) if s is not None), None)

    async with _jobs_lock:
        _jobs[job_id].update(
            status="ready",
            kind="pytorch",
            task_type="embedding_similarity",
            scalar_raster=sim_raster,
            raster_transform=patch_transform,
            raster_crs=pinned_crs,
            raster_height=int(h_patch),
            raster_width=int(w_patch),
            scene_id=recent_scene_id,
            scene_datetime=recent_scene_dt,
            patch_size=export_result["patch_size"],
            embedding_dim=export_result["embedding_dim"],
            # Carry the query location back to the UI so the legend can
            # show "query: 28.68°N, 80.70°W".
            similarity_query={
                "lon": round(float(query_lon), 6),
                "lat": round(float(query_lat), 6),
                "patch_row": int(q_row),
                "patch_col": int(q_col),
                "window_px": int(window_px),
            },
            temporal_stack={
                "n_periods": n_periods,
                "chunks_total": export_result["chunks_total"],
                "chunks_processed": export_result["chunks_processed"],
                "chunks_failed": export_result["chunks_failed"],
                "scene_ids": scene_ids,
                "scene_datetimes": scene_datetimes,
            },
        )
        return _build_response(_jobs[job_id])


async def run_embedding_tool_few_shot(
    bbox: BBox,
    model_repo_id: str,
    classes: list[dict[str, Any]],
    date_range: str = "2024-04-01/2024-10-01",
    n_periods: int = 3,
    period_days: int = 30,
    time_offset_days: int = 0,
    chunk_size_m: int = 5000,
    target_gsd_m: float = 10.0,
    patch_size: int = 4,
    disconnect_check=None,
) -> dict[str, Any]:
    """Few-shot semantic segmentation via nearest-prototype classification.

    Workflow:
      1. Chunked fetch + base encoder forward over the AOI
         (``return_float=True`` — same path as PCA / similarity)
      2. For each user-defined class, collect the embedding vectors at
         the user's labeled points, average → class prototype (D,)
      3. For every pixel: cosine similarity to each prototype, argmax
         → class index. Pixels where all similarities are low (< the
         "unclassified" threshold) get marked as class -1 / nodata so
         the user sees gaps for "no class is a good match" rather than
         being forced into the nearest class.
      4. Build a class raster + per-class colour legend and register
         as a pytorch job — the existing tile renderer + GeoJSON export
         work unchanged.

    The "few-shot" pattern matches the OlmoEarth tutorial's
    ``classify-from-clicks`` notebook: 1-N labelled points per class,
    no model fine-tuning, just embedding-space prototypes.

    Args:
        classes: list of dicts ``{name, color, points}``. ``points`` is
            ``[{lon, lat}, ...]`` in WGS-84. At least 2 classes, each
            with at least 1 point. Colour is a hex string used in the
            response legend.

    Notes:
      * Spec hash (job_id) deliberately includes the labelled points
        so re-clicking the same set hits the cache; adding a single
        point yields a fresh job.
      * Patch coordinates outside the AOI raster are silently dropped
        — picks near the edge that round outside the grid contribute
        no constraint rather than crashing.
    """
    if model_repo_id not in {
        "allenai/OlmoEarth-v1-Nano",
        "allenai/OlmoEarth-v1-Tiny",
        "allenai/OlmoEarth-v1-Base",
        "allenai/OlmoEarth-v1-Large",
    }:
        raise ValueError(
            f"Few-shot classify needs a base encoder. Got {model_repo_id!r}."
        )
    if len(classes) < 2:
        raise ValueError(
            f"Few-shot classify needs at least 2 classes; got {len(classes)}."
        )
    for i, c in enumerate(classes):
        if not c.get("points"):
            raise ValueError(
                f"Class {i} ({c.get('name')!r}) has 0 labelled points — "
                f"every class needs at least 1 example."
            )

    # Stable job_id: include the labelled points (rounded to 6 decimals
    # so floating-point chatter doesn't bust the cache) so reruns of
    # the same labelling hit the cached embedding pass.
    spec_classes = [
        {
            "name": str(c.get("name", f"class_{i}")),
            "color": str(c.get("color", "#888888")),
            "points": [
                {"lon": round(float(p["lon"]), 6), "lat": round(float(p["lat"]), 6)}
                for p in c.get("points") or []
            ],
        }
        for i, c in enumerate(classes)
    ]
    spec = {
        "bbox": bbox.model_dump(),
        "model_repo_id": model_repo_id,
        "date_range": date_range,
        "tool": "embedding_few_shot",
        "classes": spec_classes,
        "n_periods": n_periods,
        "period_days": period_days,
        "target_gsd_m": target_gsd_m,
        "patch_size": patch_size,
    }
    job_id = _make_job_id(spec)

    async with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing is not None and existing.get("status") == "ready" and existing.get("kind") == "pytorch":
            return _build_response(existing)
        if existing is not None and existing.get("status") == "running":
            logger.warning(
                "few-shot %s already running — returning pending stub (dedup)",
                job_id,
            )
            return _build_response(existing)
        _jobs[job_id] = {
            "job_id": job_id,
            "spec": spec,
            "status": "running",
            "kind": "pending",
            "colormap": "few_shot",
            "started_ts": time.time(),
        }

    model, device = await asyncio.to_thread(olmoearth_model.load_encoder, model_repo_id)

    try:
        export_result = await _run_chunked_embedding_export(
            bbox=bbox,
            model=model,
            device=device,
            model_repo_id=model_repo_id,
            date_range=date_range,
            n_periods=n_periods,
            period_days=period_days,
            time_offset_days=time_offset_days,
            chunk_size_m=chunk_size_m,
            target_gsd_m=target_gsd_m,
            patch_size=patch_size,
            return_float=True,
            disconnect_check=disconnect_check,
        )
    except Exception as e:
        async with _jobs_lock:
            _jobs[job_id].update(
                status="ready", kind="stub",
                stub_reason=f"{type(e).__name__}: {e}"[:500],
            )
        raise

    global_embedding = export_result["embedding_float32"]   # (H_patch, W_patch, D)
    patch_transform = export_result["transform"]
    pinned_crs = export_result["crs"]
    h_patch, w_patch, embed_dim = global_embedding.shape

    # Convert each class's labelled WGS-84 points to (row, col) in the
    # patch grid, gather the embedding vectors, average → prototype.
    # Points that round outside the AOI raster are silently dropped.
    from rasterio.warp import transform as _transform_coords  # noqa: PLC0415
    inv = ~patch_transform
    prototypes: list[np.ndarray] = []
    used_points_per_class: list[int] = []
    nodata_mask = ~np.any(global_embedding != 0, axis=-1)
    for i, c in enumerate(spec_classes):
        lons = [p["lon"] for p in c["points"]]
        lats = [p["lat"] for p in c["points"]]
        xs, ys = _transform_coords(
            CRS.from_string("EPSG:4326"), pinned_crs, lons, lats,
        )
        vecs: list[np.ndarray] = []
        for x, y in zip(xs, ys):
            col_f, row_f = inv * (float(x), float(y))
            r = int(round(row_f))
            ccol = int(round(col_f))
            if r < 0 or r >= h_patch or ccol < 0 or ccol >= w_patch:
                continue
            if nodata_mask[r, ccol]:
                continue
            vecs.append(global_embedding[r, ccol].copy())
        if not vecs:
            raise SentinelFetchError(
                f"few-shot: class {i} ({c['name']!r}) had {len(c['points'])} "
                f"labelled point(s) but none landed on a valid embedding "
                f"patch (all outside AOI or in nodata)."
            )
        prototypes.append(np.stack(vecs, axis=0).mean(axis=0))
        used_points_per_class.append(len(vecs))

    proto_matrix = np.stack(prototypes, axis=0)              # (K, D)

    # Cosine similarity between every pixel embedding and every prototype:
    # similarity = (E·P^T) / (||E|| ||P||). Argmax across prototypes
    # gives the class index per pixel; the max similarity itself is the
    # confidence (used for the scalar raster + nodata threshold).
    flat = global_embedding.reshape(h_patch * w_patch, embed_dim)
    flat_norm = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-9)
    proto_norm = proto_matrix / (np.linalg.norm(proto_matrix, axis=-1, keepdims=True) + 1e-9)
    cos_all = flat_norm @ proto_norm.T                       # (N, K) in [-1, 1]
    class_idx = cos_all.argmax(axis=-1).astype(np.int32)     # (N,)
    confidence = cos_all.max(axis=-1).astype(np.float32)     # (N,) in [-1, 1]

    # Nodata pixels: class -1 + confidence 0 so the tile renderer
    # paints them transparent / dark instead of forcing them into the
    # nearest class.
    nodata_flat = nodata_mask.reshape(-1)
    class_idx[nodata_flat] = -1
    confidence[nodata_flat] = 0.0

    # Rescale confidence to [0, 1] for the existing scalar tile path.
    scalar = ((confidence + 1.0) / 2.0).astype(np.float32)
    scalar[nodata_flat] = 0.0
    scalar_raster = scalar.reshape(h_patch, w_patch)
    class_raster = class_idx.reshape(h_patch, w_patch)

    class_names = [c["name"] for c in spec_classes]
    class_colors = [c["color"] for c in spec_classes]
    legend = {
        "kind": "classification",
        "label": "Few-shot classification",
        "classes": [
            {"index": i, "name": n, "color": col, "points_used": used_points_per_class[i]}
            for i, (n, col) in enumerate(zip(class_names, class_colors))
        ],
        "names_tentative": False,
        "colors_source": "user",
        "note": (
            "Each class's prototype is the mean embedding of the user's "
            "labelled clicks; per-pixel class is argmax cosine similarity "
            "to those prototypes. No fine-tuning involved — this is "
            "embedding-space nearest-prototype matching."
        ),
    }
    present_class_ids = sorted({int(v) for v in np.unique(class_raster) if int(v) >= 0})

    scene_ids = export_result.get("scene_ids") or []
    scene_datetimes = export_result.get("scene_datetimes") or []
    recent_scene_id = next((s for s in reversed(scene_ids) if s is not None), None)
    recent_scene_dt = next((s for s in reversed(scene_datetimes) if s is not None), None)

    async with _jobs_lock:
        _jobs[job_id].update(
            status="ready",
            kind="pytorch",
            task_type="classification",
            scalar_raster=scalar_raster,
            class_raster=class_raster,
            raster_transform=patch_transform,
            raster_crs=pinned_crs,
            raster_height=int(h_patch),
            raster_width=int(w_patch),
            scene_id=recent_scene_id,
            scene_datetime=recent_scene_dt,
            patch_size=export_result["patch_size"],
            embedding_dim=int(embed_dim),
            num_classes=len(spec_classes),
            class_names=class_names,
            class_names_tentative=False,
            class_probs=None,
            present_class_ids=present_class_ids,
            decoder_key="few_shot_prototype",
            colormap_override="few_shot",
            legend_override=legend,
            temporal_stack={
                "n_periods": n_periods,
                "chunks_total": export_result["chunks_total"],
                "chunks_processed": export_result["chunks_processed"],
                "chunks_failed": export_result["chunks_failed"],
                "scene_ids": scene_ids,
                "scene_datetimes": scene_datetimes,
                "target_gsd_m": target_gsd_m,
            },
            few_shot={
                "classes": [
                    {
                        "name": n,
                        "color": col,
                        "points_provided": len(spec_classes[i]["points"]),
                        "points_used": used_points_per_class[i],
                    }
                    for i, (n, col) in enumerate(zip(class_names, class_colors))
                ],
            },
        )
        return _build_response(_jobs[job_id])


def build_embedding_cog_bytes(result: dict[str, Any]) -> tuple[bytes, str]:
    """Serialize a stitched embedding raster as a multi-band int8 COG.

    One band per embedding dimension. nodata = -128 (matches Ai2 Studio's
    convention). Uses rasterio's COG driver when available (rasterio
    >= 1.3) so the output passes ``rio cogeo validate``; falls back to a
    tiled GTiff otherwise. LZW-compressed — embeddings are spatially
    correlated so LZW hits ~2-3× compression on real data.

    Returns ``(bytes, suggested_filename)``. Callers typically stream this
    to a FastAPI ``Response`` with ``Content-Disposition: attachment``.
    """
    emb: np.ndarray = result["embedding_int8"]     # (H, W, D) int8
    h, w, d = emb.shape
    transform = result["transform"]
    crs = result["crs"]
    repo_tag = result["model_repo_id"].replace("/", "_")

    # rasterio wants (band, H, W) — transpose from (H, W, D).
    data = np.transpose(emb, (2, 0, 1))

    import io as _io  # noqa: PLC0415
    import rasterio as _rio  # noqa: PLC0415
    from rasterio.io import MemoryFile  # noqa: PLC0415

    def _write(driver: str, extra: dict[str, Any]) -> bytes:
        with MemoryFile() as memfile:
            profile: dict[str, Any] = {
                "driver": driver,
                "width": w,
                "height": h,
                "count": d,
                "dtype": "int8",
                "crs": crs,
                "transform": transform,
                "nodata": -128,
                "compress": "lzw",
                "predictor": 2,
                **extra,
            }
            with memfile.open(**profile) as ds:
                ds.write(data)
                # Embed provenance so downstream consumers know what
                # pipeline produced this file.
                ds.update_tags(
                    model_repo_id=str(result["model_repo_id"]),
                    embedding_dim=str(result["embedding_dim"]),
                    patch_size=str(result["patch_size"]),
                    target_gsd_m=str(result["target_gsd_m"]),
                    patch_gsd_m=str(result["target_gsd_m"] * result["patch_size"]),
                    n_periods=str(result["n_periods"]),
                    period_days=str(result["period_days"]),
                    modality=result["modality"],
                    chunks_processed=str(result["chunks_processed"]),
                    chunks_failed=str(result["chunks_failed"]),
                    chunks_total=str(result["chunks_total"]),
                    quantization="olmoearth_pretrain.evals.embedding_transforms.quantize_embeddings",
                    nodata_value="-128",
                    roger_studio_version="0.2.0",
                )
                # Per-band descriptions so GIS tools show "dim_042" etc.
                for i in range(d):
                    ds.set_band_description(i + 1, f"dim_{i:03d}")
            buf = _io.BytesIO()
            buf.write(memfile.read())
            return buf.getvalue()

    try:
        cog_bytes = _write("COG", {"BLOCKSIZE": 256})
    except Exception as e:
        # COG driver not available in some rasterio builds. Fall back to a
        # regular tiled GTiff — users can post-process with `rio cogeo` if
        # strict COG compliance matters.
        logger.info("COG driver unavailable (%s) — writing tiled GTiff", e)
        cog_bytes = _write("GTiff", {
            "tiled": True, "blockxsize": 256, "blockysize": 256,
        })

    filename = f"{repo_tag}_embedding_{h}x{w}x{d}.tif"
    return cog_bytes, filename


async def _run_real_inference(
    bbox: BBox,
    model_repo_id: str,
    date_range: str,
    max_size_px: int,
    sliding_window: bool = False,
    window_size: int = 32,
    event_date: str | None = None,
    disconnect_check=None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Fetch S2 composite + run forward pass (encoder or full FT model).

    Three fetch paths, chosen by the FT head's declared ``input_spec``:

      1. **Pre/post pair** (ForestLossDriver) — when ``pre_post_split`` is
         set AND the caller provided an ``event_date``, we fetch two AOI-
         scope groups of scenes (pre window ~``-300d``, post window
         ~``+7d``), encode each independently, concatenate features along
         the channel dim (768 + 768 → 1536), and feed the resulting
         tensor to the conv-pool-fc head.

      2. **Temporal stack** (most FT heads) — when the head was trained on
         PER_PERIOD_MOSAIC layers (Ecosystem / AWF / Mangrove), we fetch
         ``n_periods`` sequential 30-day mosaics ending at ``date_range``
         and stack them along T.

      3. **Legacy single scene** — for base encoders, unknown repos, and
         FT heads with input_spec flags we can't satisfy yet (LFMC needs
         S1; ForestLossDriver without an ``event_date``). We log a known-
         broken warning for the latter so users aren't surprised when
         output looks degenerate.
    """
    # Load the model up-front so we can query its input_spec + patch_size.
    model, device = await asyncio.to_thread(olmoearth_model.load_encoder, model_repo_id)

    input_spec: dict[str, Any] = {}
    if isinstance(model, olmoearth_ft.FTModel):
        input_spec = dict((model.metadata or {}).get("input_spec") or {})
        effective_patch = (model.metadata or {}).get("patch_size") or _PATCH_SIZE
    else:
        effective_patch = _PATCH_SIZE

    use_pre_post = bool(
        input_spec
        and input_spec.get("pre_post_split")
        and event_date
        and not input_spec.get("s1_required")
    )

    use_temporal = bool(
        input_spec
        and input_spec.get("n_periods", 1) > 1
        and not input_spec.get("s1_required")
        and not input_spec.get("pre_post_split")
    )

    if input_spec and input_spec.get("pre_post_split") and not event_date:
        # User picked a pre/post head but didn't supply an event_date — the
        # change-detection pipeline can't run without it. Log and fall
        # through to the legacy single-scene path so the request still
        # produces *some* output rather than 500ing.
        logger.warning(
            "inference %s requires pre/post pair but event_date is missing — "
            "falling back to legacy single-scene path; output will be off-distribution",
            model_repo_id,
        )
    elif input_spec and input_spec.get("s1_required"):
        logger.warning(
            "inference %s requires sentinel1 — falling back to legacy S2-only "
            "single-scene path; output will be off-distribution",
            model_repo_id,
        )

    # Determine the per-chunk sliding-window size. The FT head's metadata
    # carries ``predict_window_px`` from its rslearn run config (e.g. 64
    # for ForestLossDriver, 32 for Ecosystem). When ``sliding_window`` is
    # turned on but the head doesn't declare a window size, we fall back
    # to the caller's ``window_size`` parameter (default 64). When the
    # head DOES declare one, use it — it's what the head was trained on.
    declared_window_px = None
    if isinstance(model, olmoearth_ft.FTModel):
        declared_window_px = (input_spec or {}).get("predict_window_px")
    effective_window_size = int(declared_window_px or window_size)

    if use_pre_post:
        assert isinstance(model, olmoearth_ft.FTModel)
        return await _run_chunked_pre_post_inference(
            bbox=bbox,
            model=model,
            device=device,
            input_spec=input_spec,
            effective_patch=effective_patch,
            event_date=event_date,
            sliding_window=sliding_window,
            window_size=effective_window_size,
            disconnect_check=disconnect_check,
            job_id=job_id,
        )

    # Temporal-path FT heads route to the chunked native-resolution
    # orchestrator — splits the AOI into ~5 km tiles, fetches each at full
    # 10 m/pixel via parallel band reads, runs FT inference per chunk, and
    # stitches into a global pixel raster. This matches the OlmoEarth
    # viewer's pipeline; quality differences vs. the viewer now reduce to
    # mosaic-compositing strategy (we still pick least-cloudy per period)
    # rather than spatial resolution.
    if use_temporal:
        assert isinstance(model, olmoearth_ft.FTModel)  # implied by use_temporal
        return await _run_chunked_aoi_inference(
            bbox=bbox,
            model=model,
            device=device,
            input_spec=input_spec,
            effective_patch=effective_patch,
            date_range=date_range,
            sliding_window=sliding_window,
            window_size=effective_window_size,
            disconnect_check=disconnect_check,
            job_id=job_id,
        )

    # Legacy single-scene path — base encoders + FT heads with input_spec
    # flags we can't satisfy yet (LFMC needs S1; ForestLossDriver needs the
    # pre/post pair). Behavior preserved exactly so nothing regresses.
    crop_step = window_size if sliding_window else effective_patch

    scene = await fetch_s2_composite(
        bbox=bbox,
        datetime_range=date_range,
        max_size_px=max_size_px,
        max_cloud_cover=40.0,
    )
    ts: tuple[int, int, int] | list[tuple[int, int, int]] = timestamp_from_iso(scene.datetime_str)
    h, w, _ = scene.image.shape
    h4 = (h // crop_step) * crop_step
    w4 = (w // crop_step) * crop_step
    if h4 == 0 or w4 == 0:
        raise ValueError(
            f"fetched scene {scene.image.shape} too small for crop_step={crop_step}"
        )
    image_bhwtc = image_to_bhwtc(scene.image[:h4, :w4, :])
    scene_fields: dict[str, Any] = {
        "raster_transform": scene.transform,
        "raster_crs": scene.crs,
        "raster_height": int(h4),
        "raster_width": int(w4),
        "scene_id": scene.scene_id,
        "scene_datetime": scene.datetime_str,
        "scene_cloud_cover": scene.cloud_cover,
        "patch_size": effective_patch,
        "sliding_window": sliding_window,
        "window_size": window_size if sliding_window else None,
        "temporal_stack": None,
    }

    if isinstance(model, olmoearth_ft.FTModel):
        if sliding_window:
            ft_result = await asyncio.to_thread(
                olmoearth_model.run_ft_tiled_inference,
                model,
                image_bhwtc,
                ts,
                window_size,
                effective_patch,
                device,
            )
        else:
            ft_result = await asyncio.to_thread(
                olmoearth_model.run_ft_inference,
                model,
                image_bhwtc,
                ts,
                effective_patch,
                device,
            )
        scalar_hi = np.repeat(
            np.repeat(ft_result.scalar, effective_patch, axis=0), effective_patch, axis=1
        )
        class_raster_hi: np.ndarray | None = None
        if ft_result.class_raster is not None:
            class_raster_hi = np.repeat(
                np.repeat(ft_result.class_raster, effective_patch, axis=0),
                effective_patch, axis=1,
            )

        # Choose colormap/legend: FT head metadata trumps the static map
        # keyed on repo id at module top, so a tentative class list still
        # gets surfaced correctly.
        colormap = ft_result.colormap
        legend = _build_ft_legend(ft_result)
        # Unique class ids actually present in the prediction raster.
        # Drives the frontend's "show only colors that appear on screen"
        # legend — the user shouldn't see 110 ecosystem swatches when
        # their Bay Area AOI only contains 4 or 5 classes. Stored as a
        # sorted Python list so it's JSON-serializable; computed on the
        # high-res raster because that's what the tile renderer shows.
        # Falls back to None for regression / embedding tasks where no
        # class raster exists. 256-element cap is defensive — a bad
        # raster shouldn't blow out the response payload with thousands
        # of IDs.
        present_class_ids: list[int] | None = None
        if class_raster_hi is not None:
            uniq = np.unique(class_raster_hi).astype(int)
            # Drop anything outside the legend range; these are artifacts
            # of the rasterization, not meaningful classes.
            uniq = uniq[(uniq >= 0) & (uniq < ft_result.num_classes)]
            present_class_ids = uniq.tolist()[:256]

        return {
            **scene_fields,
            "scalar_raster": scalar_hi,
            "task_type": ft_result.task_type,
            "num_classes": ft_result.num_classes,
            "class_names": ft_result.class_names,
            "class_names_tentative": ft_result.class_names_tentative,
            "class_raster": class_raster_hi,
            "class_probs": (
                ft_result.class_probs.tolist()
                if ft_result.class_probs is not None else None
            ),
            "present_class_ids": present_class_ids,
            "prediction_value": ft_result.prediction_value,
            "units": ft_result.units,
            "decoder_key": ft_result.decoder_key,
            "colormap_override": colormap,
            "legend_override": legend,
        }

    # Base encoder path — PCA of per-patch embedding.
    result = await asyncio.to_thread(
        olmoearth_model.run_s2_inference,
        model,
        image_bhwtc,
        ts,
        _PATCH_SIZE,
        device,
    )
    scalar_hi = np.repeat(
        np.repeat(result.scalar, _PATCH_SIZE, axis=0), _PATCH_SIZE, axis=1
    )
    return {
        **scene_fields,
        "scalar_raster": scalar_hi,
        "embedding_dim": result.embedding_dim,
        "task_type": "embedding",
    }


def _build_ft_legend(ft_result: "olmoearth_model.FTInferenceResult") -> dict[str, Any]:
    """Build a per-task legend block for the API response.

    Classification/segmentation: ``{kind, classes: [{name, color, index}]}``.
    Regression: ``{kind, units, value_range, value}``.

    Prefers the task's **published** class colors (from the olmoearth_projects
    ``olmoearth_run.yaml`` legend) when available. Falls back to cycling
    through the colormap gradient for tasks without published colors (e.g.
    ad-hoc FT repos added later).
    """
    base_colormap = _COLORMAP_LEGEND.get(ft_result.colormap) or _COLORMAP_LEGEND["embedding"]
    if ft_result.task_type == "regression":
        return {
            "kind": "regression",
            "label": base_colormap["label"],
            "stops": base_colormap["stops"],
            "units": ft_result.units,
            "value": ft_result.prediction_value,
        }
    published = ft_result.class_colors
    classes = []
    for i, name in enumerate(ft_result.class_names):
        if published is not None and i < len(published):
            color = published[i]
        else:
            stops = base_colormap["stops"]
            ratio = i / max(1, ft_result.num_classes - 1)
            color = _stop_color(stops, ratio)
        classes.append({
            "index": i,
            "name": name,
            "color": color,
        })
    return {
        "kind": ft_result.task_type,
        "classes": classes,
        "names_tentative": ft_result.class_names_tentative,
        "colors_source": "published" if published is not None else "colormap_gradient",
    }


def _stop_color(stops: list[tuple[str, float]], t: float) -> str:
    r, g, b = _interp_color(list(stops), t)
    return f"#{r:02x}{g:02x}{b:02x}"


def vectorize_classification(
    *,
    class_raster: np.ndarray,
    transform: Any,
    crs: Any,
    target_gsd_m: float,
    class_names: list[str],
    legend_classes: list[dict[str, Any]] | None = None,
    min_pixels: int = 4,
    simplify_tolerance_m: float | None = 5.0,
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a per-pixel classification raster into a GeoJSON FeatureCollection.

    Pipeline (each step justified):

      1. ``rasterio.features.shapes`` walks the raster and emits one polygon per
         contiguous int region — this is the standard rasterio-side vectorize.
      2. We compute area in the SOURCE CRS (typically UTM where pixels are
         isotropic meters) BEFORE reprojection, so the ``area_m2`` property
         reflects real ground area regardless of where the polygon lands in
         WGS84.
      3. Drop polygons below ``min_pixels`` to suppress speckle noise — a
         typical FT classification produces lots of single-pixel "regions"
         from inference jitter that aren't useful and bloat the file.
      4. Optional ``simplify_tolerance_m`` runs Douglas–Peucker via shapely
         to drop colinear vertices. Default 5 m is half the S2 GSD —
         visually identical, often 5–10× smaller GeoJSON.
      5. Reproject to EPSG:4326 (WGS84) via ``rasterio.warp.transform_geom``
         so the output is the lon/lat that GeoJSON consumers (Google Earth,
         QGIS, My Maps, leaflet, deck.gl) expect by RFC 7946.

    The output is a plain dict shaped as a GeoJSON FeatureCollection — caller
    serializes to a string at HTTP boundary. Class properties carry the
    class id, human-readable name, area (m²), pixel count, and the
    published color (when available, for QGIS / Earth styling).

    Args:
        class_raster: ``(H, W)`` int raster of class IDs. Pixels with values
            outside ``[0, len(class_names))`` are filtered out (covers
            nodata sentinels and out-of-AOI padding).
        transform: rasterio Affine for the source raster.
        crs: rasterio CRS the raster is in. Anything ``transform_geom`` can
            consume.
        target_gsd_m: source pixel size in meters (assumed isotropic), used
            for ``area_m2`` and pixel-count calculations.
        class_names: ``[name_for_id_0, name_for_id_1, ...]``.
        legend_classes: optional ``[{"index", "name", "color"}, ...]`` from
            ``_build_ft_legend``; supplies per-class hex colors. Falls back
            to ``"#888888"`` when missing.
        min_pixels: drop polygons under this pixel count. Default 4 (=
            ~160 m² at 10 m GSD).
        simplify_tolerance_m: Douglas–Peucker tolerance in meters. ``None``
            or ``0`` disables simplification.
        extra_properties: merged into the FeatureCollection's top-level
            ``properties`` block — useful for stamping model_repo_id,
            scene_datetime, etc. so consumers can trace provenance.
    """
    # Local imports — keep heavy/optional deps out of module load time and
    # near the only function that uses them.
    from rasterio.features import shapes  # noqa: PLC0415
    from rasterio.warp import transform_geom  # noqa: PLC0415
    from shapely.geometry import mapping, shape  # noqa: PLC0415

    color_lookup: dict[int, str] = {}
    if legend_classes:
        for c in legend_classes:
            try:
                color_lookup[int(c["index"])] = str(c.get("color", "#888888"))
            except (KeyError, ValueError, TypeError):
                continue

    pixel_area_m2 = float(target_gsd_m) ** 2
    n_classes = len(class_names)
    features: list[dict[str, Any]] = []

    # rasterio.features.shapes wants a contiguous typed array. int32 covers
    # OlmoEarth's 110-class ecosystem head with room to spare.
    raster_i32 = np.ascontiguousarray(class_raster, dtype=np.int32)

    for geom_dict, raw_value in shapes(raster_i32, transform=transform):
        class_id = int(raw_value)
        if class_id < 0 or class_id >= n_classes:
            # Out-of-range class — skip nodata sentinels and any padding
            # the orchestrator may have left at AOI edges.
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue
        area_m2 = float(geom.area)
        pixel_count = int(round(area_m2 / pixel_area_m2)) if pixel_area_m2 else 0
        if pixel_count < min_pixels:
            continue
        if simplify_tolerance_m and simplify_tolerance_m > 0:
            geom = geom.simplify(simplify_tolerance_m, preserve_topology=True)
            if geom.is_empty:
                continue
        wgs_geom = transform_geom(crs, "EPSG:4326", mapping(geom))
        features.append({
            "type": "Feature",
            "geometry": wgs_geom,
            "properties": {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "color": color_lookup.get(class_id, "#888888"),
                "area_m2": round(area_m2, 1),
                "pixel_count": pixel_count,
            },
        })

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if extra_properties:
        # GeoJSON spec doesn't define top-level ``properties`` on a
        # FeatureCollection, but most consumers tolerate it (Google Earth,
        # QGIS, leaflet) and it's the conventional place to stash
        # provenance metadata. Keep it as a sibling key, not inside
        # ``features``, so per-polygon properties stay clean.
        fc["properties"] = dict(extra_properties)
    return fc


def _build_response(job: dict[str, Any]) -> dict[str, Any]:
    """Public response shape for start_inference."""
    # FT jobs override both colormap + legend at inference time to reflect
    # the task (mangrove / LFMC / AWF / …) rather than the keyword-mapped
    # default keyed on repo id.
    colormap = job.get("colormap_override") or job["colormap"]
    legend = job.get("legend_override") or job.get("legend")
    resp = {
        "job_id": job["job_id"],
        "tile_url": f"/api/olmoearth/infer-tile/{job['job_id']}/{{z}}/{{x}}/{{y}}.png",
        "legend": legend,
        "colormap": colormap,
        "kind": job["kind"],
        "status": job["status"],
        "model_repo_id": job["spec"]["model_repo_id"],
        "bbox": job["spec"]["bbox"],
    }
    if job["kind"] == "pytorch":
        task_type = job.get("task_type", "embedding")
        scene_block = {
            "scene_id": job.get("scene_id"),
            "scene_datetime": job.get("scene_datetime"),
            "scene_cloud_cover": job.get("scene_cloud_cover"),
            "patch_size": job.get("patch_size"),
            "task_type": task_type,
        }
        if task_type == "embedding":
            resp.update(
                {
                    **scene_block,
                    "embedding_dim": job.get("embedding_dim"),
                    "notes": [
                        "Real OlmoEarth encoder forward pass over a Sentinel-2 L2A "
                        "composite from Microsoft Planetary Computer.",
                        "Scalar raster is the first principal component of the "
                        "per-patch embedding, rescaled to [0, 1], then colormapped.",
                    ],
                }
            )
        else:  # classification / segmentation / regression (FT head executed)
            resp.update(
                {
                    **scene_block,
                    "num_classes": job.get("num_classes"),
                    "class_names": job.get("class_names"),
                    "class_names_tentative": job.get("class_names_tentative"),
                    "class_probs": job.get("class_probs"),
                    # Class IDs that actually appear in the rendered raster
                    # (vs the full 110-element catalog). The frontend uses
                    # this to show ONLY the colors on screen in the map
                    # legend strip — a Bay Area ecosystem tile with 4
                    # classes shouldn't surface 110 swatches.
                    "present_class_ids": job.get("present_class_ids"),
                    "prediction_value": job.get("prediction_value"),
                    "units": job.get("units"),
                    "decoder_key": job.get("decoder_key"),
                    "notes": [
                        f"Real OlmoEarth fine-tuned {task_type} head "
                        f"({job.get('decoder_key')}) executed over a Sentinel-2 L2A "
                        "composite from Microsoft Planetary Computer.",
                        (
                            "Class names are tentative (not persisted in the checkpoint); "
                            "verify against olmoearth_projects rslearn configs."
                            if job.get("class_names_tentative")
                            else "Class names sourced from olmoearth_projects."
                        ),
                    ],
                }
            )
    else:
        reason = job.get("stub_reason", "unknown")
        resp["stub_reason"] = reason
        resp["notes"] = [
            "STUB output — tile colors are a deterministic gradient over the bbox, "
            "NOT a real model forward pass. A 'PREVIEW' watermark is burned into "
            "every tile so it can't be mistaken for real predictions.",
            "Fallback was triggered because the real inference pipeline errored; "
            "check `stub_reason` for details.",
        ]
        # Actionable retry suggestions keyed off the error text. Without these
        # the LLM has to invent next steps and tends to ask the user vague
        # "want me to retry?" questions. Giving it concrete param changes to
        # try means the agent can make a real recommendation. Keys/patterns
        # match the exceptions raised in olmoearth_model._validate_s2_dn_range
        # and sentinel2_fetch (empty composite, no scenes found, etc.).
        resp["suggested_retries"] = _suggested_retries_for_stub(reason, job["spec"])
    return resp


def _suggested_retries_for_stub(reason: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concrete retry suggestions the LLM can pass back to the user.
    Each entry is ``{description, params}`` where ``params`` maps to the
    ``run_olmoearth_inference`` tool schema — so the agent can literally
    copy-paste into its next tool call."""
    r = (reason or "").lower()
    current_sw = bool(spec.get("sliding_window", False))
    suggestions: list[dict[str, Any]] = []

    if (
        "empty" in r
        or "all-zero" in r
        or "no scenes" in r
        or "no sentinel" in r
        or "sentinelfetcherror" in r
        or "pre-normalized" in r
    ):
        # Empty composite — most likely fix is a different date range or
        # relaxed cloud filter. Also try turning off sliding window: a single
        # forward pass uses the whole bbox as one composite, so partial land
        # coverage is less likely to produce all-zero.
        suggestions.append({
            "description": (
                "Retry with a wider date range — 6-month windows are more likely "
                "to include a cloud-free scene than a single month."
            ),
            "params": {"date_range": "2024-03-01/2024-09-30"},
        })
        if current_sw:
            suggestions.append({
                "description": (
                    "Retry with sliding_window=false — a single forward pass over "
                    "the whole bbox avoids the per-window all-zero failure mode."
                ),
                "params": {"sliding_window": False},
            })
    if "outside" in r or "dn range" in r:
        suggestions.append({
            "description": (
                "The upstream fetch returned values outside the expected DN range. "
                "Try a different bbox or date range to rule out scene-specific corruption."
            ),
            "params": {},
        })
    if "non-finite" in r or "nan" in r:
        suggestions.append({
            "description": (
                "NaN/inf values in the composite — often a partial band read. "
                "Retry once (transient) or try a slightly different bbox."
            ),
            "params": {},
        })
    if not suggestions:
        # Generic fallback: at least tell the LLM to try a different date + smaller bbox.
        suggestions.append({
            "description": (
                "Unknown stub reason — try a different date_range and / or a smaller "
                "max_size_px (128) to reduce memory pressure on the S2 reader."
            ),
            "params": {"max_size_px": 128},
        })
    return suggestions


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def clear_jobs() -> None:
    """Drop all cached inference jobs — used by tests."""
    _jobs.clear()


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------


def _tile_to_lonlat_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Web-Mercator tile → (west, south, east, north) in degrees."""
    n = 2.0 ** z
    lon_deg_w = x / n * 360.0 - 180.0
    lon_deg_e = (x + 1) / n * 360.0 - 180.0
    lat_rad_n = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_rad_s = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    return (lon_deg_w, math.degrees(lat_rad_s), lon_deg_e, math.degrees(lat_rad_n))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interp_color(stops: list[tuple[str, float]], t: float) -> tuple[int, int, int]:
    """Piecewise-linear interpolate an (r, g, b) from (hex, position) stops."""
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        (c0, p0), (c1, p1) = stops[i], stops[i + 1]
        if p0 <= t <= p1:
            frac = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            return (
                int(r0 + frac * (r1 - r0)),
                int(g0 + frac * (g1 - g0)),
                int(b0 + frac * (b1 - b0)),
            )
    return _hex_to_rgb(stops[-1][0])


def _colorize(scalar01: np.ndarray, stops: list[tuple[str, float]]) -> np.ndarray:
    """Vectorized scalar → RGBA lookup using the colormap stops."""
    h, w = scalar01.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Flatten for a tight loop; 256×256 = 65k lookups is fast in pure python.
    flat = scalar01.reshape(-1)
    out = rgba.reshape(-1, 4)
    valid = ~np.isnan(flat)
    for i, t in enumerate(flat):
        if not valid[i]:
            continue
        r, g, b = _interp_color(stops, float(t))
        out[i] = (r, g, b, 210)
    return out.reshape(h, w, 4)


def _colorize_classes(
    class_raster: np.ndarray, outside: np.ndarray, class_colors: list[str]
) -> np.ndarray:
    """Map per-pixel class index → legend color. Out-of-bbox stays transparent.

    Uses a small lookup table so 256×256 = 65k pixels are resolved as one
    numpy ``take`` per color channel.

    Negative class indices are treated as **nodata** and rendered fully
    transparent — the few-shot pipeline uses ``class_raster = -1`` for
    pixels where the encoder produced no embedding (chunks that failed
    to fetch / scenes that returned 0). Without this branch, the
    ``np.clip`` below would map -1 → 0 and paint nodata pixels in
    class 0's colour, giving the user a false "class 0 everywhere"
    impression on AOI edges that have no real signal.
    """
    h, w = class_raster.shape
    n_classes = len(class_colors)
    # (n_classes, 3) RGB palette. Any class index outside the legend falls back
    # to a neutral grey so unexpected labels are still visible but clearly odd.
    palette = np.zeros((max(n_classes, 1), 3), dtype=np.uint8)
    for i, hex_color in enumerate(class_colors):
        palette[i] = _hex_to_rgb(hex_color)

    nodata = class_raster < 0
    idx = np.clip(class_raster, 0, n_classes - 1) if n_classes > 0 else np.zeros_like(class_raster)
    rgb = palette[idx.reshape(-1)].reshape(h, w, 3)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 210
    rgba[nodata] = (0, 0, 0, 0)
    rgba[outside] = (0, 0, 0, 0)
    return rgba


def _sample_raster_tile(
    raster: np.ndarray, job: dict[str, Any], z: int, x: int, y: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample a 2D prediction raster onto a 256×256 WGS-84 tile grid.

    Returns ``(sampled, outside)`` where ``sampled`` preserves ``raster``'s
    dtype and ``outside`` is a bool mask of pixels that fell outside the
    raster envelope. Returns ``None`` if the tile lies entirely outside the
    job bbox (caller should emit a transparent tile).
    """
    job_bbox = job["spec"]["bbox"]
    tile_w, tile_s, tile_e, tile_n = _tile_to_lonlat_bounds(z, x, y)
    if (
        tile_e < job_bbox["west"]
        or tile_w > job_bbox["east"]
        or tile_n < job_bbox["south"]
        or tile_s > job_bbox["north"]
    ):
        return None

    h, w = raster.shape
    transform = job["raster_transform"]
    scene_crs = job["raster_crs"]

    # 256×256 WGS-84 coord grid → scene CRS → raster (row, col).
    ts_per_pixel = 256
    lons = tile_w + (np.arange(ts_per_pixel) + 0.5) / ts_per_pixel * (tile_e - tile_w)
    lats = tile_n - (np.arange(ts_per_pixel) + 0.5) / ts_per_pixel * (tile_n - tile_s)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    xs, ys = rasterio.warp.transform(
        CRS.from_string("EPSG:4326"), scene_crs,
        lon_grid.ravel().tolist(), lat_grid.ravel().tolist(),
    )
    xs = np.asarray(xs).reshape(ts_per_pixel, ts_per_pixel)
    ys = np.asarray(ys).reshape(ts_per_pixel, ts_per_pixel)

    inv = ~transform
    cols_f, rows_f = inv * (xs, ys)

    # Round to nearest pixel, then compute the outside mask from the rounded
    # integers — not the raw floats. The audit caught this: previously we
    # np.clip'd the rounded indices first, then built `outside` from cols_f
    # only. A float like `w - 0.3` satisfied `cols_f < w` (so outside=False),
    # but `np.round(w - 0.3) == w` fell off the end of the raster and got
    # np.clip'd back to `w - 1`. The caller then saw outside=False and kept
    # that nearest-valid-edge sample — producing a visible smeared band of
    # color past the AOI boundary in compare-mode overlays. Computing outside
    # on the rounded integers (before clipping) means the caller's mask
    # correctly covers the clipped pixels and writes transparent/NaN.
    cols_rounded = np.round(cols_f).astype(np.int32)
    rows_rounded = np.round(rows_f).astype(np.int32)
    outside = (
        (cols_rounded < 0) | (cols_rounded >= w)
        | (rows_rounded < 0) | (rows_rounded >= h)
    )
    cols = np.clip(cols_rounded, 0, w - 1)
    rows = np.clip(rows_rounded, 0, h - 1)
    sampled = raster[rows, cols]
    return sampled, outside


def _empty_tile_png() -> bytes:
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _png_from_rgba(rgba: np.ndarray) -> bytes:
    from PIL import Image  # noqa: PLC0415

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_pytorch_tile(job: dict[str, Any], z: int, x: int, y: int) -> bytes:
    """Render one 256×256 tile for a real-inference job.

    Dispatch (in order, first match wins):
      - **rgb_raster present** (e.g. embedding PCA false-color, future
        similarity heatmaps rendered as multi-channel) → direct pass-
        through of the 3-band uint8 tensor, no colormap.
      - classification / segmentation with a ``class_raster`` → class-
        colored tile using per-class legend colors.
      - everything else (embedding, regression) → scalar gradient
        colormap.
    """
    task_type = job.get("task_type", "embedding")
    class_raster: np.ndarray | None = job.get("class_raster")
    rgb_raster: np.ndarray | None = job.get("rgb_raster")
    legend = job.get("legend_override") or job.get("legend") or {}
    legend_classes = legend.get("classes") or []

    # RGB path — for tools whose output is a 3-band uint8 visualization
    # (PCA false-color, RGB similarity overlays). Each channel samples
    # independently from the source raster; out-of-bounds pixels go
    # fully transparent so the base map shows through.
    if rgb_raster is not None and rgb_raster.ndim == 3 and rgb_raster.shape[-1] == 3:
        rgba = _render_rgb_tile(rgb_raster, job, z, x, y)
        if rgba is None:
            return _empty_tile_png()
        return _png_from_rgba(rgba)

    if (
        class_raster is not None
        and task_type in ("classification", "segmentation")
        and legend_classes
    ):
        sampled = _sample_raster_tile(class_raster, job, z, x, y)
        if sampled is None:
            return _empty_tile_png()
        idx_tile, outside = sampled
        class_colors = [c["color"] for c in legend_classes]
        rgba = _colorize_classes(idx_tile.astype(np.int64), outside, class_colors)
        return _png_from_rgba(rgba)

    scalar: np.ndarray = job["scalar_raster"]
    sampled = _sample_raster_tile(scalar, job, z, x, y)
    if sampled is None:
        return _empty_tile_png()
    sampled_2d, outside = sampled
    scalar_tile = sampled_2d.astype(np.float32)
    scalar_tile[outside] = np.nan

    colormap_key = job.get("colormap_override") or job.get("colormap", "embedding")
    colormap = _COLORMAP_LEGEND.get(colormap_key) or _COLORMAP_LEGEND["embedding"]
    stops = colormap["stops"]
    rgba = _colorize(scalar_tile, list(stops))
    return _png_from_rgba(rgba)


def _render_rgb_tile(
    rgb_raster: np.ndarray, job: dict[str, Any], z: int, x: int, y: int,
) -> np.ndarray | None:
    """Sample a ``(H, W, 3)`` uint8 RGB raster onto a 256×256 tile grid.

    Mirrors ``_sample_raster_tile`` but for a 3-channel source. Out-of-
    bounds pixels get ``alpha=0`` so the base map shows through at AOI
    edges. Inside the AOI, ``alpha=210`` matches the opacity used by
    the other tile renderers so overlays stay legible.
    """
    # Re-use the 2D sampler by independently sampling each channel at
    # the same tile coordinates. Cheaper than re-implementing the CRS
    # math for 3 dimensions — 256×256 × 3 is trivially fast.
    sampled_r = _sample_raster_tile(rgb_raster[:, :, 0], job, z, x, y)
    if sampled_r is None:
        return None
    sampled_g = _sample_raster_tile(rgb_raster[:, :, 1], job, z, x, y)
    sampled_b = _sample_raster_tile(rgb_raster[:, :, 2], job, z, x, y)
    # Both g and b must also be within bounds at this point (same source
    # transform) — defensive check anyway.
    if sampled_g is None or sampled_b is None:
        return None
    r_tile, outside = sampled_r
    g_tile, _ = sampled_g
    b_tile, _ = sampled_b

    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[..., 0] = r_tile.astype(np.uint8)
    rgba[..., 1] = g_tile.astype(np.uint8)
    rgba[..., 2] = b_tile.astype(np.uint8)
    rgba[..., 3] = 210
    rgba[outside] = (0, 0, 0, 0)
    return rgba


def raster_class_histogram(job_id: str, top_n: int = 20) -> dict[str, Any]:
    """Per-class pixel counts for a finished inference job.

    Called by the ``query_raster_histogram`` LLM tool so the explainer
    agent can ground its answer in ACTUAL pixel distribution instead of
    paraphrasing the 110-element class catalog. Returns top-``top_n``
    classes by pixel count, with percentages, names, and colors. Values
    outside [0, num_classes) are dropped (defensive — class_raster
    rounding can occasionally land just past the edge).

    Caller contract:
      - ``job_id`` is the sha16 returned by ``start_inference``.
      - Only classification / segmentation jobs have a ``class_raster``;
        everything else returns ``{"error": "no_class_raster", ...}``.
      - Stub jobs carry no real raster — surfaced as
        ``{"error": "stub_job", "stub_reason": ...}``.
    """
    job = _jobs.get(job_id)
    if job is None:
        return {"error": "unknown_job", "job_id": job_id}
    if job.get("kind") == "stub":
        return {
            "error": "stub_job",
            "stub_reason": job.get("stub_reason"),
            "hint": "No real raster was rendered; retry the inference call.",
        }
    class_raster = job.get("class_raster")
    if class_raster is None:
        return {
            "error": "no_class_raster",
            "task_type": job.get("task_type"),
            "hint": (
                "This job is regression or embedding — pixel-class "
                "histograms don't apply. Use query_raster_stats for a "
                "scalar summary instead."
            ),
        }

    class_names: list[str] = job.get("class_names") or []
    legend = job.get("legend_override") or {}
    legend_classes = legend.get("classes") or []
    color_by_index: dict[int, str] = {
        int(c.get("index", i)): c.get("color", "#6b7280")
        for i, c in enumerate(legend_classes)
    }
    num_classes = int(job.get("num_classes") or len(class_names) or 0)

    flat = class_raster.reshape(-1)
    if num_classes > 0:
        valid = (flat >= 0) & (flat < num_classes)
        flat = flat[valid]
    total = int(flat.size)
    if total == 0:
        return {
            "error": "empty_raster",
            "hint": "Class raster exists but has zero valid pixels.",
        }

    # np.bincount scales to the max class id present — OK because we've
    # already clipped to [0, num_classes) above.
    counts = np.bincount(flat.astype(np.int64))
    ranked = [
        (int(i), int(counts[i]))
        for i in np.argsort(counts)[::-1]
        if counts[i] > 0
    ]
    top = ranked[: max(1, int(top_n))]
    return {
        "job_id": job_id,
        "total_pixels": total,
        "num_classes_present": len(ranked),
        "num_classes_total": num_classes,
        "classes": [
            {
                "index": idx,
                "name": class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}",
                "color": color_by_index.get(idx, "#6b7280"),
                "pixel_count": cnt,
                "percent": round(100.0 * cnt / total, 2),
            }
            for idx, cnt in top
        ],
        "note": (
            "Percentages sum to <100 when classes below the top-N cutoff "
            "account for the rest. ``num_classes_present`` tells you how "
            "many classes actually appear on the tile."
            if len(ranked) > len(top)
            else "All present classes included."
        ),
    }


def raster_scalar_stats(job_id: str) -> dict[str, Any]:
    """Mean / min / max / quartile summary of a scalar-output raster
    (regression tasks like LFMC, or PCA-embedding base encoders).

    Returned values are in the task's native units when the metadata
    declares them (e.g. LFMC % moisture) — the scalar raster is clamped
    to [0, 1] internally so we un-normalize using the declared
    ``value_range`` when present. Gives the LLM a ground-truth scalar
    summary to cite instead of parroting the ``prediction_value`` mean.
    """
    job = _jobs.get(job_id)
    if job is None:
        return {"error": "unknown_job", "job_id": job_id}
    if job.get("kind") == "stub":
        return {"error": "stub_job", "stub_reason": job.get("stub_reason")}
    scalar = job.get("scalar_raster")
    if scalar is None:
        return {"error": "no_scalar_raster"}

    flat = scalar.reshape(-1).astype(np.float64)
    mask = ~np.isnan(flat)
    valid = flat[mask]
    if valid.size == 0:
        return {"error": "empty_scalar_raster"}

    # Un-normalize via the task's declared value_range when available.
    lo = hi = None
    md = job.get("metadata") or {}
    vr = md.get("value_range")
    if isinstance(vr, (list, tuple)) and len(vr) == 2:
        try:
            lo, hi = float(vr[0]), float(vr[1])
        except (TypeError, ValueError):
            lo = hi = None
    def _denorm(v: float) -> float:
        if lo is None or hi is None:
            return v
        return lo + v * (hi - lo)

    return {
        "job_id": job_id,
        "valid_pixels": int(valid.size),
        "mean": round(_denorm(float(valid.mean())), 3),
        "min": round(_denorm(float(valid.min())), 3),
        "max": round(_denorm(float(valid.max())), 3),
        "p10": round(_denorm(float(np.percentile(valid, 10))), 3),
        "p50": round(_denorm(float(np.percentile(valid, 50))), 3),
        "p90": round(_denorm(float(np.percentile(valid, 90))), 3),
        "units": job.get("units") or md.get("units"),
    }


def raster_geotiff_bytes(job_id: str) -> tuple[bytes, str] | None:
    """Serialize a finished job's prediction raster as a georeferenced
    GeoTIFF. Returns ``(bytes, suggested_filename)`` or ``None`` when the
    job doesn't exist / is still running / is a stub (synthetic raster
    isn't worth exporting).

    Classification / segmentation → single-band ``int16`` with the per-
    pixel class id; class names are written into GDAL's ``RAT``/band
    descriptions so desktop GIS tools (QGIS, ArcGIS) show the class
    labels on hover without the user importing a sidecar CSV.
    Regression / embedding → single-band ``float32`` with the scalar
    value; un-normalized to the task's native units when a value_range
    is declared in FT metadata.

    Uses the job's stored ``raster_transform`` + ``raster_crs`` so the
    output opens georeferenced in any tool. No tiling / masking — the
    user's AOI bbox was already the tightest crop the inference ran on.
    """
    job = _jobs.get(job_id)
    if job is None:
        return None
    if job.get("status") != "ready":
        return None
    if job.get("kind") == "stub":
        return None

    transform = job.get("raster_transform")
    crs = job.get("raster_crs")
    if transform is None or crs is None:
        logger.warning("geotiff_export: job %s missing transform/crs", job_id)
        return None

    class_raster: np.ndarray | None = job.get("class_raster")
    scalar_raster: np.ndarray | None = job.get("scalar_raster")

    # Prefer class raster when available (classification / segmentation)
    # — users typically want the argmax layer, not the confidence scalar.
    # Regression jobs have no class_raster → fall through to scalar.
    if class_raster is not None:
        data = class_raster.astype(np.int16, copy=False)
        dtype = "int16"
        nodata = -1
        band_description = "argmax_class_id"
        count = 1
    elif scalar_raster is not None:
        # Scalar is held in [0, 1] internally; un-normalize to native
        # units when the FT metadata declared a value_range.
        md = job.get("metadata") or {}
        vr = md.get("value_range")
        arr = scalar_raster.astype(np.float32, copy=True)
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            try:
                lo, hi = float(vr[0]), float(vr[1])
                arr = lo + arr * (hi - lo)
            except (TypeError, ValueError):
                pass
        data = arr
        dtype = "float32"
        nodata = float("nan")
        band_description = job.get("units") or "scalar_value"
        count = 1
    else:
        return None

    import io  # noqa: PLC0415
    import rasterio  # noqa: PLC0415

    buf = io.BytesIO()
    height, width = data.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, band_description)
            # Embed job metadata as GeoTIFF tags — helpful for downstream
            # tools (QGIS shows these under Layer Properties → Metadata).
            spec = job.get("spec") or {}
            dst.update_tags(
                job_id=job_id,
                model_repo_id=str(spec.get("model_repo_id", "")),
                task_type=str(job.get("task_type", "")),
                scene_id=str(job.get("scene_id", "")),
                scene_datetime=str(job.get("scene_datetime", "")),
                date_range=str(spec.get("date_range", "")),
                roger_studio_version="0.1.0",
            )
            # Embed the class name table when it exists so GIS tools can
            # show labels without a sidecar file. One tag per class id —
            # verbose but universally readable.
            class_names = job.get("class_names") or []
            for i, name in enumerate(class_names):
                dst.update_tags(1, **{f"CLASS_{i}": name})
        buf.write(memfile.read())

    filename = f"{job_id}_{job.get('task_type', 'raster')}.tif"
    return buf.getvalue(), filename


def _render_stub_tile(job: dict[str, Any], z: int, x: int, y: int) -> bytes:
    """Fallback tile renderer — deterministic gradient + PREVIEW watermark.

    Only used when the real pipeline failed for this job. The watermark is the
    user-visible signal that these pixels are NOT a real model output.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    job_bbox = job["spec"]["bbox"]
    tile_w, tile_s, tile_e, tile_n = _tile_to_lonlat_bounds(z, x, y)

    if (
        tile_e < job_bbox["west"]
        or tile_w > job_bbox["east"]
        or tile_n < job_bbox["south"]
        or tile_s > job_bbox["north"]
    ):
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    colormap = _COLORMAP_LEGEND.get(job.get("colormap", "embedding"))
    stops = colormap["stops"] if colormap else _COLORMAP_LEGEND["embedding"]["stops"]

    seed_a = int(hashlib.md5(job["job_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    seed_b = int(hashlib.md5(job["job_id"].encode()[::-1]).hexdigest()[:8], 16) / 0xFFFFFFFF

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    pixels = img.load()
    assert pixels is not None
    for px in range(256):
        for py in range(256):
            lon = tile_w + (px / 256.0) * (tile_e - tile_w)
            lat = tile_n - (py / 256.0) * (tile_n - tile_s)
            if not (
                job_bbox["west"] <= lon <= job_bbox["east"]
                and job_bbox["south"] <= lat <= job_bbox["north"]
            ):
                continue
            field = (
                0.5
                + 0.35 * math.sin(lat * (2 + 8 * seed_a))
                + 0.35 * math.cos(lon * (1 + 6 * seed_b))
            ) / 1.2
            r, g, b = _interp_color(list(stops), field)
            pixels[px, py] = (r, g, b, 200)

    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    watermark = "PREVIEW · stub"
    draw.text((8, 232), watermark, font=font, fill=(0, 0, 0, 160))
    draw.text((7, 231), watermark, font=font, fill=(255, 255, 255, 200))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_tile(job_id: str, z: int, x: int, y: int) -> bytes | None:
    """Public tile-render entry point. Returns PNG bytes or None if job unknown."""
    job = _jobs.get(job_id)
    if job is None:
        return None
    if job.get("status") != "ready":
        # Inference is still running — return a transparent tile so the
        # frontend's loading state stays coherent.
        from PIL import Image  # noqa: PLC0415
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    if job["kind"] == "pytorch":
        return _render_pytorch_tile(job, z, x, y)
    if job["kind"] == "stub":
        return _render_stub_tile(job, z, x, y)
    raise NotImplementedError(f"inference kind={job['kind']} not supported")
