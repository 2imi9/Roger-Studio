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
import hashlib
import io
import json
import logging
import math
import time
from typing import Any

import numpy as np
import rasterio.warp
from rasterio.crs import CRS

from app.models.schemas import BBox
from app.services import olmoearth_ft, olmoearth_model
from app.services.sentinel2_fetch import (
    SentinelFetchError,
    fetch_s2_composite,
    image_to_bhwtc,
    timestamp_from_iso,
)

logger = logging.getLogger(__name__)


# Jobs: job_id -> spec + status. A job is just a named render config; the
# real work happens up-front (real path) or per tile (stub fallback).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

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
}


async def start_inference(
    bbox: BBox,
    model_repo_id: str,
    date_range: str | None = None,
    max_size_px: int = _DEFAULT_MAX_SIZE_PX,
    sliding_window: bool = False,
    window_size: int = 32,
    _auto_retry_depth: int = 0,
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
        }

    try:
        real = await _run_real_inference(
            bbox, model_repo_id, spec["date_range"], max_size_px,
            sliding_window=sliding_window, window_size=window_size,
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


async def _run_real_inference(
    bbox: BBox,
    model_repo_id: str,
    date_range: str,
    max_size_px: int,
    sliding_window: bool = False,
    window_size: int = 32,
) -> dict[str, Any]:
    """Fetch S2 composite + run forward pass (encoder or full FT model).

    Returns payload to merge into the cached job. For base encoders, the
    payload mirrors the original scalar-raster shape; for FT models the
    payload also carries ``task_type``, ``class_raster``, ``class_names``,
    etc. so the tile renderer + response builder can expose task-aware
    output.
    """
    scene = await fetch_s2_composite(
        bbox=bbox,
        datetime_range=date_range,
        max_size_px=max_size_px,
        max_cloud_cover=40.0,
    )

    # Load the model up-front so we can query its patch_size for the crop.
    model, device = await asyncio.to_thread(olmoearth_model.load_encoder, model_repo_id)
    ts = timestamp_from_iso(scene.datetime_str)

    # Different FT models were trained with different patch sizes (Mangrove=2,
    # others=4). Crop the fetched S2 image so it's divisible by the model's
    # native patch size, and — if sliding_window is on — also by window_size.
    if isinstance(model, olmoearth_ft.FTModel):
        effective_patch = (model.metadata or {}).get("patch_size") or _PATCH_SIZE
    else:
        effective_patch = _PATCH_SIZE
    crop_step = window_size if sliding_window else effective_patch
    h, w, _ = scene.image.shape
    h4 = (h // crop_step) * crop_step
    w4 = (w // crop_step) * crop_step
    if h4 == 0 or w4 == 0:
        raise ValueError(
            f"fetched scene {scene.image.shape} too small for crop_step={crop_step}"
        )
    image = scene.image[:h4, :w4, :]

    scene_fields = {
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
    }

    if isinstance(model, olmoearth_ft.FTModel):
        if sliding_window:
            ft_result = await asyncio.to_thread(
                olmoearth_model.run_ft_tiled_inference,
                model,
                image_to_bhwtc(image),
                ts,
                window_size,
                effective_patch,
                device,
            )
        else:
            ft_result = await asyncio.to_thread(
                olmoearth_model.run_ft_inference,
                model,
                image_to_bhwtc(image),
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
        image_to_bhwtc(image),
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
    """
    h, w = class_raster.shape
    n_classes = len(class_colors)
    # (n_classes, 3) RGB palette. Any class index outside the legend falls back
    # to a neutral grey so unexpected labels are still visible but clearly odd.
    palette = np.zeros((max(n_classes, 1), 3), dtype=np.uint8)
    for i, hex_color in enumerate(class_colors):
        palette[i] = _hex_to_rgb(hex_color)

    idx = np.clip(class_raster, 0, n_classes - 1) if n_classes > 0 else np.zeros_like(class_raster)
    rgb = palette[idx.reshape(-1)].reshape(h, w, 3)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 210
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

    Dispatch:
      - classification / segmentation with a ``class_raster`` → class-colored
        tile using per-class legend colors. Each class is a flat fill — the
        user sees a thematic map whose colors exactly match the legend.
      - everything else (embedding, regression) → scalar gradient colormap.
    """
    task_type = job.get("task_type", "embedding")
    class_raster: np.ndarray | None = job.get("class_raster")
    legend = job.get("legend_override") or job.get("legend") or {}
    legend_classes = legend.get("classes") or []

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
