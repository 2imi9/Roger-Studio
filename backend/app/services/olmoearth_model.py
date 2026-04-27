"""Real OlmoEarth model wrapper — loads pretrained encoders from HuggingFace
and runs forward passes on Sentinel-2 imagery.

This is the real-inference counterpart to ``olmoearth_inference.py``'s
``_render_stub_tile``. Everything here mirrors the public API documented in
``olmoearth_pretrain/docs/Inference-Quickstart.md`` and the
``Inference-Quickstart`` bash block:

    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=4)
        features = out["tokens_and_masks"].sentinel2_l2a   # (B, H', W', T, S, D)

Design notes:
  - Models are cached in-process by repo_id so the FastAPI worker doesn't
    reload 90 MB+ of weights on every inference request.
  - Forward passes release the GIL under ``torch.no_grad()`` so this can be
    safely dispatched via ``asyncio.to_thread`` from async routes.
  - All heavy imports (torch, olmoearth_pretrain) are module-scope so import
    failures surface at backend boot, not mid-request. The service layer
    may still fall back to the stub renderer on failed inference.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from huggingface_hub import hf_hub_download

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    ModelID,
    load_model_from_id,
    load_model_from_path,
)

from app.services import olmoearth_ft
from app.services.olmoearth_ft import FTModel

try:
    # Post-v1.0 wheel: a direct repo-id loader is published.
    from olmoearth_pretrain.model_loader import (  # type: ignore[attr-defined]
        load_model_from_repo_id as _load_model_from_repo_id_native,
    )
except ImportError:  # pragma: no cover — covered by env-pin check in tests
    _load_model_from_repo_id_native = None


def _load_from_repo_id(repo_id: str) -> torch.nn.Module:
    """Load any HF repo that ships ``config.json`` + ``weights.pth``.

    Falls back to a manual ``hf_hub_download`` + ``load_model_from_path`` flow
    when the installed wheel doesn't expose ``load_model_from_repo_id`` yet.
    Both files land in the same snapshot dir so pointing
    ``load_model_from_path`` at their shared parent Just Works.
    """
    if _load_model_from_repo_id_native is not None:
        return _load_model_from_repo_id_native(repo_id)
    cfg_path = hf_hub_download(repo_id=repo_id, filename=CONFIG_FILENAME)
    # Download the weights so they share the same snapshot dir, then point
    # the loader at it. We don't use the returned path — only its parent.
    hf_hub_download(repo_id=repo_id, filename=WEIGHTS_FILENAME)
    from pathlib import Path as _Path
    return load_model_from_path(_Path(cfg_path).parent)

logger = logging.getLogger(__name__)


# Known base encoders — these resolve via ModelID so we get the enum path
# that's friendlier to cache / telemetry.
_BASE_MODEL_IDS: dict[str, ModelID] = {
    "allenai/OlmoEarth-v1-Nano": ModelID.OLMOEARTH_V1_NANO,
    "allenai/OlmoEarth-v1-Tiny": ModelID.OLMOEARTH_V1_TINY,
    "allenai/OlmoEarth-v1-Base": ModelID.OLMOEARTH_V1_BASE,
    "allenai/OlmoEarth-v1-Large": ModelID.OLMOEARTH_V1_LARGE,
}

# Process-wide cache: repo_id -> (model, device). Guarded by _cache_lock so
# concurrent /infer calls don't each spawn their own load. ``model`` is either
# a plain ``torch.nn.Module`` (base encoder) or an ``FTModel`` (fine-tuned),
# and the service layer dispatches by type.
#
# LRU-bounded by ``OE_MODEL_CACHE_MAX_ENTRIES`` (default 1, laptop-safe) to
# stop the unbounded VRAM growth observed when a user clicks Nano → Tiny →
# Base → Large in succession: each load ~tripled VRAM until OOM. ``OrderedDict``
# preserves insert order; on hit we ``move_to_end`` to mark "most recently
# used", and on insert we pop oldest entries until we're at cap.
import collections  # noqa: E402  (placed near _cache for locality)
_LoadedModel = torch.nn.Module | FTModel
_cache: collections.OrderedDict[str, tuple[_LoadedModel, torch.device]] = (
    collections.OrderedDict()
)
_cache_lock = threading.Lock()


def _max_cache_entries() -> int:
    """Max number of encoder models held in the LRU cache.

    Default 1 because each base encoder is 50 MB – 3 GB on disk and ~3×
    that resident on GPU once activated; a laptop with a 24 GB GPU can
    hold one Large + working memory but not two. Operators on Azure VMs
    with 80 GB+ GPUs can bump via ``OE_MODEL_CACHE_MAX_ENTRIES``.
    Values < 1 are clamped to 1 — no cache at all is a footgun (cold
    load ~30 s, repeat for every chunked job).
    """
    import os as _os
    env = _os.environ.get("OE_MODEL_CACHE_MAX_ENTRIES")
    if env is None:
        return 1
    try:
        value = int(env)
    except ValueError:
        logger.warning(
            "OE_MODEL_CACHE_MAX_ENTRIES=%r not parseable as int — "
            "falling back to 1", env,
        )
        return 1
    return max(1, value)


def _evict_oldest_locked() -> None:
    """Pop the oldest cache entry. Caller must hold ``_cache_lock``.

    Drops the model reference (Python GC frees the host-side tensors)
    and calls ``torch.cuda.empty_cache()`` to release any cached CUDA
    blocks that were holding onto VRAM. Safe even mid-inference: empty
    cache only reclaims blocks the allocator wasn't using; in-flight
    forwards keep their tensors alive via separate references.
    """
    if not _cache:
        return
    repo_id, (model, device) = _cache.popitem(last=False)
    logger.info("evicting LRU model %s (device=%s)", repo_id, device)
    # Move to CPU first to release VRAM more aggressively. If the model
    # doesn't have .to() (shouldn't happen, but defensive), skip.
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception as e:
        logger.warning("eviction: model.to('cpu') failed for %s: %s", repo_id, e)
    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover — defensive only
            logger.warning("eviction: torch.cuda.empty_cache() failed: %s", e)

_normalizer = Normalizer(Strategy.COMPUTED)


def _assert_s2_reflectance(image_bhwtc: np.ndarray) -> None:
    """Scientific-accuracy guard rails on Sentinel-2 L2A input.

    Called before ``Normalizer(COMPUTED)`` in every forward path
    (``run_s2_inference`` + ``run_ft_inference``) so off-distribution
    input fails loudly instead of producing confident-looking but
    meaningless predictions.

    Two classes of bug this catches:

    1. **Non-finite values (NaN/inf)**. The previous ``sentinel2_fetch``
       zero-filled failed band reads — OlmoEarth happily processed those
       as "zero reflectance, very dark pixels" and emitted predictions
       the user couldn't tell from real output. That fetch path was
       hardened to reject partial failures (audit #5), but a future code
       path, an upstream format change, or double-normalization could
       re-introduce non-finite values. Fail here rather than let the
       encoder eat them.

    2. **Wrong units**. OlmoEarth expects raw L2A digital numbers
       (roughly [0, 10000], with some specular highlights reaching
       ~15000 and MSIL2A's BOA_ADD_OFFSET shifting the floor slightly
       negative). If a caller passes already-normalized [0, 1] floats
       with ``normalize=True``, the double normalization produces
       garbage. Range check catches this: a max value under 1.5 is
       almost certainly pre-normalized input by mistake.

    Raises ``ValueError`` — upstream ``start_inference`` catches this
    (via the broad ``except Exception`` in ``olmoearth_inference.py``)
    and falls back to the stub renderer, which the frontend now badges
    prominently so the user sees the failure mode.
    """
    if not np.isfinite(image_bhwtc).all():
        n_bad = int(np.count_nonzero(~np.isfinite(image_bhwtc)))
        raise ValueError(
            f"Sentinel-2 input contains {n_bad} non-finite values "
            f"(NaN/inf) out of {image_bhwtc.size} — inference would "
            f"silently corrupt outputs. Check sentinel2_fetch for partial "
            f"band reads or upstream data drift."
        )
    vmin = float(image_bhwtc.min())
    vmax = float(image_bhwtc.max())
    # Upper bound widened from 20000 → 40000 (2026-04-27): ESA's Processing
    # Baseline 04.00+ rolled out for new Sentinel-2 scenes started landing
    # values up to ~32000 in our chunks (specular highlights on water +
    # the BOA_ADD_OFFSET shift), causing every fresh fetch to fail the old
    # ceiling. The encoder's internal normalization (divide by ~10000 then
    # standardize) handles brief overshoots past the nominal range cleanly
    # — the original 20000 ceiling was meant to catch "wrong asset"
    # (B*1000 metadata bands have totally different value spaces) and
    # "already-normalized floats" (vmax < 1.5 case below), not legitimate
    # bright pixels in modern scenes. 40000 still catches the wrong-asset
    # mode (B*1000 metadata reaches mid-60000s) while letting current PC
    # scenes through.
    if vmin < -1100.0 or vmax > 40000.0:
        raise ValueError(
            f"Sentinel-2 input outside expected DN range "
            f"[min={vmin:.2f}, max={vmax:.2f}] — expected roughly "
            f"[-1000, 40000] (post-Baseline-04.00). Either the upstream "
            f"fetch returned the wrong asset, or values got rescaled "
            f"somewhere in the pipeline."
        )
    # Empty composite — distinct failure mode from "already normalized":
    # the fetch returned a raster that's entirely zero (or effectively so).
    # Common causes: STAC search hit zero scenes for the bbox/date window,
    # a cloud-cover filter was too strict, or a sliding-window tile fell
    # outside the S2 scene's footprint. Report this distinctly so the
    # caller can retry with different params instead of chasing a
    # normalization red-herring.
    if vmax < 1e-3:
        raise ValueError(
            f"Sentinel-2 input is effectively all-zero "
            f"(max={vmax:.6f}, min={vmin:.6f}) — the composite fetch "
            f"returned an empty raster. Common causes: no scenes in the "
            f"date range, cloud filter too strict, or the bbox / sliding "
            f"window is outside the scene footprint. Try a different "
            f"date_range, loosen max_cloud_cover, or use a smaller bbox."
        )
    # Strong signal for the 'already pre-normalized' case. Raw S2 scenes
    # have per-band stdev in the hundreds-to-thousands; a max under 1.5
    # across all 12 bands + a whole raster is almost certainly already
    # in [0, 1] space. Error message nudges the caller toward the fix.
    if vmax < 1.5:
        raise ValueError(
            f"Sentinel-2 input looks pre-normalized (max={vmax:.3f} << "
            f"typical DN). If you've already normalized upstream, call "
            f"``run_s2_inference(..., normalize=False)`` so the built-in "
            f"normalizer doesn't run a second time."
        )


def preferred_device() -> torch.device:
    """Return cuda if available, otherwise cpu. Resolved once per call so a
    mid-run GPU attach (rare) is picked up on the next load."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_nearest(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a 2D numpy array to ``target_shape``.

    Used by the per-pixel regression path to align a head's spatial output
    to the coarse target grid when they don't exactly match. Keeps the
    output dtype as ``float32`` since downstream colormap lookup expects
    that. Zero-dependency (no scipy) — the tile renderer is already
    numpy-heavy so we avoid adding a scipy floor to the backend install.
    """
    target_h, target_w = target_shape
    src_h, src_w = arr.shape[:2]
    if src_h == target_h and src_w == target_w:
        return arr.astype(np.float32, copy=False)
    # Index maps — nearest-neighbor via integer rescale.
    row_idx = (np.arange(target_h) * src_h / max(1, target_h)).astype(np.int32)
    col_idx = (np.arange(target_w) * src_w / max(1, target_w)).astype(np.int32)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)
    return arr[row_idx[:, None], col_idx[None, :]].astype(np.float32)


def load_encoder(
    repo_id: str, device: torch.device | None = None
) -> tuple[_LoadedModel, torch.device]:
    """Load an OlmoEarth model by HuggingFace repo id. Cached.

    Three dispatch paths:

    1. Base encoder (``allenai/OlmoEarth-v1-{Nano,Tiny,Base,Large}``) —
       routed through the olmoearth_pretrain ``ModelID`` API. Returns a
       ``LatentMIM`` whose ``.encoder`` takes a ``MaskedOlmoEarthSample``.
    2. Fine-tuned checkpoint (``...-FT-...`` repos) — routed through
       :func:`olmoearth_ft.load_ft_model`, which assembles encoder + head
       from the repo's ``model.ckpt``. Returns a :class:`FTModel` wrapper.
    3. Other ``config.json`` + ``weights.pth`` repos — handled by the
       compatibility shim ``_load_from_repo_id``.

    On cache hit the existing ``(model, device)`` tuple is returned.
    """
    target_device = device or preferred_device()
    with _cache_lock:
        hit = _cache.get(repo_id)
        if hit is not None:
            # Mark as most-recently-used. Keeps the just-touched entry
            # safe from eviction on the next insert.
            _cache.move_to_end(repo_id)
            return hit

        logger.info("loading OlmoEarth model %s to %s", repo_id, target_device)
        if repo_id in _BASE_MODEL_IDS:
            model: _LoadedModel = load_model_from_id(_BASE_MODEL_IDS[repo_id])
            model.eval()
            model.to(target_device)
        elif olmoearth_ft.is_ft_repo(repo_id):
            ft_model = olmoearth_ft.load_ft_model(repo_id, device=target_device)
            ft_model.eval()
            model = ft_model
        else:
            model = _load_from_repo_id(repo_id)
            model.eval()
            model.to(target_device)
        # LRU eviction BEFORE insert so we never exceed the cap. The new
        # entry slots in at the most-recently-used end automatically.
        max_entries = _max_cache_entries()
        while len(_cache) >= max_entries:
            _evict_oldest_locked()
        _cache[repo_id] = (model, target_device)
        return model, target_device


def clear_cache() -> None:
    """Drop all cached models. Used by tests + the unload endpoint."""
    with _cache_lock:
        _cache.clear()


def loaded_repo_ids() -> list[str]:
    """Return repo_ids of every model currently resident in the process cache.

    Cheap read-under-lock — used by the ``/olmoearth/loaded-models`` route
    to let the frontend badge demo-pair buttons with a "warm (~3 s)" vs
    "cold (~30 s)" expectation BEFORE the user clicks. The list is
    in-memory only (not disk cache state); a repo_id here means
    ``load_encoder()`` will hit the fast path and skip the 2–10 s
    safetensors re-read.
    """
    with _cache_lock:
        return sorted(_cache.keys())


@dataclass(frozen=True)
class InferenceResult:
    """Dense per-patch output of an OlmoEarth encoder pass.

    ``embedding`` shape is ``(H', W', D)`` in numpy float32, where ``H' =
    H // patch_size`` and ``D`` is the encoder embedding dim (128 for
    Nano/Tiny, 384 for Base, 768 for Large). ``scalar`` is a ``(H', W')``
    float32 raster derived from the embedding (PCA projection onto the 1st
    principal component, rescaled to [0, 1]) — suitable for colormap display.
    """

    embedding: np.ndarray  # (H', W', D)
    scalar: np.ndarray     # (H', W'), [0, 1]
    patch_size: int
    embedding_dim: int


@dataclass(frozen=True)
class FTInferenceResult:
    """Task-specific output of a fine-tuned OlmoEarth forward pass.

    For every task type we fill in ``scalar`` — a ``(H', W')`` float32 raster
    in ``[0, 1]`` that the tile renderer colormaps directly — plus task-
    specific fields:

    - **classification** (scene-level, e.g. ForestLossDriver 10-class):
      uniform raster at ``argmax_prob``; ``class_raster`` is a constant
      raster of the argmax class index; ``class_probs`` is the (C,) per-
      class probability vector.
    - **segmentation** (per-patch, e.g. AWF 10-class, Mangrove 4-class):
      ``scalar`` is the argmax-class probability at each patch;
      ``class_raster`` is a ``(H', W')`` int32 array of argmax class indices.
    - **regression** (e.g. LFMC): ``scalar`` is the per-scene regression
      value normalized via ``value_range``; ``prediction_value`` is the raw
      predicted float.
    """

    task_type: str                       # classification / segmentation / regression
    scalar: np.ndarray                   # (H', W'), [0, 1] — colormap-ready
    class_raster: np.ndarray | None      # (H', W') int32 of argmax class index
    class_probs: np.ndarray | None       # (C,) — scene-level only
    class_names: list[str]
    class_names_tentative: bool
    class_colors: list[str] | None       # published hex colors if available
    colormap: str
    units: str | None
    prediction_value: float | None
    num_classes: int
    patch_size: int
    repo_id: str
    decoder_key: str


def _normalize_timestamps(
    timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    n_periods: int,
) -> list[tuple[int, int, int]]:
    """Accept a single (d, m, y) tuple OR a length-T list; return length-T list.

    Single-tuple inputs are replicated across T (matches the legacy
    single-scene inference path's behavior). Length-T lists are returned
    unchanged after a length check.
    """
    if isinstance(timestamp_dmy, tuple):
        return [timestamp_dmy] * n_periods
    if len(timestamp_dmy) != n_periods:
        raise ValueError(
            f"timestamp_dmy list length {len(timestamp_dmy)} does not "
            f"match T={n_periods} from the input image"
        )
    return list(timestamp_dmy)


def run_s2_inference(
    model: torch.nn.Module,
    image_bhwtc: np.ndarray,
    timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    patch_size: int = 4,
    device: torch.device | None = None,
    normalize: bool = True,
) -> InferenceResult:
    """Run one Sentinel-2 L2A forward pass and return a prediction raster.

    Args:
        model: an OlmoEarth model returned by :func:`load_encoder`.
        image_bhwtc: Sentinel-2 image in ``(B=1, H, W, T, C=12)`` layout in
            the band order from ``Modality.SENTINEL2_L2A.band_order``
            (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09).
            ``T`` may be 1 (legacy single-scene path) or N (temporal stack
            from :func:`fetch_s2_temporal_stack`). Raw DN reflectance; pass
            ``normalize=False`` if it's already been pushed through
            ``Normalizer``.
        timestamp_dmy: either a single ``(day 1-31, month 0-11, year)`` tuple
            (replicated across T) or a length-T list — one tuple per period
            mosaic. Per the Inference-Quickstart convention.
        patch_size: 1-8 per the official quickstart; smaller is higher-res
            but more GPU time. 4 matches the quickstart default.
        device: override device; defaults to the one the model was loaded on.
        normalize: apply ``Normalizer(Strategy.COMPUTED)`` before the forward
            pass. Set False if the caller already normalized.

    Returns an :class:`InferenceResult` holding both the raw per-patch
    embedding tensor and a 2D scalar raster ready for colormap rendering.
    """
    if image_bhwtc.ndim != 5 or image_bhwtc.shape[0] != 1:
        raise ValueError(
            f"expected BHWTC with B=1, got shape {image_bhwtc.shape}"
        )
    _, h, w, t_dim, c = image_bhwtc.shape
    if c != 12:
        raise ValueError(f"Sentinel-2 L2A needs 12 bands, got C={c}")
    if t_dim < 1:
        raise ValueError(f"Sentinel-2 input needs T>=1, got T={t_dim}")
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"image H={h} W={w} must be divisible by patch_size={patch_size}"
        )

    timestamps_list = _normalize_timestamps(timestamp_dmy, t_dim)

    target_device = device
    if target_device is None:
        target_device = next(model.parameters()).device

    img = image_bhwtc
    if normalize:
        _assert_s2_reflectance(img)  # scientific-accuracy guard
        img = _normalizer.normalize(Modality.SENTINEL2_L2A, img)
        if not np.isfinite(img).all():
            raise ValueError(
                "Sentinel-2 normalization produced non-finite values — "
                "bug in olmoearth_pretrain.Normalizer or its COMPUTED stats."
            )
    img_t = torch.as_tensor(img, dtype=torch.float32, device=target_device)

    mask = torch.full(
        (1, h, w, t_dim, 3),
        float(MaskValue.ONLINE_ENCODER.value),
        dtype=torch.float32,
        device=target_device,
    )
    ts = torch.tensor(
        [[[int(d), int(m), int(y)] for d, m, y in timestamps_list]],
        dtype=torch.long,
        device=target_device,
    )

    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=img_t,
        sentinel2_l2a_mask=mask,
        timestamps=ts,
    )

    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=patch_size)
    tokens = out["tokens_and_masks"].sentinel2_l2a  # (1, H', W', T, S, D)
    # Average over the time and band-set dims to collapse to (H', W', D).
    pooled = tokens.mean(dim=(3, 4)).squeeze(0).cpu().numpy().astype(np.float32)

    scalar = _pca_to_scalar(pooled)
    return InferenceResult(
        embedding=pooled,
        scalar=scalar,
        patch_size=patch_size,
        embedding_dim=int(pooled.shape[-1]),
    )


def run_ft_inference(
    model: FTModel,
    image_bhwtc: np.ndarray,
    timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    patch_size: int = 4,
    device: torch.device | None = None,
    normalize: bool = True,
) -> FTInferenceResult:
    """Run a fine-tuned OlmoEarth model and return task-specific rasters.

    Same S2 preprocessing as :func:`run_s2_inference` — normalize, build a
    :class:`MaskedOlmoEarthSample`, run the full encoder → head chain — but
    the output layout depends on the head's task type. ``T`` may be 1 (legacy
    single-scene fallback) or N (PER_PERIOD_MOSAIC temporal stack, which is
    what every FT head was actually trained on — see FT_TASK_METADATA
    ``input_spec``).
    """
    if image_bhwtc.ndim != 5 or image_bhwtc.shape[0] != 1:
        raise ValueError(
            f"expected BHWTC with B=1, got shape {image_bhwtc.shape}"
        )
    _, h, w, t_dim, c = image_bhwtc.shape
    if c != 12:
        raise ValueError(f"Sentinel-2 L2A needs 12 bands, got C={c}")
    if t_dim < 1:
        raise ValueError(f"Sentinel-2 input needs T>=1, got T={t_dim}")
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"image H={h} W={w} must be divisible by patch_size={patch_size}"
        )

    timestamps_list = _normalize_timestamps(timestamp_dmy, t_dim)

    target_device = device or model.device
    img = image_bhwtc
    if normalize:
        _assert_s2_reflectance(img)  # scientific-accuracy guard (same contract as run_s2_inference)
        img = _normalizer.normalize(Modality.SENTINEL2_L2A, img)
        if not np.isfinite(img).all():
            raise ValueError(
                "Sentinel-2 normalization produced non-finite values — "
                "bug in olmoearth_pretrain.Normalizer or its COMPUTED stats."
            )
    img_t = torch.as_tensor(img, dtype=torch.float32, device=target_device)

    mask = torch.full(
        (1, h, w, t_dim, 3),
        float(MaskValue.ONLINE_ENCODER.value),
        dtype=torch.float32,
        device=target_device,
    )
    ts = torch.tensor(
        [[[int(d), int(m), int(y)] for d, m, y in timestamps_list]],
        dtype=torch.long,
        device=target_device,
    )
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=img_t,
        sentinel2_l2a_mask=mask,
        timestamps=ts,
    )

    out = model.forward(sample, patch_size=patch_size)
    task_type: str = out["task_type"]  # type: ignore[assignment]
    spec = model.spec
    md = model.metadata or {}
    class_names, tentative = olmoearth_ft.class_names_for(model.repo_id, spec.num_classes)
    class_colors = olmoearth_ft.class_colors_for(model.repo_id, spec.num_classes)

    hi_h = h // patch_size
    hi_w = w // patch_size

    if task_type == "regression":
        # Two regression shapes are possible depending on which head the
        # FT loader rebuilt:
        #   * Scalar ``(B,)``          — scene-level (e.g. _LinearRegressionHead).
        #   * Per-pixel ``(B, 1, H', W')`` — UNet / conv-stack (LFMC). The head
        #     preserves spatial dims so we can render a true heatmap instead
        #     of painting a constant-fill tile.
        # Normalize via the task's declared value_range so the colormap maps
        # raw values (e.g. 30–200% LFMC) to [0, 1] meaningfully either way.
        pred_t = out["prediction"].detach().cpu().numpy()
        vr = md.get("value_range") or [0.0, 1.0]
        lo, hi = float(vr[0]), float(vr[1])
        denom = max(1e-9, hi - lo)
        if pred_t.ndim >= 3:
            # Per-pixel regression — squeeze the batch + channel dims.
            arr = pred_t[0]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]                                   # (H', W')
            # Clamp to value_range then map to [0, 1] so the existing
            # colormap renderer receives the same scale as scalar regression.
            normed = np.clip((arr.astype(np.float32) - lo) / denom, 0.0, 1.0)
            # If the head's spatial output doesn't match the coarse target
            # grid (edge cases where encoder patching + head stride
            # combine differently), resize via nearest-neighbor so we
            # don't silently crop or pad. scipy is optional — use a
            # dumb index-map when it's absent.
            scalar = _resize_nearest(normed, (hi_h, hi_w))
            # Summary scalar for the legend pill — mean over valid pixels.
            pred = float(arr.reshape(-1).mean())
        else:
            pred = float(pred_t.reshape(-1)[0])
            scalar_val = max(0.0, min(1.0, (pred - lo) / denom))
            scalar = np.full((hi_h, hi_w), scalar_val, dtype=np.float32)
        return FTInferenceResult(
            task_type="regression",
            scalar=scalar,
            class_raster=None,
            class_probs=None,
            class_names=class_names,
            class_names_tentative=tentative,
            class_colors=class_colors,
            colormap=md.get("colormap", "embedding"),
            units=md.get("units"),
            prediction_value=pred,
            num_classes=spec.num_classes,
            patch_size=patch_size,
            repo_id=model.repo_id,
            decoder_key=spec.decoder_key,
        )

    if task_type == "classification":
        probs_t = out["probs"]  # (B, C)
        probs = probs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        argmax = int(probs.argmax())
        scalar = np.full((hi_h, hi_w), float(probs[argmax]), dtype=np.float32)
        class_raster = np.full((hi_h, hi_w), argmax, dtype=np.int32)
        return FTInferenceResult(
            task_type="classification",
            scalar=scalar,
            class_raster=class_raster,
            class_probs=probs,
            class_names=class_names,
            class_names_tentative=tentative,
            class_colors=class_colors,
            colormap=md.get("colormap", "embedding"),
            units=md.get("units"),
            prediction_value=None,
            num_classes=spec.num_classes,
            patch_size=patch_size,
            repo_id=model.repo_id,
            decoder_key=spec.decoder_key,
        )

    # segmentation
    probs_t = out["probs"]  # (B, C, H', W')
    probs = probs_t.detach().cpu().numpy()[0].astype(np.float32)  # (C, H', W')
    class_raster_seg = probs.argmax(axis=0).astype(np.int32)       # (H', W')
    # scalar = probability of the argmax class at each patch — lets the
    # existing single-channel tile renderer paint a confidence map that
    # the user can pair with class_raster for discrete overlays later.
    max_prob = probs.max(axis=0).astype(np.float32)                # (H', W')
    return FTInferenceResult(
        task_type="segmentation",
        scalar=max_prob,
        class_raster=class_raster_seg,
        class_probs=None,
        class_names=class_names,
        class_names_tentative=tentative,
        class_colors=class_colors,
        colormap=md.get("colormap", "embedding"),
        units=md.get("units"),
        prediction_value=None,
        num_classes=spec.num_classes,
        patch_size=patch_size,
        repo_id=model.repo_id,
        decoder_key=spec.decoder_key,
    )


def run_ft_pre_post_inference(
    model: FTModel,
    pre_image_bhwtc: np.ndarray,
    post_image_bhwtc: np.ndarray,
    pre_timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    post_timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    patch_size: int = 4,
    device: torch.device | None = None,
    normalize: bool = True,
) -> FTInferenceResult:
    """Pre/post change-detection forward pass for ForestLossDriver-style heads.

    The decoder expects 1536-channel features = concat([pre_768, post_768]).
    We mirror rslearn's ``SimpleTimeSeries(groups=[[0],[1]])`` wrapper:

      1. Encode pre stack independently → tokens_pre (1, H', W', T_pre, S, 768)
      2. Encode post stack independently → tokens_post (1, H', W', T_post, S, 768)
      3. Pool each over (T, S) so per-group features are time-averaged
      4. Concatenate along the feature dim → (1, H', W', 1536)
      5. Add length-1 T and S dims and feed to ``model.head`` directly,
         bypassing FTModel.forward (which would re-run the encoder)

    The post-processing path (regression / classification / segmentation)
    matches :func:`run_ft_inference` 1:1.
    """
    for name, arr in (("pre", pre_image_bhwtc), ("post", post_image_bhwtc)):
        if arr.ndim != 5 or arr.shape[0] != 1:
            raise ValueError(f"expected {name} BHWTC with B=1, got shape {arr.shape}")
        if arr.shape[-1] != 12:
            raise ValueError(f"{name} S2 needs 12 bands, got C={arr.shape[-1]}")
    if pre_image_bhwtc.shape[1:3] != post_image_bhwtc.shape[1:3]:
        raise ValueError(
            f"pre H/W {pre_image_bhwtc.shape[1:3]} must equal post H/W "
            f"{post_image_bhwtc.shape[1:3]} — pre/post stacks must share spatial grid"
        )
    _, h, w, t_pre, _ = pre_image_bhwtc.shape
    t_post = post_image_bhwtc.shape[3]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"image H={h} W={w} must be divisible by patch_size={patch_size}"
        )

    target_device = device or model.device

    def _encode(img_bhwtc: np.ndarray, ts_in: Any) -> torch.Tensor:
        t_dim = img_bhwtc.shape[3]
        ts_list = _normalize_timestamps(ts_in, t_dim)
        img = img_bhwtc
        if normalize:
            _assert_s2_reflectance(img)
            img = _normalizer.normalize(Modality.SENTINEL2_L2A, img)
            if not np.isfinite(img).all():
                raise ValueError(
                    "Sentinel-2 normalization produced non-finite values — "
                    "bug in olmoearth_pretrain.Normalizer or its COMPUTED stats."
                )
        img_t = torch.as_tensor(img, dtype=torch.float32, device=target_device)
        mask = torch.full(
            (1, h, w, t_dim, 3),
            float(MaskValue.ONLINE_ENCODER.value),
            dtype=torch.float32,
            device=target_device,
        )
        ts = torch.tensor(
            [[[int(d), int(m), int(y)] for d, m, y in ts_list]],
            dtype=torch.long,
            device=target_device,
        )
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=img_t,
            sentinel2_l2a_mask=mask,
            timestamps=ts,
        )
        with torch.no_grad():
            out = model.encoder_parent.encoder(
                sample, fast_pass=True, patch_size=patch_size,
            )
        # (1, H', W', T, S, D)
        return out["tokens_and_masks"].sentinel2_l2a

    tokens_pre = _encode(pre_image_bhwtc, pre_timestamp_dmy)
    tokens_post = _encode(post_image_bhwtc, post_timestamp_dmy)

    # Pool T and S so each group collapses to (1, H', W', D), then concat
    # along D so the head sees (1, H', W', 1, 1, 2D) — matches the decoder's
    # expected 1536-channel input.
    pre_pooled = tokens_pre.mean(dim=(3, 4))    # (1, H', W', D)
    post_pooled = tokens_post.mean(dim=(3, 4))  # (1, H', W', D)
    combined = torch.cat([pre_pooled, post_pooled], dim=-1)  # (1, H', W', 2D)
    tokens_in = combined.unsqueeze(3).unsqueeze(4)  # (1, H', W', 1, 1, 2D)

    with torch.no_grad():
        logits = model.head(tokens_in)

    spec = model.spec
    md = model.metadata or {}
    class_names, tentative = olmoearth_ft.class_names_for(model.repo_id, spec.num_classes)
    class_colors = olmoearth_ft.class_colors_for(model.repo_id, spec.num_classes)

    hi_h = h // patch_size
    hi_w = w // patch_size

    if spec.task_type == "regression":
        # Regression in pre/post mode is unusual but handled symmetrically
        # with run_ft_inference for shape-correctness. ForestLossDriver
        # itself is classification, so this branch is defensive.
        pred_t = logits.detach().cpu().numpy()
        vr = md.get("value_range") or [0.0, 1.0]
        lo, hi = float(vr[0]), float(vr[1])
        denom = max(1e-9, hi - lo)
        if pred_t.ndim >= 3:
            arr = pred_t[0]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            normed = np.clip((arr.astype(np.float32) - lo) / denom, 0.0, 1.0)
            scalar = _resize_nearest(normed, (hi_h, hi_w))
            pred = float(arr.reshape(-1).mean())
        else:
            pred = float(pred_t.reshape(-1)[0])
            scalar_val = max(0.0, min(1.0, (pred - lo) / denom))
            scalar = np.full((hi_h, hi_w), scalar_val, dtype=np.float32)
        return FTInferenceResult(
            task_type="regression",
            scalar=scalar,
            class_raster=None,
            class_probs=None,
            class_names=class_names,
            class_names_tentative=tentative,
            class_colors=class_colors,
            colormap=md.get("colormap", "embedding"),
            units=md.get("units"),
            prediction_value=pred,
            num_classes=spec.num_classes,
            patch_size=patch_size,
            repo_id=model.repo_id,
            decoder_key=spec.decoder_key,
        )

    if spec.task_type == "classification":
        # Scene-level: head returns (B, num_classes) logits.
        import torch.nn.functional as F  # noqa: PLC0415
        probs_t = F.softmax(logits, dim=-1)
        probs = probs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        argmax = int(probs.argmax())
        scalar = np.full((hi_h, hi_w), float(probs[argmax]), dtype=np.float32)
        class_raster = np.full((hi_h, hi_w), argmax, dtype=np.int32)
        return FTInferenceResult(
            task_type="classification",
            scalar=scalar,
            class_raster=class_raster,
            class_probs=probs,
            class_names=class_names,
            class_names_tentative=tentative,
            class_colors=class_colors,
            colormap=md.get("colormap", "embedding"),
            units=md.get("units"),
            prediction_value=None,
            num_classes=spec.num_classes,
            patch_size=patch_size,
            repo_id=model.repo_id,
            decoder_key=spec.decoder_key,
        )

    # Segmentation — defensive parity with run_ft_inference.
    import torch.nn.functional as F  # noqa: PLC0415
    probs_t = F.softmax(logits, dim=1)
    probs = probs_t.detach().cpu().numpy()[0].astype(np.float32)
    class_raster_seg = probs.argmax(axis=0).astype(np.int32)
    max_prob = probs.max(axis=0).astype(np.float32)
    return FTInferenceResult(
        task_type="segmentation",
        scalar=max_prob,
        class_raster=class_raster_seg,
        class_probs=None,
        class_names=class_names,
        class_names_tentative=tentative,
        class_colors=class_colors,
        colormap=md.get("colormap", "embedding"),
        units=md.get("units"),
        prediction_value=None,
        num_classes=spec.num_classes,
        patch_size=patch_size,
        repo_id=model.repo_id,
        decoder_key=spec.decoder_key,
    )


def run_ft_pre_post_tiled_inference(
    model: FTModel,
    pre_image_bhwtc: np.ndarray,
    post_image_bhwtc: np.ndarray,
    pre_timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    post_timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    window_size: int = 64,
    patch_size: int = 4,
    device: torch.device | None = None,
    normalize: bool = True,
) -> FTInferenceResult:
    """Sliding-window pre/post inference for ForestLossDriver-style heads.

    The conv-pool-fc head was trained on ~64 × 64 px windows; running it
    on a single 500-px chunk emits ONE scene-level class for the whole
    chunk. This tiled variant runs the head over a grid of non-
    overlapping ``window_size``-pixel windows, each encoded as a
    pre/post pair → 1536-channel concat → single class label.

    Output is a (n_rows × win_patches, n_cols × win_patches) class
    raster at patch resolution — same convention as
    :func:`run_ft_tiled_inference` so the chunked orchestrator's stitch
    and upsample paths work unchanged.

    Edge windows that don't fit a full ``window_size`` get dropped on
    the bottom + right; output is trimmed to the largest aligned grid.
    """
    for name, arr in (("pre", pre_image_bhwtc), ("post", post_image_bhwtc)):
        if arr.ndim != 5 or arr.shape[0] != 1:
            raise ValueError(f"{name}: expected BHWTC with B=1, got shape {arr.shape}")
        if arr.shape[-1] != 12:
            raise ValueError(f"{name}: S2 needs 12 bands, got C={arr.shape[-1]}")
    if pre_image_bhwtc.shape[1:3] != post_image_bhwtc.shape[1:3]:
        raise ValueError(
            f"pre H/W {pre_image_bhwtc.shape[1:3]} must equal post H/W "
            f"{post_image_bhwtc.shape[1:3]}"
        )
    _, H, W, _, _ = pre_image_bhwtc.shape
    if window_size % patch_size != 0:
        raise ValueError(
            f"window_size={window_size} must be divisible by patch_size={patch_size}"
        )

    n_rows = H // window_size
    n_cols = W // window_size
    if n_rows < 1 or n_cols < 1:
        # Bbox too small for windowing — single-window path is identical
        # to the existing non-tiled pre/post inference, just on the trimmed
        # tensor that fits one window.
        h_trim = max(patch_size, (H // patch_size) * patch_size)
        w_trim = max(patch_size, (W // patch_size) * patch_size)
        return run_ft_pre_post_inference(
            model,
            pre_image_bhwtc[:, :h_trim, :w_trim, :, :],
            post_image_bhwtc[:, :h_trim, :w_trim, :, :],
            pre_timestamp_dmy,
            post_timestamp_dmy,
            patch_size=patch_size,
            device=device,
            normalize=normalize,
        )

    H_trim = n_rows * window_size
    W_trim = n_cols * window_size
    pre_image = pre_image_bhwtc[:, :H_trim, :W_trim, :, :]
    post_image = post_image_bhwtc[:, :H_trim, :W_trim, :, :]

    win_patches = window_size // patch_size
    out_h = n_rows * win_patches
    out_w = n_cols * win_patches

    class_raster = np.zeros((out_h, out_w), dtype=np.int32)
    scalar_raster = np.zeros((out_h, out_w), dtype=np.float32)
    any_class_output = False
    skipped = 0

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * window_size
            x0 = j * window_size
            pre_w = pre_image[:, y0 : y0 + window_size, x0 : x0 + window_size, :, :]
            post_w = post_image[:, y0 : y0 + window_size, x0 : x0 + window_size, :, :]
            # Skip all-zero windows (S2 off-footprint or empty composite).
            # Either side empty is enough to flag it — the encoder needs
            # both halves of the pre/post pair to produce a real prediction.
            if float(pre_w.max()) < 1e-3 or float(post_w.max()) < 1e-3:
                skipped += 1
                continue
            res = run_ft_pre_post_inference(
                model, pre_w, post_w,
                pre_timestamp_dmy, post_timestamp_dmy,
                patch_size=patch_size, device=device, normalize=normalize,
            )
            y0p = i * win_patches
            x0p = j * win_patches
            scalar_raster[y0p : y0p + win_patches, x0p : x0p + win_patches] = res.scalar
            if res.class_raster is not None:
                class_raster[y0p : y0p + win_patches, x0p : x0p + win_patches] = res.class_raster
                any_class_output = True

    if skipped == n_rows * n_cols:
        raise ValueError(
            f"pre/post tiled: every {n_rows * n_cols} window was empty — "
            f"the AOI may be entirely ocean / off-footprint, or one of the "
            f"pre/post groups returned no usable scenes."
        )

    md = model.metadata or {}
    class_names, tentative = olmoearth_ft.class_names_for(model.repo_id, model.spec.num_classes)
    class_colors = olmoearth_ft.class_colors_for(model.repo_id, model.spec.num_classes)
    src_task = model.spec.task_type
    effective_task = "segmentation" if src_task == "classification" else src_task
    return FTInferenceResult(
        task_type=effective_task,
        scalar=scalar_raster,
        class_raster=class_raster if any_class_output else None,
        class_probs=None,
        class_names=class_names,
        class_names_tentative=tentative,
        class_colors=class_colors,
        colormap=md.get("colormap", "embedding"),
        units=md.get("units"),
        prediction_value=None,
        num_classes=model.spec.num_classes,
        patch_size=patch_size,
        repo_id=model.repo_id,
        decoder_key=model.spec.decoder_key,
    )


def run_ft_tiled_inference(
    model: FTModel,
    image_bhwtc: np.ndarray,
    timestamp_dmy: tuple[int, int, int] | list[tuple[int, int, int]],
    window_size: int = 32,
    patch_size: int | None = None,
    device: torch.device | None = None,
    normalize: bool = True,
) -> FTInferenceResult:
    """Run the FT model on a grid of non-overlapping ``window_size`` tiles and
    stitch the outputs into a spatially-varying prediction raster.

    Why this exists: the FT heads were trained on small tiles (~32 px), so
    feeding a 256 px bbox through in a single forward pass moves the
    distribution off-policy. For **scene-level** tasks (classification,
    regression) this is also the only way to get a non-uniform output — a
    single forward produces one class/value for the whole bbox, while a
    tiled forward gives one class/value per tile, yielding a real map.

    Behaviour by source task type:
      - segmentation → stitch the per-window ``class_raster`` strips;
        task_type stays ``"segmentation"``.
      - classification → each window contributes a uniform (window_patches,
        window_patches) block of its argmax class, so the stitched output
        is a coarse class map. task_type is reported as ``"segmentation"``
        since the output is now a spatial raster.
      - regression → each window contributes a block at its predicted value
        (normalized via ``value_range``); task_type reports ``"regression"``
        but ``scalar`` varies spatially.

    Tiles that are too small for a full ``window_size`` are dropped on the
    bottom / right edges — the output tensor is trimmed to ``(n_rows *
    window_size, n_cols * window_size)`` in input pixels.
    """
    if image_bhwtc.ndim != 5 or image_bhwtc.shape[0] != 1:
        raise ValueError(
            f"expected BHWTC with B=1, got shape {image_bhwtc.shape}"
        )
    _, H, W, T_dim, C = image_bhwtc.shape
    if C != 12:
        raise ValueError(f"Sentinel-2 L2A needs 12 bands, got C={C}")
    if T_dim < 1:
        raise ValueError(f"Sentinel-2 input needs T>=1, got T={T_dim}")

    md = model.metadata or {}
    effective_patch = patch_size or md.get("patch_size") or 4
    if window_size % effective_patch != 0:
        raise ValueError(
            f"window_size={window_size} must be divisible by patch_size={effective_patch}"
        )

    n_rows = H // window_size
    n_cols = W // window_size
    if n_rows < 1 or n_cols < 1:
        # Bbox too small — single-window path is identical to run_ft_inference.
        return run_ft_inference(
            model, image_bhwtc, timestamp_dmy,
            patch_size=effective_patch, device=device, normalize=normalize,
        )

    H_trim = n_rows * window_size
    W_trim = n_cols * window_size
    image = image_bhwtc[:, :H_trim, :W_trim, :, :]

    win_patches = window_size // effective_patch
    out_h = n_rows * win_patches
    out_w = n_cols * win_patches

    class_raster: np.ndarray | None = np.zeros((out_h, out_w), dtype=np.int32)
    scalar_raster = np.zeros((out_h, out_w), dtype=np.float32)

    # Track whether any window produced real class output so we can null
    # class_raster for pure-regression tasks.
    any_class_output = False
    # Track windows that were skipped as nodata so the caller can distinguish
    # "model predicted class 0 everywhere" from "most of this bbox is ocean /
    # off-footprint and we couldn't infer at all." The scalar raster stays at
    # 0 for those cells; the class raster leaves them at 0 but `any_class_output`
    # only flips to True if at least one window actually ran.
    skipped = 0

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * window_size
            x0 = j * window_size
            window = image[:, y0 : y0 + window_size, x0 : x0 + window_size, :, :]
            # Skip all-zero windows (S2 off-footprint, deep-ocean no-data,
            # or a stripe outside the scene). The per-window validator in
            # `_validate_s2_dn_range` would otherwise raise and abort the
            # entire tiled inference, so one empty corner of the bbox
            # poisoned the whole raster. Leave the output cells at 0 —
            # frontends can read `nodata_windows` to gray them out.
            if float(window.max()) < 1e-3:
                skipped += 1
                continue
            res = run_ft_inference(
                model, window, timestamp_dmy,
                patch_size=effective_patch, device=device, normalize=normalize,
            )
            y0p = i * win_patches
            x0p = j * win_patches
            scalar_raster[y0p : y0p + win_patches, x0p : x0p + win_patches] = res.scalar
            if res.class_raster is not None:
                class_raster[y0p : y0p + win_patches, x0p : x0p + win_patches] = res.class_raster
                any_class_output = True

    # All windows were skipped — the whole fetched composite was effectively
    # empty. Surface this as an error rather than returning a zeros raster
    # that the UI would happily paint as "class 0 everywhere."
    if skipped == n_rows * n_cols:
        raise ValueError(
            f"Sentinel-2 composite was empty across all {n_rows * n_cols} "
            f"sliding windows — the bbox may be entirely ocean or the "
            f"fetch returned no valid data. Try a different date_range, "
            f"loosen max_cloud_cover, or pick a bbox with land coverage."
        )

    if not any_class_output:
        class_raster = None

    # Effective task type surfaced to the UI. Classification tasks become
    # segmentation once we have per-window class variation; regression stays
    # regression but with a spatially varying scalar.
    src_task = model.spec.task_type
    if src_task == "classification":
        effective_task = "segmentation"
    else:
        effective_task = src_task

    class_names, tentative = olmoearth_ft.class_names_for(model.repo_id, model.spec.num_classes)
    class_colors = olmoearth_ft.class_colors_for(model.repo_id, model.spec.num_classes)

    return FTInferenceResult(
        task_type=effective_task,
        scalar=scalar_raster,
        class_raster=class_raster,
        class_probs=None,
        class_names=class_names,
        class_names_tentative=tentative,
        class_colors=class_colors,
        colormap=md.get("colormap", "embedding"),
        units=md.get("units"),
        prediction_value=None,
        num_classes=model.spec.num_classes,
        patch_size=effective_patch,
        repo_id=model.repo_id,
        decoder_key=model.spec.decoder_key,
    )


def _pca_to_scalar(embedding_hwd: np.ndarray) -> np.ndarray:
    """Project a ``(H, W, D)`` embedding onto its first principal component
    and rescale to ``[0, 1]``. Used as a default "show the encoder's view of
    this scene" visualization when no fine-tuned head is attached.
    """
    h, w, d = embedding_hwd.shape
    flat = embedding_hwd.reshape(h * w, d)
    centered = flat - flat.mean(axis=0, keepdims=True)
    # Full SVD on (N, D) — N = H*W is small (usually <=16384), cheap on CPU.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vh[0]
    lo, hi = float(pc1.min()), float(pc1.max())
    if hi - lo < 1e-9:
        return np.zeros((h, w), dtype=np.float32)
    return ((pc1 - lo) / (hi - lo)).reshape(h, w).astype(np.float32)


def _smooth_seams(
    raster: np.ndarray, *, sigma: float, nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a small Gaussian blur to bridge encoder-chunk seams in a
    PCA / similarity raster while preserving most spatial detail.

    The visible "grid lines" in PCA / similarity output come from the
    encoder's attention context — patches at the edge of one chunk see
    only that chunk's tokens, while the same physical pixel near the
    edge of the neighbour chunk sees a different attention context.
    The embedding values shift subtly at the seam, which becomes a
    visible color jump after PCA / cosine projection.

    A tight Gaussian blur (default σ ≈ 1.5 patches ≈ 60 m in pixel
    space at patch_size=4 + 10 m GSD) bridges 2-3 patch wide seams
    while only smearing real spatial features by the same ~60 m —
    well below the scale of any meaningful landscape boundary.

    ``nodata_mask`` (True = nodata) is preserved bit-for-bit; the blur
    pretends nodata pixels are 0 during the convolution, then the mask
    is reapplied afterward so AOI edges don't "bleed" colour into
    chunks that never produced data.

    Set ``sigma <= 0`` (or ``OE_PCA_SMOOTH_SIGMA=0``) to disable; this
    used to be the implicit default and is preserved for callers that
    want the raw chunk-grid view.
    """
    if sigma <= 0:
        return raster
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    work = raster.astype(np.float32, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = 0.0

    if work.ndim == 3:
        # Smooth each channel independently in (H, W); leave the channel
        # dim alone (we don't want PC bands bleeding into each other).
        for ch in range(work.shape[-1]):
            work[..., ch] = gaussian_filter(work[..., ch], sigma=sigma)
    else:
        work = gaussian_filter(work, sigma=sigma)

    if nodata_mask is not None:
        if work.ndim == 3:
            work[nodata_mask] = 0.0
        else:
            work[nodata_mask] = 0.0
    return work


def _smooth_sigma_default(env_var: str, fallback: float) -> float:
    """Read a smoothing sigma from env, with a safe fallback. ``0`` or
    negative disables smoothing; non-numeric falls back."""
    raw = os.environ.get(env_var)
    if raw is None:
        return fallback
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning("ignored bad %s=%r — using default %s", env_var, raw, fallback)
        return fallback


def cosine_similarity_map(
    embedding_hwd: np.ndarray,
    query_vec: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> np.ndarray:
    """Compute per-patch cosine similarity against a query vector.

    The Ai2 OlmoEarth tutorial uses this as the "find more like this"
    workflow — pick a patch, compute its cosine similarity against every
    other patch, render the result as a heatmap. Works globally (no
    labels required).

    Args:
        embedding_hwd: ``(H, W, D)`` float32 embedding from the chunked
            export pipeline. Nodata patches (all-zero vectors) get a
            similarity of 0 so the tile renderer draws them at the dark
            end of the colormap rather than propagating NaN.
        query_vec: ``(D,)`` query vector — typically ``embedding_hwd``
            at the clicked pixel, or the mean over a small window for
            noise robustness.
        smooth_sigma: Gaussian sigma (in patch units) applied to the
            similarity raster after projection, to mask encoder-chunk
            seams. ``None`` reads ``OE_SIM_SMOOTH_SIGMA`` env var
            (default 1.5). Set ``0`` to disable.

    Returns:
        ``(H, W)`` float32 in ``[0, 1]``. The raw cosine range is
        ``[-1, 1]`` but we rescale to ``[0, 1]`` for direct colormap
        rendering (0 = most dissimilar, 1 = identical to query).
    """
    h, w, d = embedding_hwd.shape
    if query_vec.shape != (d,):
        raise ValueError(
            f"query_vec shape {query_vec.shape} doesn't match embedding "
            f"last dim {d}"
        )

    flat = embedding_hwd.reshape(h * w, d)
    # Nodata: pixels where every D dim is exactly 0 (untouched chunk).
    nodata_flat = ~np.any(flat != 0, axis=-1)

    # Normalize both sides. Epsilon prevents div-by-zero on all-zero
    # rows without biasing the valid rows meaningfully.
    flat_norm = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-9)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    cos_sim = flat_norm @ q_norm                  # (H*W,) in [-1, 1]
    # Rescale [-1, 1] → [0, 1]. Anchors at 0.5 = perpendicular, which
    # is semantically "unrelated" and reads as mid-colormap.
    sim01 = ((cos_sim + 1.0) / 2.0).astype(np.float32)
    # Force nodata pixels to 0 so they render at the dark end rather
    # than leaking a spurious "somewhat similar" value from the zero
    # vector's dot product with the query.
    sim01[nodata_flat] = 0.0
    sim_2d = sim01.reshape(h, w)
    nodata_2d = nodata_flat.reshape(h, w)
    sigma = smooth_sigma if smooth_sigma is not None else _smooth_sigma_default(
        "OE_SIM_SMOOTH_SIGMA", 1.5,
    )
    return _smooth_seams(sim_2d, sigma=sigma, nodata_mask=nodata_2d)


def pca_to_rgb(
    embedding_hwd: np.ndarray, *, smooth_sigma: float | None = None,
) -> np.ndarray:
    """Project a ``(H, W, D)`` embedding onto its top-3 principal components
    and map to uint8 RGB.

    The Ai2 OlmoEarth tutorial shows this pattern verbatim — PCA to 3
    dimensions gives "the same structure the encoder sees, rendered as
    colors". Similar embeddings → similar colors, so agricultural parcels,
    urban cores, and water bodies each pick up their own hue with zero
    labels. Works globally — no FT head region lock.

    Pixels where every ``D`` dimension is exactly zero are treated as
    ``nodata`` and emit ``(0, 0, 0)`` — matches the convention used by
    ``_run_chunked_embedding_export`` for unwritten chunks.

    Args:
        embedding_hwd: ``(H, W, D)`` float32 stitched embedding tensor.
        smooth_sigma: Gaussian sigma (in patch units) applied to the
            top-3 PC rasters before quantizing to uint8, to mask the
            encoder-chunk seams. ``None`` reads ``OE_PCA_SMOOTH_SIGMA``
            env var (default 1.5). Set ``0`` to disable and recover the
            old behaviour where chunk grid lines are visible.

    Returns a ``(H, W, 3)`` ``uint8`` array suitable for direct RGB tile
    rendering via the ``rgb_raster`` path in
    ``olmoearth_inference._render_pytorch_tile``.
    """
    h, w, d = embedding_hwd.shape
    flat = embedding_hwd.reshape(h * w, d)
    # Nodata mask — untouched chunks sit at exact zero across all D dims.
    nodata_flat = ~np.any(flat != 0, axis=-1)

    rgb = np.zeros((h * w, 3), dtype=np.uint8)
    valid = ~nodata_flat
    if not valid.any():
        return rgb.reshape(h, w, 3)

    flat_valid = flat[valid]
    centered = flat_valid - flat_valid.mean(axis=0, keepdims=True)
    # Top-3 components via SVD. ``full_matrices=False`` keeps the matrix
    # small (N × 3 instead of N × D) — cheap for typical N = H×W ≤ 16 k.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    k = min(3, vh.shape[0])
    pcs = centered @ vh[:k].T                                # (N_valid, k)
    if k < 3:
        # Tiny embedding dim — pad missing components with zeros.
        pad = np.zeros((pcs.shape[0], 3 - k), dtype=pcs.dtype)
        pcs = np.concatenate([pcs, pad], axis=-1)

    # Reshape PCs back into (H, W, 3) so we can smooth at chunk seams
    # in image space. Nodata pixels stay at exact-zero so the smoother
    # treats them as nodata-aware and doesn't bleed colour outside the
    # AOI footprint.
    pcs_2d = np.zeros((h, w, 3), dtype=np.float32)
    pcs_2d.reshape(h * w, 3)[valid] = pcs.astype(np.float32)
    nodata_2d = nodata_flat.reshape(h, w)
    sigma = smooth_sigma if smooth_sigma is not None else _smooth_sigma_default(
        "OE_PCA_SMOOTH_SIGMA", 1.5,
    )
    pcs_2d = _smooth_seams(pcs_2d, sigma=sigma, nodata_mask=nodata_2d)
    pcs_smoothed = pcs_2d.reshape(h * w, 3)[valid]

    # Per-component min/max rescale to [0, 255]. Each channel spans its own
    # range so the image uses the full dynamic range even when PC variance
    # decays quickly (common for low-D embeddings).
    lo = pcs_smoothed.min(axis=0, keepdims=True)
    hi = pcs_smoothed.max(axis=0, keepdims=True)
    denom = np.maximum(hi - lo, 1e-9)
    normed = (pcs_smoothed - lo) / denom                     # (N_valid, 3)
    rgb[valid] = (np.clip(normed, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb.reshape(h, w, 3)


def model_summary(repo_id: str) -> dict[str, Any]:
    """Lightweight introspection: params / device / dtype for the cached model.
    Returns ``{cached: False}`` if the model isn't loaded yet."""
    with _cache_lock:
        hit = _cache.get(repo_id)
    if hit is None:
        return {"cached": False, "repo_id": repo_id}
    model, device = hit
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "cached": True,
        "repo_id": repo_id,
        "device": str(device),
        "num_parameters": n_params,
        "class_name": type(model).__name__,
    }
