"""Ai2 OlmoEarth datasets + models catalog for the Roger Studio backend.

What Ai2 publishes on HuggingFace (verified Nov 2025):

Datasets (``allenai/*``):
  - ``olmoearth_pretrain_dataset`` — Sentinel-2 L2A + Sentinel-1 GRD image
    chips, webdataset format, 100M-1B samples, CC-BY-4.0.
  - ``olmoearth_projects_mangrove`` — Global Mangrove Watch v4 reference
    samples, CC-BY-4.0.
  - ``olmoearth_projects_awf`` — Southern Kenya land use / land cover,
    annotated by African Wildlife Foundation experts, Apache-2.0.

Models (``allenai/OlmoEarth-v1-*``):
  - Encoders: ``Nano``, ``Tiny``, ``Base``, ``Large``.
  - Fine-tuned heads on v1-Base: ``FT-Mangrove``, ``FT-AWF``, ``FT-LFMC``,
    ``FT-ForestLossDriver``, ``FT-EcosystemTypeMapping``.

What is NOT published — do not pretend otherwise:
  - Pre-computed global embedding tiles indexed by bbox.
  - Pre-computed global land-cover classifications.
  - Multi-temporal NDVI / NDWI timeseries (those are Sentinel-2 indices,
    computed elsewhere — wire them in ``geo_tools.py`` via STAC, not here).

So this adapter exposes:
  1. The authoritative catalog (hardcoded constants).
  2. Live HF metadata (downloads, last-modified) via the public REST API.
  3. ``project_coverage(bbox)`` — does this bbox sit inside a known
     project-labelled region?
  4. ``recommend_model(task)`` — pick a repo ID for a task.
  5. ``catalog_summary(bbox)`` — the one-call snapshot that geo_tools.py
     wraps into the ``query_olmoearth`` LLM tool.

Running inference (applying an OlmoEarth encoder to user imagery to produce
embeddings or class probabilities for an arbitrary bbox) is out of scope for
this module — it needs PyTorch + the weights downloaded — and will live in a
future ``olmoearth_inference.py``.
"""
from __future__ import annotations

import logging
import re
import asyncio
import time
from typing import Any

import httpx

from app.models.schemas import BBox

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"


DATASETS: dict[str, dict[str, Any]] = {
    "allenai/olmoearth_pretrain_dataset": {
        "description": (
            "OlmoEarth pre-training corpus. Sentinel-2 L2A + Sentinel-1 GRD "
            "image chips, webdataset format."
        ),
        "license": "cc-by-4.0",
        "size": "100M-1B samples",
        "modalities": ["sentinel-2-l2a", "sentinel-1-grd"],
        "format": "webdataset",
        "docs": "https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Pretraining-Dataset.md",
    },
    "allenai/olmoearth_projects_mangrove": {
        "description": (
            "Global Mangrove Watch v4 reference samples, used to fine-tune "
            "OlmoEarth-v1-Base for mangrove extent mapping."
        ),
        "license": "cc-by-4.0",
        "task": "mangrove extent binary segmentation",
        "coverage": "global tropical / subtropical coasts",
        "docs": "https://github.com/allenai/olmoearth_projects/blob/main/docs/mangrove.md",
    },
    "allenai/olmoearth_projects_awf": {
        "description": (
            "Expert-annotated land use / land cover in southern Kenya "
            "(African Wildlife Foundation)."
        ),
        "license": "apache-2.0",
        "size": "10K-100K samples",
        "task": "land use / land cover classification",
        "coverage": "southern Kenya",
        "docs": "https://github.com/allenai/olmoearth_projects/blob/main/docs/awf.md",
    },
}

# Size tier is the one deterministic ranking we can give without fabricating
# parameter counts (Ai2 hasn't published exact sizes). Callers that need a
# numeric VRAM budget should read the HF model card directly.
_ENCODER_ORDER = [
    "allenai/OlmoEarth-v1-Nano",
    "allenai/OlmoEarth-v1-Tiny",
    "allenai/OlmoEarth-v1-Base",
    "allenai/OlmoEarth-v1-Large",
]

MODELS: dict[str, dict[str, Any]] = {
    "allenai/OlmoEarth-v1-Nano":  {"size_tier": "smallest", "type": "encoder"},
    "allenai/OlmoEarth-v1-Tiny":  {"size_tier": "small",    "type": "encoder"},
    "allenai/OlmoEarth-v1-Base":  {"size_tier": "medium",   "type": "encoder"},
    "allenai/OlmoEarth-v1-Large": {"size_tier": "largest",  "type": "encoder"},
    "allenai/OlmoEarth-v1-FT-Mangrove-Base":             {"base": "v1-Base", "task": "mangrove extent mapping"},
    "allenai/OlmoEarth-v1-FT-AWF-Base":                  {"base": "v1-Base", "task": "southern Kenya land use / land cover"},
    "allenai/OlmoEarth-v1-FT-LFMC-Base":                 {"base": "v1-Base", "task": "live fuel moisture content"},
    "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base":     {"base": "v1-Base", "task": "forest loss driver classification"},
    "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": {"base": "v1-Base", "task": "ecosystem type mapping"},
}

# Approximate coverage for the project datasets. Mangrove is global-coastal,
# approximated here by the tropical belt — a bbox inside this band MAY have
# labelled samples, but only near coastlines (overlay with a coastline check
# for a real answer). AWF is a tight bbox over southern Kenya.
PROJECT_REGIONS: dict[str, BBox] = {
    "allenai/olmoearth_projects_mangrove": BBox(west=-180.0, south=-30.0, east=180.0, north=30.0),
    "allenai/olmoearth_projects_awf":      BBox(west=33.5,   south=-5.0,  east=42.0,  north=-1.0),
}


def list_datasets() -> list[dict[str, Any]]:
    """Return the OlmoEarth dataset catalog as a list of dicts with ``repo_id``."""
    return [{"repo_id": repo_id, **meta} for repo_id, meta in DATASETS.items()]


def list_models() -> list[dict[str, Any]]:
    """Return the OlmoEarth model catalog as a list of dicts with ``repo_id``."""
    return [{"repo_id": repo_id, **meta} for repo_id, meta in MODELS.items()]


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a.east < b.west or a.west > b.east or a.north < b.south or a.south > b.north)


def project_coverage(bbox: BBox) -> list[dict[str, Any]]:
    """List OlmoEarth project datasets whose coverage region overlaps ``bbox``."""
    hits: list[dict[str, Any]] = []
    for repo_id, region in PROJECT_REGIONS.items():
        if _bbox_intersects(bbox, region):
            hits.append({
                "repo_id": repo_id,
                "region": region.model_dump(),
                "dataset": DATASETS.get(repo_id, {}),
            })
    return hits


# Keyword → fine-tuned repo. First match wins.
_TASK_KEYWORDS: list[tuple[str, str]] = [
    ("mangrove",       "allenai/OlmoEarth-v1-FT-Mangrove-Base"),
    ("kenya",          "allenai/OlmoEarth-v1-FT-AWF-Base"),
    ("awf",            "allenai/OlmoEarth-v1-FT-AWF-Base"),
    ("land use",       "allenai/OlmoEarth-v1-FT-AWF-Base"),
    ("land cover",     "allenai/OlmoEarth-v1-FT-AWF-Base"),
    ("fuel moisture",  "allenai/OlmoEarth-v1-FT-LFMC-Base"),
    ("lfmc",           "allenai/OlmoEarth-v1-FT-LFMC-Base"),
    ("wildfire",       "allenai/OlmoEarth-v1-FT-LFMC-Base"),
    ("forest loss",    "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base"),
    ("deforestation",  "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base"),
    ("ecosystem",      "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base"),
]


def recommend_model(
    task: str | None = None,
    size_preference: str = "medium",
) -> dict[str, Any]:
    """Recommend an OlmoEarth model for a task.

    ``task`` is matched against known fine-tuning heads first. If no match,
    falls back to a pretrained encoder at the requested ``size_preference``
    (``smallest`` / ``small`` / ``medium`` / ``largest``, default ``medium``).
    """
    task_l = (task or "").lower().strip()
    for keyword, repo_id in _TASK_KEYWORDS:
        if keyword in task_l:
            return {
                "repo_id": repo_id,
                "reason": f"fine-tuned head for task keyword '{keyword}'",
                **MODELS[repo_id],
            }

    tier_to_repo = {MODELS[r]["size_tier"]: r for r in _ENCODER_ORDER}
    repo_id = tier_to_repo.get(size_preference, "allenai/OlmoEarth-v1-Base")
    return {
        "repo_id": repo_id,
        "reason": f"no fine-tuned head matched; fell back to {size_preference} encoder",
        **MODELS[repo_id],
    }


async def get_dataset_info(repo_id: str, hf_token: str | None = None) -> dict[str, Any]:
    """Fetch live HF dataset metadata (downloads, lastModified, tags, siblings)."""
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            r = await client.get(f"{HF_API}/datasets/{repo_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        logger.warning("HF dataset info fetch failed for %s: %s", repo_id, e)
        return {"repo_id": repo_id, "error": str(e)}


async def get_model_info(repo_id: str, hf_token: str | None = None) -> dict[str, Any]:
    """Fetch live HF model metadata (downloads, tags, siblings — the file list)."""
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            r = await client.get(f"{HF_API}/models/{repo_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        logger.warning("HF model info fetch failed for %s: %s", repo_id, e)
        return {"repo_id": repo_id, "error": str(e)}


async def sample_rows(
    repo_id: str = "allenai/olmoearth_pretrain_dataset",
    config: str = "default",
    split: str = "train",
    limit: int = 5,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Fetch the first rows of a dataset via HF Datasets Server.

    Use this to prove reachability and show a tiny preview — NOT for spatial
    retrieval. The Datasets Server indexes webdataset rows by position, not
    by geography.
    """
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    params = {"dataset": repo_id, "config": config, "split": split, "length": limit}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            r = await client.get(f"{HF_DATASETS_SERVER}/first-rows", params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        logger.warning("HF datasets-server fetch failed for %s: %s", repo_id, e)
        return {"repo_id": repo_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Live HF catalog fetch with TTL cache
#
# Ai2 ships new fine-tuned OlmoEarth heads every few weeks (FT-Mangrove,
# FT-AWF, FT-LFMC, FT-ForestLossDriver, FT-EcosystemTypeMapping so far). The
# hardcoded MODELS dict above is a fallback; this block queries the HF Hub
# listing API on every catalog_summary() call, coalesces repeats within a
# 10-minute window via a simple in-memory TTL cache, and merges the live data
# with our static task/size metadata so the LLM + UI always see new heads
# without a redeploy.
# ---------------------------------------------------------------------------

_LIVE_CACHE_TTL_SEC = 600
_live_cache: dict[str, Any] = {"ts": 0.0, "models": None, "datasets": None}

# Parsed from the repo name. e.g. "OlmoEarth-v1-FT-Mangrove-Base" → "Mangrove".
# Any FT keyword we don't have a human-friendly task string for is kept as-is
# and lowercased so `recommend_model` can still match on it (e.g. a future
# "FT-CropYield-Base" becomes task keyword "cropyield" → partial substring
# matches against user tasks like "crop yield" still line up).
_FT_TASK_HUMAN: dict[str, str] = {
    "mangrove": "mangrove extent mapping",
    "awf": "southern Kenya land use / land cover",
    "lfmc": "live fuel moisture content",
    "forestlossdriver": "forest loss driver classification",
    "ecosystemtypemapping": "ecosystem type mapping",
}

_ENCODER_TIER_FROM_NAME = {
    "nano": "smallest",
    "tiny": "small",
    "base": "medium",
    "large": "largest",
}


def _parse_olmoearth_model(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a raw HF model-listing entry into our catalog schema.

    Returns ``None`` for repos that aren't OlmoEarth-v1-* (the HF search is a
    substring match, so it's worth re-filtering here).
    """
    repo_id = raw.get("id") or ""
    name = repo_id.split("/")[-1]
    if not name.startswith("OlmoEarth-v1-"):
        return None
    suffix = name[len("OlmoEarth-v1-"):]  # e.g. "Base", "FT-Mangrove-Base"
    common = {
        "repo_id": repo_id,
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "last_modified": raw.get("lastModified"),
    }
    m = re.fullmatch(r"FT-(?P<task>[A-Za-z0-9]+)-(?P<base>Base|Nano|Tiny|Large)", suffix)
    if m:
        task_key = m.group("task").lower()
        return {
            **common,
            "type": "fine-tuned",
            "base": f"v1-{m.group('base')}",
            "task_key": task_key,
            "task": _FT_TASK_HUMAN.get(task_key, task_key),
        }
    if suffix.lower() in _ENCODER_TIER_FROM_NAME:
        return {**common, "type": "encoder", "size_tier": _ENCODER_TIER_FROM_NAME[suffix.lower()]}
    # Unknown OlmoEarth-v1-* variant — surface it anyway so new shapes appear
    # in the catalog even before we learn how to classify them.
    return {**common, "type": "unknown", "raw_suffix": suffix}


def _parse_olmoearth_dataset(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a raw HF dataset-listing entry into our catalog schema."""
    repo_id = raw.get("id") or ""
    name = repo_id.split("/")[-1].lower()
    if not (name.startswith("olmoearth") or "olmoearth" in name):
        return None
    static = DATASETS.get(repo_id, {})
    return {
        "repo_id": repo_id,
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "last_modified": raw.get("lastModified"),
        **static,  # description / license / task / coverage / docs when we know them
    }


async def _fetch_hf_list(
    kind: str,
    author: str = "allenai",
    search: str = "OlmoEarth",
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """GET https://huggingface.co/api/<kind>?author=...&search=... — raw JSON."""
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers=headers,
            transport=httpx.AsyncHTTPTransport(retries=2),
        ) as client:
            r = await client.get(
                f"{HF_API}/{kind}",
                params={"author": author, "search": search, "limit": 200},
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        logger.warning("HF %s list fetch failed: %s", kind, e)
        return []


async def refresh_live_catalog(
    hf_token: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch + parse the OlmoEarth model + dataset lists from HF.

    Returns the cached copy if it's less than ``_LIVE_CACHE_TTL_SEC`` old.
    Falls back to the hardcoded catalog if the network hop fails.
    """
    now = time.time()
    if not force and _live_cache["models"] and now - _live_cache["ts"] < _LIVE_CACHE_TTL_SEC:
        return {"models": _live_cache["models"], "datasets": _live_cache["datasets"]}

    # Fire the two HF list calls in parallel. They're independent GETs —
    # previously serialized, so cold-cache latency was models(~10s, 2 retries
    # up to ~30s) + datasets(same) = up to ~60s during an HF hiccup. Every
    # backend restart clears ``_live_cache``, which the user sees as
    # "after refresh everything is slow": the first caller to /olmoearth/catalog
    # (or /analyze, since it transitively hits this) paid the full serial cost.
    # asyncio.gather halves the worst case; combined with a 60 s client
    # timeout, cold-cache calls stop surfacing as 30 s UI timeouts.
    raw_models, raw_datasets = await asyncio.gather(
        _fetch_hf_list("models", search="OlmoEarth", hf_token=hf_token),
        _fetch_hf_list("datasets", search="olmoearth", hf_token=hf_token),
    )

    parsed_models = [p for m in raw_models if (p := _parse_olmoearth_model(m))]
    parsed_datasets = [p for d in raw_datasets if (p := _parse_olmoearth_dataset(d))]

    # If the HF hop returned nothing (network / transient), fall back to the
    # hardcoded catalog so the rest of the app keeps working.
    if not parsed_models:
        parsed_models = list_models()
    if not parsed_datasets:
        parsed_datasets = list_datasets()

    _live_cache["ts"] = now
    _live_cache["models"] = parsed_models
    _live_cache["datasets"] = parsed_datasets
    return {"models": parsed_models, "datasets": parsed_datasets}


def _recommend_from_live(
    models: list[dict[str, Any]],
    task: str | None,
    size_preference: str,
) -> dict[str, Any]:
    """Pick a live-catalog model for ``task`` (or best-fit encoder fallback)."""
    task_l = (task or "").lower().strip()

    if task_l:
        # Prefer fine-tuned heads whose task_key appears in the user's task
        # string. Works for every OlmoEarth-v1-FT-* repo automatically, so new
        # FT heads Ai2 ships light up here without any code change.
        for m in models:
            if m.get("type") != "fine-tuned":
                continue
            kw = (m.get("task_key") or "").lower()
            human = (m.get("task") or "").lower()
            if kw and (kw in task_l or task_l in kw or task_l in human):
                return {**m, "reason": f"fine-tuned head matched task keyword '{kw}'"}
        # Also allow legacy keyword aliases from the hardcoded list (e.g.
        # "kenya" → FT-AWF) that aren't encoded in the repo name itself.
        for keyword, repo_id in _TASK_KEYWORDS:
            if keyword in task_l:
                found = next((m for m in models if m["repo_id"] == repo_id), None)
                if found:
                    return {**found, "reason": f"fine-tuned head for alias '{keyword}'"}

    encoders = [m for m in models if m.get("type") == "encoder"]
    match = next((m for m in encoders if m.get("size_tier") == size_preference), None)
    if match:
        return {**match, "reason": f"no fine-tuned head matched; fell back to {size_preference} encoder"}
    if encoders:
        return {**encoders[0], "reason": "no size match; returned first available encoder"}
    # Last-ditch fallback to the hardcoded recommendation
    return recommend_model(task=task, size_preference=size_preference)


async def catalog_summary(
    bbox: BBox | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """One-call snapshot of the OlmoEarth catalog, scoped to a bbox if given.

    Pulls the live model + dataset listings from HuggingFace (10-min TTL
    cache) so new fine-tuned heads appear without code changes. The hardcoded
    catalog at the top of this module is the fallback when HF is unreachable.
    """
    live = await refresh_live_catalog(hf_token=hf_token)
    models = live["models"]
    datasets = live["datasets"]

    coverage = project_coverage(bbox) if bbox is not None else []
    recommended_task = coverage[0]["dataset"].get("task") if coverage else None
    recommended = _recommend_from_live(
        models, task=recommended_task, size_preference="medium",
    )

    return {
        "datasets": datasets,
        "models": models,
        "project_coverage": coverage,
        "recommended_model": recommended,
        "notes": [
            "Ai2 does NOT publish pre-computed global embeddings or land-cover "
            "tiles. For embeddings / classifications on an arbitrary bbox, run "
            "an OlmoEarth-v1-* encoder on Sentinel-2/1 imagery locally — that "
            "path is not yet wired into this backend.",
            f"Catalog pulled live from huggingface.co/allenai (cache TTL {_LIVE_CACHE_TTL_SEC}s).",
        ],
    }
