"""OlmoEarth compare-demo registry.

Curates a handful of A/B inference pairs that showcase the FT heads we ship
on disk (Mangrove, AWF, EcosystemTypeMapping) — so users opening the
SplitMap compare view get OlmoEarth-sourced demos instead of generic NASA
GIBS true-color tiles. Mirrors the OlmoEarth Studio viewer's
"Prediction Nigeria Mangrove 2018 vs 2024" affordance.

Each demo has two specs (A + B) that feed ``olmoearth_inference.start_inference``.
``_make_job_id`` produces a deterministic hash from the spec, so we can
pre-compute the tile URL at registry-build time without kicking inference
yet. The UI then decides whether to warm-up inference eagerly or lazily
(see ``/api/olmoearth/demo-pairs`` + ``/api/olmoearth/demo-pairs/prebake``).

Bbox size is kept at ≤ 2.56 km × 2.56 km (single 256-px OlmoEarth tile at
10 m/px). That's the pretraining-native window size — inference is fast
(~30 s cold, ≈ instant cached) and the tile matches the model's training
distribution.

Colormap + legend for each demo come from ``olmoearth_inference._COLORMAPS``
and ``_COLORMAP_LEGEND`` so the compare divider shows semantically
meaningful colors (mangrove → cyan, landuse → green/gold, etc.) that
match OlmoEarth Studio's visual language.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# These three FT heads are cached on the typical Roger Studio install:
#   models--allenai--OlmoEarth-v1-FT-Mangrove-Base
#   models--allenai--OlmoEarth-v1-FT-AWF-Base
#   models--allenai--OlmoEarth-v1-FT-EcosystemTypeMapping-Base
# Each has a single ``model.ckpt`` with PyTorch Lightning state + head
# config inferred from tensor shapes at load time (see olmoearth_ft.py).
_MANGROVE = "allenai/OlmoEarth-v1-FT-Mangrove-Base"
_AWF = "allenai/OlmoEarth-v1-FT-AWF-Base"
_ECOSYSTEM = "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base"


def _make_job_id(spec: dict[str, Any]) -> str:
    """Must match ``olmoearth_inference._make_job_id`` byte-for-byte.

    Duplicated here (rather than imported) to keep the demo registry as
    a pure-data module without triggering the heavy inference-service
    import chain (torch, rasterio, PC STAC client, …) just to build a
    URL. The contract is: same spec dict → same 16-hex-char job_id, in
    both places.
    """
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class DemoSpec:
    """One side of a compare demo (A or B)."""

    label: str
    model_repo_id: str
    west: float
    south: float
    east: float
    north: float
    date_range: str  # "YYYY-MM-DD/YYYY-MM-DD"

    def to_inference_spec(self) -> dict[str, Any]:
        """Mirror the spec shape used by ``start_inference`` so job_ids match."""
        return {
            "bbox": {
                "west": self.west,
                "south": self.south,
                "east": self.east,
                "north": self.north,
            },
            "model_repo_id": self.model_repo_id,
            "date_range": self.date_range,
            "max_size_px": 256,
            "sliding_window": False,
            "window_size": None,
        }

    @property
    def job_id(self) -> str:
        return _make_job_id(self.to_inference_spec())

    @property
    def tile_url(self) -> str:
        """Path served by the existing ``/olmoearth/infer-tile/{job_id}``
        route. Includes the ``/api`` prefix so the Vite dev-server proxy
        picks it up unchanged (same pattern used elsewhere in the app)."""
        return f"/api/olmoearth/infer-tile/{self.job_id}/{{z}}/{{x}}/{{y}}.png"


@dataclass(frozen=True)
class DemoPair:
    """A compare preset — two DemoSpecs (A and B) plus UI metadata."""

    id: str
    title: str
    blurb: str
    fit_bbox: tuple[float, float, float, float]  # west, south, east, north
    a: DemoSpec
    b: DemoSpec


# Sizing rule: inference bbox ≈ 0.10° × 0.10° at the equator ≈ 11 km × 11 km,
# which downsampled to 256 × 256 px gives ~43 m/pixel. That reads sharply
# at web-map zoom 10–13 and still shows *some* content at zoom 14. Keep
# ``fit_bbox`` AT MOST slightly larger than the inference bbox so the
# camera zooms to a view where the raster fills the map — otherwise the
# tiles show up as a postage-stamp surrounded by basemap (the user
# reported this for the original 0.20° fit-bbox-vs-0.05° inference-bbox
# mismatch on AWF).
#
# For compare pairs: A and B MUST share the same spatial extent (same
# bbox on both sides) — otherwise the maplibre-gl-compare divider spans
# two different geographies and the user can't visually align them. Cross-
# region comparisons are better handled as two separate inference runs the
# user launches themselves via OlmoEarthPanel, not via compare presets.
DEMO_PAIRS: list[DemoPair] = [
    DemoPair(
        id="mangrove-niger-delta-2020-vs-2024",
        title="Mangrove FT — Niger Delta, 2020 vs 2024",
        blurb=(
            "OlmoEarth Mangrove head run on Sentinel-2 L2A composites at the "
            "Niger Delta mouth (~11 km AOI). Cyan = mangrove probability. "
            "Drag the divider to spot mangrove loss / regrowth across 4 "
            "years — the same demo OlmoEarth Studio ships on its viewer "
            "homepage."
        ),
        fit_bbox=(6.30, 4.35, 6.50, 4.55),
        a=DemoSpec(
            label="Mangrove · Niger Delta · 2020",
            model_repo_id=_MANGROVE,
            west=6.30, south=4.35, east=6.50, north=4.55,
            date_range="2020-04-01/2020-10-01",
        ),
        b=DemoSpec(
            label="Mangrove · Niger Delta · 2024",
            model_repo_id=_MANGROVE,
            west=6.30, south=4.35, east=6.50, north=4.55,
            date_range="2024-04-01/2024-10-01",
        ),
    ),
    DemoPair(
        id="awf-tsavo-2020-vs-2024",
        title="AWF land-use FT — Tsavo/Amboseli, 2020 vs 2024",
        blurb=(
            "African Wildlife Foundation head on southern-Kenya savanna "
            "(~22 km AOI). Gold/green/cyan classes split cropland, rangeland, "
            "and water. Useful for spotting conversion between wildlife "
            "corridors and agriculture across a 4-year span."
        ),
        fit_bbox=(37.30, -2.75, 37.50, -2.55),
        a=DemoSpec(
            label="AWF · Tsavo · 2020",
            model_repo_id=_AWF,
            west=37.30, south=-2.75, east=37.50, north=-2.55,
            date_range="2020-04-01/2020-10-01",
        ),
        b=DemoSpec(
            label="AWF · Tsavo · 2024",
            model_repo_id=_AWF,
            west=37.30, south=-2.75, east=37.50, north=-2.55,
            date_range="2024-04-01/2024-10-01",
        ),
    ),
    DemoPair(
        id="ecosystem-california-2020-vs-2024",
        title="Ecosystem Type — Central California, 2020 vs 2024",
        blurb=(
            "EcosystemTypeMapping head on the central-California coast "
            "(~22 km AOI covering Monterey Bay / Salinas Valley). Same "
            "location, 4-year span — drag the divider to see agricultural "
            "rotation, wildfire recovery, and coastal dynamics. Replaces "
            "the earlier cross-continent compare that didn't align "
            "spatially across the divider."
        ),
        fit_bbox=(-122.0, 36.5, -121.8, 36.7),
        a=DemoSpec(
            label="Ecosystem · Central CA · 2020",
            model_repo_id=_ECOSYSTEM,
            west=-122.0, south=36.5, east=-121.8, north=36.7,
            date_range="2020-04-01/2020-10-01",
        ),
        b=DemoSpec(
            label="Ecosystem · Central CA · 2024",
            model_repo_id=_ECOSYSTEM,
            west=-122.0, south=36.5, east=-121.8, north=36.7,
            date_range="2024-04-01/2024-10-01",
        ),
    ),
]


def _static_legend_for(model_repo_id: str) -> dict[str, Any] | None:
    """Return the static colormap + legend hint for a demo-side model repo.

    Demo pairs stream tile URLs BEFORE inference actually runs, so we can't
    return the full per-class legend that ``start_inference`` assembles (it
    depends on the FT head's runtime class metadata). But the colormap key
    and its human label / note ARE known up-front — each FT repo is wired
    to a specific key in ``olmoearth_inference._COLORMAPS``. Surfacing the
    gradient stops + label + honesty note now lets SplitMap paint a
    compact "what are these colors?" legend immediately when the demo
    loads, instead of waiting 30 s for the post-inference legend. This is
    the upgrade users asked for: raster colors stop being mystery blobs.

    Imports ``olmoearth_inference`` lazily to keep the demo registry a
    pure-data module (see ``_make_job_id`` doc-string for the rationale).
    """
    from app.services import olmoearth_inference  # noqa: PLC0415

    colormap_key = olmoearth_inference._COLORMAPS.get(model_repo_id)
    if not colormap_key:
        return None
    entry = olmoearth_inference._COLORMAP_LEGEND.get(colormap_key)
    if not entry:
        return None
    return {
        "colormap": colormap_key,
        "label": entry.get("label"),
        "note": entry.get("note"),
        # Flatten tuple stops to JSON-safe lists. Matches the shape
        # ``_COLORMAP_LEGEND`` already uses on the live inference path so
        # the frontend can share a single renderer across demo + real jobs.
        "stops": [[color, pos] for color, pos in (entry.get("stops") or [])],
        # Semantic anchors so the UI doesn't show meaningless "low / high"
        # under the gradient. E.g. for mangrove → "non-mangrove / mangrove";
        # for landuse → "cropland / water". Users otherwise had to read the
        # note tooltip to know what the colors mean.
        "low_label": entry.get("low_label"),
        "high_label": entry.get("high_label"),
    }


def describe_pairs() -> list[dict[str, Any]]:
    """JSON-serializable description consumed by ``/api/olmoearth/demo-pairs``.

    Shape matches what the frontend SplitMap expects: id / title / blurb /
    fit_bbox + per-side ``{id, label, tile_url}``. The id of each side is
    the job_id — stable + deterministic — so the frontend can treat it
    like any other imagery layer id (prefix ``demo-`` to distinguish from
    user-loaded rasters, matching the existing ``example-`` convention).

    Each side also carries ``legend_hint`` — the static colormap label /
    note / gradient stops for that FT head — so the split view can draw a
    legend the moment the demo loads, not after inference finishes.
    """
    out: list[dict[str, Any]] = []
    for pair in DEMO_PAIRS:
        a_legend = _static_legend_for(pair.a.model_repo_id)
        b_legend = _static_legend_for(pair.b.model_repo_id)
        out.append(
            {
                "id": pair.id,
                "title": pair.title,
                "blurb": pair.blurb,
                "fit_bbox": {
                    "west": pair.fit_bbox[0],
                    "south": pair.fit_bbox[1],
                    "east": pair.fit_bbox[2],
                    "north": pair.fit_bbox[3],
                },
                "a": {
                    "id": f"demo-{pair.a.job_id}",
                    "label": pair.a.label,
                    "tile_url": pair.a.tile_url,
                    "job_id": pair.a.job_id,
                    "spec": pair.a.to_inference_spec(),
                    "legend_hint": a_legend,
                },
                "b": {
                    "id": f"demo-{pair.b.job_id}",
                    "label": pair.b.label,
                    "tile_url": pair.b.tile_url,
                    "job_id": pair.b.job_id,
                    "spec": pair.b.to_inference_spec(),
                    "legend_hint": b_legend,
                },
            }
        )
    return out
