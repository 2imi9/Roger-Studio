"""TIPSv2 zero-shot point-classification benchmark on the
``allenai/olmoearth_projects_mangrove`` ground-truth points.

The right shape of the question (per Ziming, 2026-04-30):

  *Use the official dataset's (lon, lat, ref_cls) as ground truth +
  AOI. Fetch any recent RGB satellite imagery for that AOI. Run
  TIPSv2 zero-shot with a restricted prompt set. Compare predicted
  class to ref_cls.*

This bench is deliberately **NOT** an OlmoEarth FT-Mangrove inference run.
It tests the TIPSv2 zero-shot panel — the labeling tool surface in Roger
Studio — on the classes the user actually cares about (Mangrove / Water /
Other). Mangroves are temporally stable (a 2026 aerial image still shows
the same mangrove extent that was labelled in 2020), so a single recent
basemap tile is sufficient — no 12-month S2 composite needed.

What it does:
  1. Read ``mangrove_classification/input.csv`` (100k labelled points;
     hash-stable subsample of N).
  2. For each point, fetch a 2x2 grid of ESRI World Imagery tiles
     (free, no auth, ~1m/pixel at zoom 16) centered on the point. Stitch
     into a 512x512 RGB image.
  3. Call ``classify_image_zeroshot`` directly (Python, no HTTP) with a
     3-class prompt set. Take the center-patch class as the prediction.
  4. Compare to ``ref_cls`` and aggregate confusion matrix + per-class
     precision/recall/F1 + overall accuracy.

Reasoning on imagery source:
  - ESRI World Imagery is free, key-less, high-res, globally available.
  - Resolution ~1m at zoom 16 vs Sentinel-2's 10m — better for visually
    distinguishing mangroves from other coastal vegetation.
  - Imagery date is "current" (post-2020), but mangrove extent changes
    slowly — the dominant class at a labelled point is highly likely
    the same in 2020 and 2026.

Usage::

    python scripts/bench_tipsv2_on_mangrove.py --limit 30 --model l14 --output results.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger("bench_tipsv2_on_mangrove")

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

DATA_DIR = _BACKEND_ROOT / "data" / "olmoearth_projects" / "mangrove"
INPUT_CSV = DATA_DIR / "mangrove_classification" / "input.csv"

# Three-class prompt set tuned for visual aerial RGB at ~1m/pixel resolution.
# Wording chosen so each prompt names the class with descriptive cues that
# exist at this scale (mangrove root systems, water surfaces with no
# vegetation, generic land surfaces). Class names match the dataset's
# ``ref_cls`` values 1:1.
PROMPT_SET = [
    {
        "name": "Mangrove",
        "prompt": "dense mangrove forest along a tropical coastline with brackish water and aerial roots",
        "color": "#94eb63",
    },
    {
        "name": "Water",
        "prompt": "open water body, ocean or wide river surface with no vegetation visible above the water",
        "color": "#63d8eb",
    },
    {
        "name": "Other",
        "prompt": "other land cover such as inland forest, farmland, beach, urban area, or bare ground",
        "color": "#eba963",
    },
]
CLASS_NAMES = [c["name"] for c in PROMPT_SET]

# ESRI World Imagery: free, no auth, globally available, ~0.6-1m/pixel
# at z=16. Y/X order matches their REST API: tile/{z}/{y}/{x}. Other
# providers (Bing, Google) require keys; Sentinel-2 via STAC is slower
# (10s+ per tile vs <1s here) and has a coarser 10m/pixel that doesn't
# resolve mangrove root structure as crisply.
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def _wgs84_to_tile_xy(lon: float, lat: float, z: int) -> tuple[int, int, float, float]:
    """Slippy-map tile coordinates + within-tile pixel offset for a
    (lon, lat) at zoom level ``z``."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    tile_x = int(x)
    tile_y = int(y)
    px = (x - tile_x) * 256
    py = (y - tile_y) * 256
    return tile_x, tile_y, px, py


def _fetch_aoi_image(
    lon: float, lat: float, zoom: int = 16, tiles_per_side: int = 2,
    session: requests.Session | None = None,
) -> Image.Image | None:
    """Stitch a (tiles_per_side x tiles_per_side) ESRI tile grid centered
    on the (lon, lat) point at the given zoom level. Returns a PIL RGB
    image, or ``None`` if any tile fetch fails."""
    sess = session or requests.Session()
    tx0, ty0, _, _ = _wgs84_to_tile_xy(lon, lat, zoom)
    half = tiles_per_side // 2
    canvas = Image.new("RGB", (tiles_per_side * 256, tiles_per_side * 256))
    for j in range(tiles_per_side):
        for i in range(tiles_per_side):
            tx = tx0 - half + i
            ty = ty0 - half + j
            url = ESRI_TILE_URL.format(z=zoom, x=tx, y=ty)
            try:
                r = sess.get(url, timeout=20.0)
                if not r.ok:
                    return None
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                canvas.paste(tile, (i * 256, j * 256))
            except Exception:
                return None
    return canvas


def _hash_ring(uid: str, seed: int = 0) -> int:
    import hashlib
    return int.from_bytes(
        hashlib.blake2b(f"{seed}|{uid}".encode("utf-8"), digest_size=4).digest(),
        "big",
    )


def _load_samples(limit: int, seed: int) -> list[dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Expected {INPUT_CSV}. Download mangrove.tar from HuggingFace and extract."
        )
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: _hash_ring(r["uid"], seed=seed))
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", choices=("b14", "l14", "g14"), default="l14")
    ap.add_argument("--zoom", type=int, default=16,
                    help="ESRI tile zoom (16 ≈ 1 m/px, 14 ≈ 5 m/px)")
    ap.add_argument("--tiles", type=int, default=2,
                    help="tiles per side; 2 = 512x512 image, 4 = 1024x1024")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = _load_samples(args.limit, args.seed)
    logger.info("Loaded %d hash-stable samples (seed=%d)", len(samples), args.seed)
    class_dist = Counter(s["ref_cls"] for s in samples)
    logger.info("Class dist: %s", dict(class_dist))

    # Lazy-import the labeler so failures load fast.
    from app.services.tipsv2_labeler import _ensure_imports, _get_model, classify_image_zeroshot
    _ensure_imports()
    repo = f"google/tipsv2-{args.model}"
    logger.info("Loading TIPSv2 model: %s", repo)
    _get_model(repo)

    sess = requests.Session()
    sess.headers["User-Agent"] = "RogerStudio-bench/1 (research)"

    rows: list[dict[str, Any]] = []
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    n_skipped_imagery = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        lon = float(s["longitude"])
        lat = float(s["latitude"])
        ref_cls = s["ref_cls"]
        if ref_cls not in CLASS_NAMES:
            continue

        img = _fetch_aoi_image(lon, lat, zoom=args.zoom, tiles_per_side=args.tiles, session=sess)
        if img is None:
            n_skipped_imagery += 1
            rows.append({"uid": s["uid"], "lon": lon, "lat": lat, "ref": ref_cls, "pred": None, "reason": "imagery_fetch_failed"})
            if (i + 1) % args.log_every == 0:
                logger.info("[%d/%d] imagery fetch failed for %s", i + 1, len(samples), s["uid"])
            continue

        result = classify_image_zeroshot(img, classes=PROMPT_SET, model_name=repo)
        # Prediction = the global class from the cls-token. This matches
        # how the existing TIPSv2 panel reports a "scene class" badge,
        # and corresponds to the cls-token-with-prompt similarity that
        # carries the strongest signal at scene level. Per-patch label_map
        # is also captured for diagnostic.
        pred = result["global_class"]
        center = result["grid_size"] // 2
        center_pred_idx = result["label_map"][center][center]
        center_pred = CLASS_NAMES[center_pred_idx] if center_pred_idx < len(CLASS_NAMES) else None

        ref_idx = CLASS_NAMES.index(ref_cls)
        pred_idx = CLASS_NAMES.index(pred) if pred in CLASS_NAMES else None
        if pred_idx is not None:
            confusion[ref_idx, pred_idx] += 1
        rows.append({
            "uid": s["uid"], "lon": lon, "lat": lat,
            "ref": ref_cls,
            "pred_global": pred,
            "pred_center_patch": center_pred,
            "global_confidence": result.get("global_confidence"),
            "needs_review_pct": result.get("needs_review_pct"),
        })

        if (i + 1) % args.log_every == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            ips = (i + 1) / max(elapsed, 1e-6)
            n_eval = int(confusion.sum())
            acc = float(np.diag(confusion).sum() / max(n_eval, 1)) if n_eval else 0.0
            logger.info("[%d/%d] ips=%.2f acc(running)=%.3f ref=%s → pred=%s",
                        i + 1, len(samples), ips, acc, ref_cls, pred)

    elapsed = time.time() - t0
    n_eval = int(confusion.sum())
    diag = np.diag(confusion).astype(np.float64)
    overall_acc = float(diag.sum() / max(n_eval, 1)) if n_eval else 0.0

    # Per-class precision / recall / F1.
    eps = 1e-9
    per_class = []
    for ci, name in enumerate(CLASS_NAMES):
        tp = float(confusion[ci, ci])
        fp = float(confusion[:, ci].sum()) - tp
        fn = float(confusion[ci, :].sum()) - tp
        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)
        f1 = 2 * precision * recall / max(precision + recall, eps)
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int(confusion[ci, :].sum())})
    macro_f1 = float(np.mean([c["f1"] for c in per_class])) if per_class else 0.0

    print(f"\n=== TIPSv2-{args.model.upper()} on olmoearth_projects_mangrove (n_evaluated={n_eval}) ===")
    print(f"requested {len(samples)}; imagery_failed={n_skipped_imagery}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/max(len(samples),1):.2f}s/sample)")
    print(f"overall acc : {overall_acc:.4f}")
    print(f"macro F1    : {macro_f1:.4f}")
    print("per-class:")
    for c in per_class:
        print(f"  {c['class']:10s} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} support={c['support']}")
    print("\nconfusion (rows=ref, cols=pred):")
    print("            " + " ".join(f"{c:>10s}" for c in CLASS_NAMES))
    for r, row in enumerate(confusion):
        print(f"  {CLASS_NAMES[r]:10s} " + " ".join(f"{v:>10d}" for v in row))

    if args.output:
        out = {
            "model": repo,
            "dataset": "allenai/olmoearth_projects_mangrove",
            "imagery_source": "ESRI World Imagery (ArcGIS REST)",
            "zoom": args.zoom,
            "tiles_per_side": args.tiles,
            "n_requested": len(samples),
            "n_evaluated": n_eval,
            "n_imagery_failed": n_skipped_imagery,
            "elapsed_s": elapsed,
            "overall_accuracy": overall_acc,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion": confusion.tolist(),
            "classes": CLASS_NAMES,
            "prompts": [{"name": c["name"], "prompt": c["prompt"]} for c in PROMPT_SET],
            "samples": rows,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
