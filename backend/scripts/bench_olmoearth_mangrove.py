"""Point-classification benchmark for OlmoEarth FT-Mangrove on the
``allenai/olmoearth_projects_mangrove`` reference samples.

Approach (light, end-to-end through the running backend):
  1. Read ``mangrove_classification/input.csv`` from the dataset tar
     (downloaded once into ``backend/data/olmoearth_projects/mangrove/``).
  2. Take a deterministic random subset of rows. The dataset documents a
     hash-based train/val split (87.5 / 12.5 % by 2x2 pixel grid hash); we
     mirror the *spirit* of that with a Python hash on ``uid`` so the same
     subset reproduces across runs without needing the rslearn-internal
     hash function. For "real" reporting we'd pin to the rslearn split, but
     for sanity-bench numbers a hashed subsample is accurate to within
     standard sampling noise on a balanced dataset.
  3. For each point, hit the backend's ``/api/olmoearth/infer`` endpoint
     with a small bbox around the point + the year-2020 date range FT-
     Mangrove was trained on. The endpoint runs the full FT pipeline
     (S2 fetch from STAC, 12-period composite, FT forward).
  4. Fetch a tile that covers the point, decode the predicted color at
     the labelled pixel, map it back to a class via the response legend.
  5. Compare against ``ref_cls``. Aggregate confusion matrix → per-class
     IoU + overall accuracy.

This is a POINT classification bench (single label vs single predicted
class at the labelled pixel), not full per-pixel mIoU on the inference
window. The reference dataset is points-only, not segmentation masks, so
that's the cleanest definition.

Usage::

    python scripts/bench_olmoearth_mangrove.py --limit 30 --output results.json

Args:
    --limit N           Cap evaluated samples. Default 30 (small smoke).
    --bbox-half DEG     Half-side of the inference bbox in degrees.
                        Default 0.013 (~1.4 km tile, 256 px at 10 m S2).
    --backend URL       Backend root. Default http://127.0.0.1:8000.
    --output PATH       Write per-class JSON results to this path.
    --seed N            RNG seed for the deterministic subsample. Default 0.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger("bench_olmoearth_mangrove")

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _BACKEND_ROOT / "data" / "olmoearth_projects" / "mangrove"
INPUT_CSV = DATA_DIR / "mangrove_classification" / "input.csv"

MODEL_REPO_ID = "allenai/OlmoEarth-v1-FT-Mangrove-Base"
DATE_RANGE = "2020-01-01/2020-12-31"

REF_TO_MODEL_CLASS = {
    "Mangrove": "mangrove",
    "Water": "water",
    "Other": "other",
}

CLASSES_FOR_IOU = ["mangrove", "water", "other"]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _wgs84_to_tile_xy(lon: float, lat: float, z: int) -> tuple[int, int, float, float]:
    """Return tile (x, y) and within-tile pixel offset (px, py) for a (lon, lat) at zoom z."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    tile_x = int(x)
    tile_y = int(y)
    px = (x - tile_x) * 256
    py = (y - tile_y) * 256
    return tile_x, tile_y, px, py


def _legend_color_to_class(
    legend: dict | None,
) -> dict[tuple[int, int, int], str]:
    """Build {(r,g,b) → class_name} from the response legend so we can decode tile pixels."""
    if not legend:
        return {}
    mapping: dict[tuple[int, int, int], str] = {}
    for stop in (legend.get("stops") or []):
        color_hex = stop.get("color")
        label = (stop.get("label") or "").lower().strip()
        if not color_hex or not label:
            continue
        try:
            mapping[_hex_to_rgb(color_hex)] = label
        except (ValueError, TypeError):
            continue
    return mapping


def _nearest_legend_class(
    pixel_rgb: tuple[int, int, int],
    color_to_class: dict[tuple[int, int, int], str],
) -> str | None:
    """Map an arbitrary RGB to the closest legend color's class. Returns None
    if the legend is empty."""
    if not color_to_class:
        return None
    pr, pg, pb = pixel_rgb
    best_dist = float("inf")
    best_cls: str | None = None
    for (r, g, b), cls in color_to_class.items():
        d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d < best_dist:
            best_dist = d
            best_cls = cls
    return best_cls


def _hash_ring(uid: str, seed: int = 0) -> int:
    """Stable 32-bit hash of ``uid`` salted by ``seed``. Avoids Python's salted
    ``hash()`` so subsamples reproduce across runs."""
    import hashlib
    h = hashlib.blake2b(f"{seed}|{uid}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big")


def _load_samples(limit: int, seed: int) -> list[dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Expected {INPUT_CSV}. Download with: "
            "curl -fL -o backend/data/olmoearth_projects/mangrove/mangrove.tar "
            "https://huggingface.co/datasets/allenai/olmoearth_projects_mangrove/resolve/main/mangrove.tar "
            "&& tar -xf <path>/mangrove.tar -C <path>/"
        )
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Hash-based stable subsampling: select the rows whose uid hash falls in
    # the lowest ``limit / N`` fraction. Equivalent to a deterministic random
    # 12.5 %-style val split when limit ≈ N / 8.
    n = len(rows)
    rows.sort(key=lambda r: _hash_ring(r["uid"], seed=seed))
    return rows[:limit]


def _request_inference(
    backend: str, lon: float, lat: float, bbox_half: float, timeout: float = 600.0,
) -> dict[str, Any]:
    body = {
        "bbox": {
            "west": lon - bbox_half,
            "south": lat - bbox_half,
            "east": lon + bbox_half,
            "north": lat + bbox_half,
        },
        "model_repo_id": MODEL_REPO_ID,
        "date_range": DATE_RANGE,
        "max_size_px": 256,
        "sliding_window": False,
    }
    r = requests.post(f"{backend}/api/olmoearth/infer", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _fetch_tile_pixel(
    backend: str, tile_url_template: str,
    lon: float, lat: float, zoom: int = 14,
) -> tuple[int, int, int] | None:
    """Fetch the tile that covers (lon, lat) at zoom z, return the RGB at the
    point's pixel position. ``None`` if the tile fetch fails."""
    tx, ty, px, py = _wgs84_to_tile_xy(lon, lat, zoom)
    url = tile_url_template.replace("{z}", str(zoom)).replace("{x}", str(tx)).replace("{y}", str(ty))
    if not url.startswith("http"):
        url = f"{backend}{url}"
    r = requests.get(url, timeout=30.0)
    if not r.ok:
        return None
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    arr = np.asarray(img)
    pxi = max(0, min(arr.shape[1] - 1, int(round(px))))
    pyi = max(0, min(arr.shape[0] - 1, int(round(py))))
    r_, g_, b_ = arr[pyi, pxi]
    return int(r_), int(g_), int(b_)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--bbox-half", type=float, default=0.013)
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=int, default=14)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = _load_samples(args.limit, args.seed)
    logger.info("Loaded %d samples (seed=%d)", len(samples), args.seed)
    class_dist = Counter(s["ref_cls"] for s in samples)
    logger.info("Class dist in subset: %s", dict(class_dist))

    rows: list[dict[str, Any]] = []
    confusion = np.zeros((len(CLASSES_FOR_IOU), len(CLASSES_FOR_IOU)), dtype=np.int64)
    n_stub = 0
    n_unmapped = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        lon = float(s["longitude"])
        lat = float(s["latitude"])
        ref_cls_raw = s["ref_cls"]
        ref_cls = REF_TO_MODEL_CLASS.get(ref_cls_raw)
        try:
            resp = _request_inference(args.backend, lon, lat, args.bbox_half)
        except Exception as e:
            logger.warning("[%d] inference request failed: %s", i, e)
            rows.append({"uid": s["uid"], "lon": lon, "lat": lat, "ref": ref_cls_raw, "pred": None, "kind": "error", "error": str(e)})
            continue
        kind = resp.get("kind")
        legend = resp.get("legend")
        tile_template = resp.get("tile_url") or ""
        if kind == "stub":
            n_stub += 1
        rgb = _fetch_tile_pixel(args.backend, tile_template, lon, lat, zoom=args.zoom)
        color_map = _legend_color_to_class(legend)
        pred_cls = _nearest_legend_class(rgb, color_map) if rgb else None
        if pred_cls is None:
            n_unmapped += 1
        rows.append({
            "uid": s["uid"], "lon": lon, "lat": lat,
            "ref": ref_cls_raw, "pred_raw": pred_cls, "kind": kind,
            "rgb": list(rgb) if rgb else None,
        })
        if ref_cls and pred_cls in CLASSES_FOR_IOU:
            ref_idx = CLASSES_FOR_IOU.index(ref_cls)
            pred_idx = CLASSES_FOR_IOU.index(pred_cls)
            confusion[ref_idx, pred_idx] += 1
        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            ips = (i + 1) / max(elapsed, 1e-6)
            logger.info("[%d/%d] ips=%.2f kind=%s ref=%s → pred=%s", i + 1, len(samples), ips, kind, ref_cls_raw, pred_cls)

    elapsed = time.time() - t0
    diag = np.diag(confusion).astype(np.float64)
    gt_total = confusion.sum(axis=1).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    denom = gt_total + pred_total - diag
    iou = np.where(denom > 0, diag / np.maximum(denom, 1), 0.0)
    n_eval = int(confusion.sum())
    overall_acc = float(diag.sum() / max(n_eval, 1))
    miou = float(iou.mean())

    print(f"\n=== FT-Mangrove on olmoearth_projects_mangrove (n_evaluated={n_eval}) ===")
    print(f"requested {len(samples)}; stub={n_stub}; unmapped={n_unmapped}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"overall acc : {overall_acc:.4f}")
    print(f"mIoU        : {miou:.4f}")
    print("per-class IoU:")
    for c, v in zip(CLASSES_FOR_IOU, iou):
        print(f"  {c:10s} {v:.4f}")
    print("\nconfusion matrix (rows=ref, cols=pred):")
    print("            " + " ".join(f"{c:>10s}" for c in CLASSES_FOR_IOU))
    for r, row in enumerate(confusion):
        print(f"  {CLASSES_FOR_IOU[r]:10s} " + " ".join(f"{v:>10d}" for v in row))

    if args.output:
        out = {
            "model": MODEL_REPO_ID,
            "dataset": "allenai/olmoearth_projects_mangrove",
            "n_requested": len(samples),
            "n_evaluated": n_eval,
            "n_stub": n_stub,
            "n_unmapped": n_unmapped,
            "elapsed_s": elapsed,
            "overall_acc": overall_acc,
            "miou": miou,
            "per_class_iou": dict(zip(CLASSES_FOR_IOU, [float(v) for v in iou])),
            "confusion": confusion.tolist(),
            "classes": CLASSES_FOR_IOU,
            "samples": rows,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
