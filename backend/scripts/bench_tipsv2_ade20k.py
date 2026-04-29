"""ADE20K mIoU benchmark for TIPSv2 zero-shot semantic segmentation.

Loads the canonical ADE20K val split (2000 images, 150 classes) from the
MIT release zip, runs sliding-window TIPSv2 across each image, and reports
mean IoU + per-class IoU. The published TIPSv2-L/14 number on this exact
bench is 25.06 mIoU (with the values trick + 9 prompt templates + slide
stride 336). Our impl uses a single prompt template and raw patch tokens,
so the absolute number is expected to be lower — the value of the bench
is verifying the recent plumbing fixes (logit-scale calibration, dataset
seeding) actually move the needle off the random baseline (~0.67 % for
150 classes).

Usage from ``backend/``::

    python scripts/bench_tipsv2_ade20k.py --model b14 --limit 50

Args:
    --model {b14,l14,g14}  TIPSv2 model size to evaluate.
    --limit N              Cap evaluated images (smoke runs).
    --stride INT           Sliding-window stride in pixels (default 336 = 448*0.75).
    --output PATH          Write per-class JSON to this path.

Reads:  backend/data/ade20k/ADEChallengeData2016.zip (auto-downloaded if absent)
Writes: optional JSON results
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

# Backend root is one above this script's parent. Without this on sys.path
# the ``app.services.tipsv2_labeler`` import below fails when the script
# is run from elsewhere.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from app.services.tipsv2_labeler import (  # noqa: E402
    PROMPT_TEMPLATES,
    _encode_image_dense,
    _ensure_imports,
    _get_model,
    _preprocess_image,
    encode_text_with_templates,
)

logger = logging.getLogger("bench_tipsv2_ade20k")

ADE_DATA_DIR = _BACKEND_ROOT / "data" / "ade20k"
ADE_ZIP = ADE_DATA_DIR / "ADEChallengeData2016.zip"
ADE_VAL_IMAGES = "ADEChallengeData2016/images/validation"
ADE_VAL_ANNOTATIONS = "ADEChallengeData2016/annotations/validation"

MODEL_REPOS = {
    "b14": "google/tipsv2-b14",
    "l14": "google/tipsv2-l14",
    "g14": "google/tipsv2-g14",
}

TILE_SIZE = 448

# Canonical 150-class ADE20K labels (id 1..150; id 0 is "unlabeled" — ignored
# in mIoU). Order matches MIT's ``objectInfo150.txt``. Each entry uses the
# first synonym for the prompt — matches the convention of public TIPSv2
# zero-shot evals.
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass", "clock",
    "flag",
]
N_CLASSES = len(ADE20K_CLASSES)
assert N_CLASSES == 150

SINGLE_PROMPT_TEMPLATE = "a photo of a {}"


def _ensure_ade20k_extracted() -> Path:
    """Ensure the val split is extracted under data/ade20k/. Returns the
    extraction root (the dir that contains ``ADEChallengeData2016/``).

    Skips re-extraction if both ``images/validation`` and
    ``annotations/validation`` already exist.
    """
    val_imgs_dir = ADE_DATA_DIR / ADE_VAL_IMAGES
    val_anno_dir = ADE_DATA_DIR / ADE_VAL_ANNOTATIONS
    if val_imgs_dir.is_dir() and val_anno_dir.is_dir():
        n_imgs = len(list(val_imgs_dir.glob("*.jpg")))
        n_anno = len(list(val_anno_dir.glob("*.png")))
        if n_imgs > 0 and n_anno > 0:
            logger.info("ADE20K val already extracted: %d images, %d annotations", n_imgs, n_anno)
            return ADE_DATA_DIR
    if not ADE_ZIP.exists():
        raise FileNotFoundError(
            f"Expected ADE20K release at {ADE_ZIP}. "
            "Download with: curl -fL -o {zip} http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip".format(zip=ADE_ZIP)
        )
    logger.info("Extracting ADE20K val split from %s", ADE_ZIP)
    with zipfile.ZipFile(ADE_ZIP) as zf:
        # Selective extraction — only the validation halves of images and
        # annotations. Skipping the 20k-image training set is a 4× speedup
        # on this step.
        members = [m for m in zf.namelist() if (ADE_VAL_IMAGES in m or ADE_VAL_ANNOTATIONS in m) and not m.endswith("/")]
        for m in members:
            zf.extract(m, ADE_DATA_DIR)
    logger.info("Extracted %d files", len(members))
    return ADE_DATA_DIR


def _list_val_pairs() -> list[tuple[Path, Path]]:
    """Return [(image_path, annotation_path), ...] for the val split."""
    img_dir = ADE_DATA_DIR / ADE_VAL_IMAGES
    anno_dir = ADE_DATA_DIR / ADE_VAL_ANNOTATIONS
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        anno_path = anno_dir / (img_path.stem + ".png")
        if anno_path.exists():
            pairs.append((img_path, anno_path))
    return pairs


def _tile_positions(dim: int, tile: int, stride: int) -> list[int]:
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] + tile < dim:
        pos.append(dim - tile)
    return pos


@torch.no_grad()
def _predict_slide(
    model,
    text_norm: torch.Tensor,
    logit_scale: float,
    image: Image.Image,
    device: torch.device,
    stride: int,
    attn_mode: str = "default",
    cls_subtract: float = 0.0,
    drop_residual: bool = False,
) -> np.ndarray:
    """Sliding-window forward pass; returns argmax label (H, W) at the
    image's original pixel size, classes 0..N_CLASSES-1."""
    img = image.convert("RGB")
    W, H = img.size
    rgb = np.asarray(img, dtype=np.uint8)

    pad_h = max(0, TILE_SIZE - H)
    pad_w = max(0, TILE_SIZE - W)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    Hp, Wp = rgb.shape[:2]

    n_classes = text_norm.shape[0]
    prob_sum = torch.zeros((n_classes, Hp, Wp), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, TILE_SIZE, device=device),
        torch.linspace(-1, 1, TILE_SIZE, device=device),
        indexing="ij",
    )
    weight_mask = (torch.cos(yy * np.pi / 2) * torch.cos(xx * np.pi / 2)).clamp(0.1, 1.0)

    y_positions = _tile_positions(Hp, TILE_SIZE, stride)
    x_positions = _tile_positions(Wp, TILE_SIZE, stride)

    for y0 in y_positions:
        for x0 in x_positions:
            tile_np = rgb[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            tile_pil = Image.fromarray(tile_np)
            pixel_values = _preprocess_image(tile_pil).to(device)
            _, patch_tokens = _encode_image_dense(
                model, pixel_values,
                attn_mode=attn_mode,
                cls_subtract=cls_subtract,
                drop_residual=drop_residual,
            )
            patch = F.normalize(patch_tokens, dim=-1)
            sim = (patch @ text_norm.T).squeeze(0)
            probs = F.softmax(sim * logit_scale, dim=-1)

            P = probs.shape[0]
            grid = int(round(P ** 0.5))
            assert grid * grid == P, f"non-square patch grid: {P}"
            probs = probs.view(grid, grid, n_classes).permute(2, 0, 1).unsqueeze(0)
            probs_full = F.interpolate(
                probs, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
            ).squeeze(0)

            prob_sum[:, y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] += probs_full * weight_mask
            weight_sum[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] += weight_mask

    weight_sum = weight_sum.clamp(min=1e-6)
    prob_final = prob_sum / weight_sum.unsqueeze(0)
    prob_final = prob_final[:, :H, :W]
    return prob_final.argmax(dim=0).to(torch.int32).cpu().numpy()


def _update_confusion(conf: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """gt: 0=ignore, 1..150 valid. pred: 0..149."""
    valid = (gt > 0) & (gt <= N_CLASSES)
    if not valid.any():
        return
    gt_idx = gt[valid].astype(np.int64) - 1
    pred_idx = pred[valid].astype(np.int64)
    flat = gt_idx * N_CLASSES + pred_idx
    counts = np.bincount(flat, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    conf += counts


def _summarize(conf: np.ndarray) -> tuple[np.ndarray, float, float]:
    tp = np.diag(conf).astype(np.float64)
    gt_total = conf.sum(axis=1).astype(np.float64)
    pred_total = conf.sum(axis=0).astype(np.float64)
    denom = gt_total + pred_total - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    appears = gt_total > 0
    miou = float(iou[appears].mean()) if appears.any() else 0.0
    pixel_acc = float(tp.sum() / max(gt_total.sum(), 1))
    return iou, miou, pixel_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODEL_REPOS), default="b14")
    ap.add_argument("--limit", type=int, default=None, help="cap evaluated images")
    ap.add_argument("--stride", type=int, default=336)
    ap.add_argument("--output", default=None, help="path to write per-class JSON")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--templates", action="store_true",
        help="average text embeddings across the 9 TCL prompt templates "
             "(default: single 'a photo of a {}' prompt)",
    )
    ap.add_argument(
        "--values-trick", action="store_true",
        help="(legacy alias) equivalent to --attn-mode values",
    )
    ap.add_argument(
        "--attn-mode", choices=("default", "values", "msa"), default=None,
        help="last-block attention modification: default | values (MaskCLIP) | "
             "msa (SegEarth-OV modulated self-attention). "
             "Default depends on --values-trick for back-compat.",
    )
    ap.add_argument(
        "--cls-subtract", type=float, default=0.0,
        help="SegEarth-OV CLS-bias subtraction λ (Eq. 9). 0.0 disables; 0.3 is the paper default.",
    )
    ap.add_argument(
        "--drop-residual", action="store_true",
        help="ClearCLIP modification #1 — drop the attention residual on the last block.",
    )
    args = ap.parse_args()
    if args.attn_mode is None:
        args.attn_mode = "values" if args.values_trick else "default"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _ensure_ade20k_extracted()
    pairs = _list_val_pairs()
    if not pairs:
        logger.error("No val pairs found under %s", ADE_DATA_DIR / ADE_VAL_IMAGES)
        return 1
    if args.limit is not None:
        pairs = pairs[: args.limit]
    logger.info("evaluating on %d images", len(pairs))

    _ensure_imports()
    repo_id = MODEL_REPOS[args.model]
    logger.info("loading %s", repo_id)
    model = _get_model(repo_id)
    device = next(model.parameters()).device
    logit_scale = 1.0 / float(model.config.temperature)
    logger.info("device=%s logit_scale=%.2f temperature=%.6f", device, logit_scale, model.config.temperature)

    if args.templates:
        text_norm = encode_text_with_templates(model, ADE20K_CLASSES)
        logger.info(
            "encoded %d class prompts averaged over %d templates",
            N_CLASSES, len(PROMPT_TEMPLATES),
        )
    else:
        prompts = [SINGLE_PROMPT_TEMPLATE.format(c) for c in ADE20K_CLASSES]
        with torch.no_grad():
            text_emb = model.encode_text(prompts)
            text_norm = F.normalize(text_emb, dim=-1).to(device)
        logger.info("encoded %d class prompts (single template)", N_CLASSES)
    logger.info("attn_mode=%s cls_subtract=%.2f", args.attn_mode, args.cls_subtract)

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    t0 = time.time()
    for i, (img_path, anno_path) in enumerate(pairs):
        with Image.open(img_path) as image:
            image.load()
        with Image.open(anno_path) as anno:
            gt = np.asarray(anno, dtype=np.int32)
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        pred = _predict_slide(
            model, text_norm, logit_scale, image, device,
            stride=args.stride,
            attn_mode=args.attn_mode,
            cls_subtract=args.cls_subtract,
            drop_residual=args.drop_residual,
        )
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred.astype(np.int32)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                ),
                dtype=np.int32,
            )
        _update_confusion(confusion, gt, pred)

        if (i + 1) % args.log_every == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t0
            ips = (i + 1) / max(elapsed, 1e-6)
            iou, miou, pix = _summarize(confusion)
            logger.info(
                "[%d/%d] ips=%.2f mIoU(running)=%.4f pix_acc=%.4f",
                i + 1, len(pairs), ips, miou, pix,
            )

    iou, miou, pix = _summarize(confusion)
    print(f"\n=== {repo_id} on ADE20K validation (n={len(pairs)}) ===")
    print(f"mIoU       : {miou:.4f}  ({miou * 100:.2f} %)")
    print(f"pixel acc  : {pix:.4f}")
    print(f"reference  : 25.06 mIoU (TIPSv2-L/14, values trick + 9 templates + stride 336)")
    print(
        f"impl notes : templates={'9-TCL' if args.templates else 'single'}, "
        f"attn_mode={args.attn_mode}, cls_subtract={args.cls_subtract}, "
        f"BILINEAR upsample, stride={args.stride}"
    )

    order = np.argsort(iou)
    print("\nWorst 10 classes (with non-zero ground truth):")
    appears = confusion.sum(axis=1) > 0
    appearing_order = [i for i in order if appears[i]]
    for idx in appearing_order[:10]:
        print(f"  {iou[idx]:.4f}  {ADE20K_CLASSES[idx]}")
    print("\nBest 10 classes:")
    for idx in appearing_order[::-1][:10]:
        print(f"  {iou[idx]:.4f}  {ADE20K_CLASSES[idx]}")

    if args.output:
        out = {
            "model": repo_id,
            "split": "validation",
            "n_images": len(pairs),
            "stride": args.stride,
            "templates": args.templates,
            "attn_mode": args.attn_mode,
            "cls_subtract": args.cls_subtract,
            "drop_residual": args.drop_residual,
            "values_trick": args.attn_mode == "values",
            "miou": miou,
            "pixel_acc": pix,
            "logit_scale": logit_scale,
            "per_class": [
                {"id": i + 1, "name": ADE20K_CLASSES[i], "iou": float(iou[i]), "appears": bool(appears[i])}
                for i in range(N_CLASSES)
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info("wrote results to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
