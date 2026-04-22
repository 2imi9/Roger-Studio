"""
TIPSv2 zero-shot auto-labeling service.

Uses Google DeepMind's TIPSv2 (CVPR 2026) for:
  1. Zero-shot land cover classification via text prompts
  2. Per-patch spatial segmentation (32x32 grid at 448px)
  3. Confidence scoring via cosine similarity

No training needed — just provide text labels and an image.

Models (Apache 2.0):
  - google/tipsv2-b14  (~180M vision encoder, fast)
  - google/tipsv2-l14  (~303M vision encoder, balanced)
  - google/tipsv2-g14  (~1B vision encoder, best quality)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Lazy imports — avoid torchvision entirely (broken nms operator on this env)
torch = None
F = None
Image = None


def _ensure_imports():
    global torch, F, Image
    if torch is None:
        import torch as _torch
        import torch.nn.functional as _F
        from PIL import Image as _Image
        torch = _torch
        F = _F
        Image = _Image

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_model = None
_model_name = None
_device = None

# Default land cover classes for geospatial auto-labeling
DEFAULT_CLASSES = [
    {"name": "Forest", "prompt": "dense forest or woodland with trees", "color": "#228b22"},
    {"name": "Cropland", "prompt": "agricultural cropland or farmland", "color": "#f0e68c"},
    {"name": "Grassland", "prompt": "grassland or meadow or pasture", "color": "#90ee90"},
    {"name": "Urban", "prompt": "urban area with buildings and roads", "color": "#808080"},
    {"name": "Water", "prompt": "water body such as river lake or ocean", "color": "#1e90ff"},
    {"name": "Barren", "prompt": "barren land or desert or bare soil", "color": "#d2b48c"},
    {"name": "Wetland", "prompt": "wetland or marsh or swamp", "color": "#5f9ea0"},
    {"name": "Snow", "prompt": "snow or ice covered surface", "color": "#e0f0ff"},
]

def _preprocess_image(image: "Image.Image") -> "torch.Tensor":
    """Resize to 448x448 and convert to [0,1] float tensor. No torchvision needed."""
    _ensure_imports()
    img = image.convert("RGB").resize((448, 448), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 448, 448)
    return tensor


def _get_model(model_name: str = "google/tipsv2-b14"):
    """Load TIPSv2 model (cached, lazy)."""
    global _model, _model_name, _device
    _ensure_imports()

    if _model is not None and _model_name == model_name:
        return _model

    logger.info(f"Loading TIPSv2 model: {model_name}")
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _model = _model.to(device).eval()
    _model_name = model_name
    _device = device
    logger.info(f"TIPSv2 loaded on {device}")
    return _model


def classify_image_zeroshot(
    image: Image.Image,
    classes: list[dict] | None = None,
    model_name: str = "google/tipsv2-b14",
) -> dict:
    """
    Zero-shot classify an image using TIPSv2 text-image alignment.

    Returns global class + per-patch spatial classification map.
    """
    model = _get_model(model_name)
    classes = classes or DEFAULT_CLASSES

    # Encode image
    pixel_values = _preprocess_image(image).to(_device)

    with torch.no_grad():
        out = model.encode_image(pixel_values)
        # out.cls_token: (1, 1, D), out.patch_tokens: (1, N, D)

        # Encode text prompts
        prompts = [c["prompt"] for c in classes]
        text_emb = model.encode_text(prompts)  # (num_classes, D)

    # Global classification (cls token)
    cls = F.normalize(out.cls_token[:, 0, :], dim=-1)  # (1, D)
    text_norm = F.normalize(text_emb, dim=-1)  # (C, D)
    global_sim = (cls @ text_norm.T).squeeze(0)  # (C,)
    global_probs = F.softmax(global_sim * 10, dim=0).cpu().numpy()

    # Per-patch classification (spatial map)
    patch_tokens = F.normalize(out.patch_tokens, dim=-1)  # (1, N, D)
    patch_sim = (patch_tokens @ text_norm.T).squeeze(0)  # (N, C)
    patch_probs = F.softmax(patch_sim * 10, dim=-1).cpu().numpy()  # (N, C)
    patch_labels = patch_probs.argmax(axis=-1)  # (N,)
    patch_confidence = patch_probs.max(axis=-1)  # (N,)

    # Compute grid size
    n_patches = patch_tokens.shape[1]
    grid_size = int(np.sqrt(n_patches))

    # Reshape to spatial grid
    label_map = patch_labels.reshape(grid_size, grid_size)
    confidence_map = patch_confidence.reshape(grid_size, grid_size)

    # Compute class distribution
    class_results = []
    for i, cls_def in enumerate(classes):
        mask = patch_labels == i
        pct = float(mask.sum()) / len(patch_labels) * 100
        avg_conf = float(patch_confidence[mask].mean()) if mask.any() else 0
        class_results.append({
            "id": i + 1,
            "name": cls_def["name"],
            "color": cls_def["color"],
            "percentage": round(pct, 1),
            "avg_confidence": round(avg_conf, 3),
            "pixel_count": int(mask.sum()),
        })

    class_results.sort(key=lambda c: c["percentage"], reverse=True)

    return {
        "global_class": classes[global_probs.argmax()]["name"],
        "global_confidence": round(float(global_probs.max()), 3),
        "classes": class_results,
        "label_map": label_map.tolist(),
        "confidence_map": confidence_map.tolist(),
        "grid_size": grid_size,
        "avg_confidence": round(float(patch_confidence.mean()), 3),
        "needs_review_pct": round(float((patch_confidence < 0.3).sum()) / len(patch_confidence) * 100, 1),
        "model": model_name,
        "method": "tipsv2_zeroshot",
    }


def _sliding_window_classify(
    rgb: "np.ndarray",
    classes: list[dict],
    model_name: str,
    tile_size: int = 448,
    overlap: float = 0.25,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Run TIPSv2 on overlapping tiles, merge predictions with confidence-weighted averaging.

    Returns label_full (H, W) and conf_full (H, W) at the ORIGINAL raster resolution.
    The grid per tile is ~32x32, so stride between tiles gives us ~1 prediction per
    (tile_size / 32) = 14 pixels. With overlap, boundaries align to ~7 pixels instead
    of ~22 pixels (single-shot).
    """
    _ensure_imports()
    from PIL import Image as PILImage

    H, W, _ = rgb.shape
    model = _get_model(model_name)
    n_classes = len(classes)

    # Pre-encode text prompts once
    prompts = [c["prompt"] for c in classes]
    with torch.no_grad():
        text_emb = model.encode_text(prompts)
        text_norm = F.normalize(text_emb, dim=-1)

    # Accumulate per-class probability × confidence, and confidence weights
    # Shape: (H, W, n_classes) for probs, (H, W) for weight sum
    class_prob_sum = np.zeros((H, W, n_classes), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    stride = int(tile_size * (1 - overlap))

    # Compute tile positions (inclusive of edge tiles)
    def tile_positions(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        positions = list(range(0, dim - tile_size + 1, stride))
        if positions[-1] + tile_size < dim:
            positions.append(dim - tile_size)
        return positions

    y_positions = tile_positions(H)
    x_positions = tile_positions(W)

    # Cosine falloff weight mask — edges contribute less than centers
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, tile_size), np.linspace(-1, 1, tile_size), indexing="ij"
    )
    weight_mask = np.cos(yy * np.pi / 2) * np.cos(xx * np.pi / 2)
    weight_mask = np.clip(weight_mask, 0.1, 1.0).astype(np.float32)

    logger.info(f"Sliding window: {len(y_positions)}x{len(x_positions)} tiles, stride={stride}")

    for y0 in y_positions:
        for x0 in x_positions:
            tile = rgb[y0 : y0 + tile_size, x0 : x0 + tile_size]

            # Pad if needed (edge tiles)
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded[:th, :tw] = tile
                tile = padded

            # Run TIPSv2
            pil = PILImage.fromarray(tile)
            pixel_values = _preprocess_image(pil).to(_device)
            with torch.no_grad():
                out = model.encode_image(pixel_values)
                patch_tokens = F.normalize(out.patch_tokens, dim=-1)
                patch_sim = (patch_tokens @ text_norm.T).squeeze(0)  # (N, C)
                patch_probs = F.softmax(patch_sim * 10, dim=-1).cpu().numpy()  # (N, C)

            n_patches = patch_probs.shape[0]
            grid = int(np.sqrt(n_patches))
            prob_grid = patch_probs.reshape(grid, grid, n_classes)  # (grid, grid, C)

            # Upscale each class channel to tile_size x tile_size via BILINEAR
            prob_full = np.zeros((tile_size, tile_size, n_classes), dtype=np.float32)
            for c in range(n_classes):
                ch_img = PILImage.fromarray((prob_grid[:, :, c] * 255).astype(np.uint8))
                prob_full[:, :, c] = (
                    np.array(ch_img.resize((tile_size, tile_size), PILImage.BILINEAR)).astype(
                        np.float32
                    )
                    / 255
                )

            # Accumulate into global arrays
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            wy = y1 - y0
            wx = x1 - x0

            w = weight_mask[:wy, :wx]
            class_prob_sum[y0:y1, x0:x1] += prob_full[:wy, :wx] * w[..., None]
            weight_sum[y0:y1, x0:x1] += w

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-6)
    class_prob_final = class_prob_sum / weight_sum[..., None]

    label_full = class_prob_final.argmax(axis=-1).astype(np.int32)
    conf_full = class_prob_final.max(axis=-1).astype(np.float32)

    return label_full, conf_full


def auto_label_geotiff_tipsv2(
    filepath: str,
    classes: list[dict] | None = None,
    model_name: str = "google/tipsv2-b14",
    tile_size: int = 448,
    sliding_window: bool = True,
) -> dict:
    """
    Auto-label a GeoTIFF using TIPSv2 zero-shot classification.

    Args:
        sliding_window: When True (default), uses overlapping tile inference for
                        accurate pixel-level boundaries. When False, single-shot
                        32x32 grid (fast but blocky).

    Returns labeled GeoJSON polygons with confidence scores.
    """
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape, mapping

    _ensure_imports()
    from PIL import Image as PILImage

    classes = classes or DEFAULT_CLASSES
    model = _get_model(model_name)

    with rasterio.open(filepath) as src:
        data = src.read()  # (bands, H, W)
        transform = src.transform
        src_crs = src.crs
        bands, height, width = data.shape

    # Convert raster to RGB image for TIPSv2
    if bands >= 3:
        rgb = data[:3].transpose(1, 2, 0)  # (H, W, 3)
    elif bands == 1:
        # Single band → grayscale → RGB
        band = data[0]
        # Normalize to 0-255
        bmin, bmax = np.nanpercentile(band[band != 0], [2, 98]) if np.any(band != 0) else (0, 1)
        normalized = np.clip((band - bmin) / (bmax - bmin + 1e-8) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([normalized] * 3, axis=-1)
    else:
        rgb = data[:3].transpose(1, 2, 0) if bands >= 3 else np.stack([data[0]] * 3, axis=-1)

    # Ensure uint8
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if np.any(rgb > 0) else (0, 255)
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb)

    # Global classification via cls token (for summary/badge)
    global_result = classify_image_zeroshot(image, classes, model_name)

    if sliding_window and (height > tile_size or width > tile_size):
        # High-accuracy path: overlapping tiles merged with cosine-weighted confidence
        label_full, conf_full = _sliding_window_classify(
            rgb, classes, model_name, tile_size=tile_size, overlap=0.25
        )
    else:
        # Fast single-shot path — 32x32 grid upscaled (blocky boundaries)
        label_grid = np.array(global_result["label_map"])
        conf_grid = np.array(global_result["confidence_map"])
        label_img = PILImage.fromarray(label_grid.astype(np.uint8))
        label_full = np.array(label_img.resize((width, height), PILImage.NEAREST)).astype(np.int32)
        conf_img = PILImage.fromarray((conf_grid * 255).astype(np.uint8))
        conf_full = np.array(conf_img.resize((width, height), PILImage.BILINEAR)).astype(np.float32) / 255

    result = global_result

    # Set up CRS reprojection to WGS84
    _reproject_poly = None
    if src_crs and str(src_crs) != "EPSG:4326":
        try:
            from pyproj import Transformer, CRS as ProjCRS
            from shapely.ops import transform as shapely_transform
            from functools import partial
            transformer = Transformer.from_crs(ProjCRS(src_crs), ProjCRS.from_epsg(4326), always_xy=True)
            _reproject_poly = partial(shapely_transform, transformer.transform)
        except Exception:
            pass

    # Vectorize to GeoJSON polygons
    features_list = []
    for geom, value in shapes(label_full.astype(np.int32), transform=transform):
        cls_id = int(value)
        if cls_id < 0 or cls_id >= len(classes):
            continue

        poly = shapely_shape(geom)

        # Tight simplification — 0.5 pixel keeps spatial accuracy within half-pixel
        pixel_size = abs(transform.a)
        poly = poly.simplify(tolerance=pixel_size * 0.5, preserve_topology=True)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                continue

        # Reproject to WGS84 if needed
        if _reproject_poly:
            try:
                poly = _reproject_poly(poly)
            except Exception:
                continue
            if poly.is_empty:
                continue

        cls_def = classes[cls_id]

        # Average confidence for this segment
        seg_mask = label_full == cls_id
        seg_conf = float(conf_full[seg_mask].mean()) if seg_mask.any() else 0

        features_list.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "class_id": cls_id + 1,
                "class_name": cls_def["name"],
                "color": cls_def["color"],
                "confidence": round(seg_conf, 3),
                "area_m2": round(poly.area, 1),
                "needs_review": seg_conf < 0.3,
            },
        })

    features_list.sort(key=lambda f: f["properties"]["area_m2"], reverse=True)

    # Summary
    total_area = sum(f["properties"]["area_m2"] for f in features_list)
    class_summary = {}
    for f in features_list:
        name = f["properties"]["class_name"]
        if name not in class_summary:
            class_summary[name] = {"area_m2": 0, "count": 0, "color": f["properties"]["color"]}
        class_summary[name]["area_m2"] += f["properties"]["area_m2"]
        class_summary[name]["count"] += 1

    for s in class_summary.values():
        s["percentage"] = round(s["area_m2"] / max(total_area, 1) * 100, 1)

    return {
        "type": "FeatureCollection",
        "features": features_list,
        "properties": {
            "total_features": len(features_list),
            "total_area_m2": round(total_area, 1),
            "n_classes": len(classes),
            "class_summary": class_summary,
            "needs_review_count": sum(1 for f in features_list if f["properties"]["needs_review"]),
            "avg_confidence": result["avg_confidence"],
            "global_class": result["global_class"],
            "global_confidence": result["global_confidence"],
            "method": "tipsv2_zeroshot",
            "model_version": model_name,
        },
    }
