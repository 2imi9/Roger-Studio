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
import os
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

# 9-template prompt set used for zero-shot ADE20K eval in the official
# TIPSv2 release (`pytorch/TIPS_zeroshot_segmentation.ipynb`,
# ``_TCL_PROMPTS`` constant). Averaging text embeddings across these
# templates is worth ~2–5 mIoU on dense seg vs a single prompt — the
# templates probe complementary slices of the joint image-text space and
# the average is a more stable class anchor than any one prompt.
PROMPT_TEMPLATES = (
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a photo of {}s.",
)

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


def encode_text_with_templates(model, class_terms: list[str]) -> "torch.Tensor":
    """Encode each class term across ``PROMPT_TEMPLATES`` and average the
    embeddings (L2-normalised before averaging). Returns a (C, D) tensor of
    L2-normalised class anchors.

    The averaging is done in the L2-normalised embedding space — that's
    what the official TIPSv2 release does for its zero-shot eval and
    matches the standard CLIP-style template-ensembling pattern. Averaging
    raw embeddings would let a single template with large magnitude
    dominate the others; normalising first gives every template equal say.
    """
    _ensure_imports()
    device = next(model.parameters()).device
    out = torch.zeros((len(class_terms), model.config.embed_dim), device=device)
    for tpl in PROMPT_TEMPLATES:
        prompts = [tpl.format(t) for t in class_terms]
        emb = model.encode_text(prompts).to(device)
        emb = F.normalize(emb, dim=-1)
        out += emb
    out = out / len(PROMPT_TEMPLATES)
    return F.normalize(out, dim=-1)


def _encode_image_dense(
    model,
    pixel_values,
    *,
    attn_mode: str = "default",
    cls_subtract: float = 0.0,
    drop_residual: bool = False,
):
    """Run the vision encoder for dense readout, with optional MaskCLIP /
    SegEarth-OV style modifications on the last block + CLS-bias removal.

    Args:
        attn_mode:
          ``"default"``  — unchanged (model's own forward).
          ``"values"``   — MaskCLIP / TIPSv2 official ``encode_image_value_attention``.
                            Replace last attention with ``ls1(proj(v))`` and skip MLP.
          ``"msa"``      — SegEarth-OV's "modulated self-attention" (CVPR'25):
                            sum of q·q, k·k, v·v softmax-attentions, applied to v.
                            Outperformed plain values on RS imagery in their ablations.
        cls_subtract:
          λ in SegEarth-OV's Eq. 9 — subtracts ``λ · cls_token`` from each
          patch token to remove the global-context bias the [CLS] token
          contaminates patch tokens with during contrastive pretraining.
          Recommended λ=0.3 (paper default). 0.0 disables.

    Returns ``(cls_token: (B, 1, D), patch_tokens: (B, N, D))``.
    """
    _ensure_imports()
    if attn_mode == "default":
        # Fast path: use the model's own forward unchanged.
        out = model.encode_image(pixel_values)
        cls_token, patch_tokens = out.cls_token, out.patch_tokens
    elif attn_mode in ("values", "msa"):
        cls_token, patch_tokens = _encode_image_modified_last_block(
            model, pixel_values, attn_mode, drop_residual=drop_residual,
        )
    else:
        raise ValueError(f"unknown attn_mode={attn_mode!r}")

    if cls_subtract != 0.0:
        # SegEarth-OV Eq. 9: Ô = O[1:hw+1] − λ · O[0], where O[0] is the
        # CLS token broadcast over the patch positions. Trims the global-
        # context bias the contrastive [CLS] objective bakes into every
        # patch — gain on dense seg without cost.
        patch_tokens = patch_tokens - cls_subtract * cls_token

    return cls_token, patch_tokens


def _encode_image_modified_last_block(
    model, pixel_values, attn_mode: str, *, drop_residual: bool = False,
):
    """Forward through the vision encoder with a modified last attention
    block. Helper for ``_encode_image_dense``.

    When ``drop_residual`` is True, the residual connection on the last
    block is also removed (ClearCLIP ECCV'24, modification #1) — only the
    modulated attention output passes through ``last.ls1``, no addition
    of the input ``x``. Empirically worth ~1–2 mIoU on dense seg in
    addition to the values/M-SA + no-FFN modifications.
    """
    vit = model.vision_encoder
    pixel_values = pixel_values.to(model.device)
    with torch.no_grad():
        x = vit.prepare_tokens_with_masks(pixel_values)
        for blk in vit.blocks[:-1]:
            x = blk(x)
        last = vit.blocks[-1]
        x_n1 = last.norm1(x)
        attn = last.attn
        b_dim, n_dim, c_dim = x_n1.shape
        qkv = (
            attn.qkv(x_n1)
            .reshape(b_dim, n_dim, 3, attn.num_heads, c_dim // attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, num_heads, n, head_dim)

        if attn_mode == "values":
            x_attn = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
        else:  # "msa" — SegEarth-OV Eq. 10
            scale = attn.scale  # head_dim**-0.5
            attn_qq = (q * scale @ q.transpose(-2, -1)).softmax(dim=-1)
            attn_kk = (k * scale @ k.transpose(-2, -1)).softmax(dim=-1)
            attn_vv = (v * scale @ v.transpose(-2, -1)).softmax(dim=-1)
            attn_sum = attn_qq + attn_kk + attn_vv
            x_attn = (attn_sum @ v).transpose(1, 2).reshape(b_dim, n_dim, c_dim)

        x_attn = attn.proj(x_attn)
        x_attn = last.ls1(x_attn)
        # MLP residual intentionally skipped — matches MaskCLIP / SegEarth-OV.
        # Optional: also drop the attention residual (ClearCLIP ECCV'24).
        x = x_attn if drop_residual else x + x_attn
        x_norm = vit.norm(x)
        cls_token = x_norm[:, :1]
        patch_tokens = x_norm[:, 1 + vit.num_register_tokens :]
        return cls_token, patch_tokens


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

    # Global classification (cls token).
    # ``temperature`` comes from the model config (~0.005 for B/14, similar
    # for L/14 and g/14). Earlier code hardcoded ``* 10`` which produced
    # near-uniform softmax (8-class avg confidence ~0.15 — barely above the
    # 0.125 random baseline). The trained logit scale is ~1/0.005 ≈ 200,
    # which is what calibrates the cosine similarities into a peaky
    # distribution. Use the model's own value so each model size gets the
    # scale it was trained with.
    logit_scale = 1.0 / float(model.config.temperature)
    cls = F.normalize(out.cls_token[:, 0, :], dim=-1)  # (1, D)
    text_norm = F.normalize(text_emb, dim=-1)  # (C, D)
    global_sim = (cls @ text_norm.T).squeeze(0)  # (C,)
    global_probs = F.softmax(global_sim * logit_scale, dim=0).cpu().numpy()

    # Per-patch classification (spatial map)
    patch_tokens = F.normalize(out.patch_tokens, dim=-1)  # (1, N, D)
    patch_sim = (patch_tokens @ text_norm.T).squeeze(0)  # (N, C)
    patch_probs = F.softmax(patch_sim * logit_scale, dim=-1).cpu().numpy()  # (N, C)
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
    # Use the model's trained logit scale (see classify_image_zeroshot).
    logit_scale = 1.0 / float(model.config.temperature)

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
                patch_probs = F.softmax(patch_sim * logit_scale, dim=-1).cpu().numpy()  # (N, C)

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


def _write_label_raster(
    src_path: str,
    label_full: "np.ndarray",
    conf_full: "np.ndarray",
    classes: list[dict],
    out_path: str,
) -> None:
    """Render the per-pixel TIPSv2 zero-shot result as a 3-band RGB GeoTIFF.

    The polygon vectorization throws away soft probabilities (each patch
    just becomes its argmax class). The raster preserves both the class
    decision AND the confidence behind it: each pixel is rendered as the
    user's class color, blended toward white by ``1 - confidence`` so
    ambiguous regions visually fade. Researchers reading the map can see
    where the model was sure vs uncertain at a glance — something the
    polygons can't show.

    Output is georeferenced with the same CRS / transform as the source
    so the existing ``render_geotiff_tile`` path serves it directly via
    ``/api/datasets/{filename}/tiles/{z}/{x}/{y}.png``.
    """
    import rasterio  # noqa: PLC0415

    # Hex color palette → uint8 triples
    palette = np.full((max(len(classes), 1), 3), 200, dtype=np.float32)
    for i, c in enumerate(classes):
        h = (c.get("color") or "#888888").lstrip("#")
        try:
            palette[i] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except (ValueError, IndexError):
            palette[i] = (136, 136, 136)

    # Look up each pixel's class color, then blend toward white by confidence
    safe_labels = np.clip(label_full.astype(np.int64), 0, len(classes) - 1)
    color_per_pixel = palette[safe_labels]  # (H, W, 3)
    conf3 = np.clip(conf_full, 0.0, 1.0)[..., None].astype(np.float32)  # (H, W, 1)
    rgb_pixels = color_per_pixel * conf3 + 255.0 * (1.0 - conf3)
    rgb_pixels = np.clip(rgb_pixels, 0, 255).astype(np.uint8)
    rgb_chw = rgb_pixels.transpose(2, 0, 1)  # (3, H, W)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(count=3, dtype="uint8", nodata=None, photometric="rgb")
    # Drop any compression/predictor that conflicts with uint8 RGB.
    for k in ("predictor",):
        profile.pop(k, None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb_chw)


def auto_label_geotiff_tipsv2(
    filepath: str,
    classes: list[dict] | None = None,
    model_name: str = "google/tipsv2-b14",
    tile_size: int = 448,
    sliding_window: bool = True,
    raster_out_path: str | None = None,
) -> dict:
    """
    Auto-label a GeoTIFF using TIPSv2 zero-shot classification.

    Args:
        sliding_window: When True (default), uses overlapping tile inference for
                        accurate pixel-level boundaries. When False, single-shot
                        32x32 grid (fast but blocky).
        raster_out_path: When provided, also writes a 3-band RGB GeoTIFF where
                        pixel color = class color × confidence + white × (1 -
                        confidence). The router registers it as a dataset so
                        the frontend can drop it on the map alongside the
                        polygons. Polygons throw away soft probabilities;
                        the raster preserves the uncertainty.

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

    # Geodesic area in m² for the WGS84 polygon. Earlier code did ``poly.area``
    # AFTER reprojection, which returned degrees² (~1e-10 of the real m²),
    # collapsing every feature's reported area to 0 and breaking the
    # percentage roll-up in the class summary. Geod gives ellipsoidal area
    # that's correct regardless of source CRS.
    from pyproj import Geod  # noqa: PLC0415
    _geod = Geod(ellps="WGS84")

    def _area_m2(p) -> float:
        if p is None or p.is_empty:
            return 0.0
        if p.geom_type == "Polygon":
            return abs(_geod.geometry_area_perimeter(p)[0])
        if p.geom_type == "MultiPolygon":
            return sum(abs(_geod.geometry_area_perimeter(g)[0]) for g in p.geoms)
        return 0.0

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
                "area_m2": round(_area_m2(poly), 1),
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

    # Write the per-pixel raster (confidence-modulated class colors) to
    # the path the caller chose. Failure here is non-fatal: polygons are
    # still returned. Researchers asked for a raster overlay because the
    # polygon argmax loses soft probabilities, so the raster preserves
    # uncertainty as fading-toward-white at low-confidence pixels.
    raster_filename: str | None = None
    if raster_out_path:
        try:
            _write_label_raster(filepath, label_full, conf_full, classes, raster_out_path)
            raster_filename = os.path.basename(raster_out_path)
        except Exception as e:
            logger.warning("tipsv2: raster write failed (%s) — polygons only", e)

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
            "raster_filename": raster_filename,
        },
    }
