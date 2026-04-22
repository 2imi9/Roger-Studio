"""
Auto-labeling service — segment raster data and classify regions.

Architecture:
  Step 1: Feature extraction    (raw spectral bands — NDVI / NDWI / NDBI / etc.)
  Step 2: Spatial segmentation  (K-means clustering over spectral features)
  Step 3: Classification        (spectral heuristics per cluster centroid)
  Step 4: Confidence scoring    (cluster purity + distance to centroid)
  Step 5: Vectorization         (raster segments → GeoJSON polygons)

For foundation-model-backed auto-labeling (OlmoEarth encoder embeddings
or TIPSv2 zero-shot classification) see ``olmoearth_inference.py`` and
``tipsv2_labeler.py`` respectively. Those are independent pipelines the
frontend can route to via the ``method`` query param on
``/api/auto-label/{filename}``.

The pipeline returns labeled GeoJSON with per-feature confidence scores,
ready for human QC in Roger Studio.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Land cover class definitions (ESA WorldCover-compatible)
LAND_COVER_CLASSES = [
    {"id": 1, "name": "Forest", "color": "#228b22", "spectral_hint": "high_nir_high_green"},
    {"id": 2, "name": "Cropland", "color": "#f0e68c", "spectral_hint": "medium_ndvi_seasonal"},
    {"id": 3, "name": "Grassland", "color": "#90ee90", "spectral_hint": "medium_nir_medium_green"},
    {"id": 4, "name": "Urban", "color": "#808080", "spectral_hint": "high_swir_low_ndvi"},
    {"id": 5, "name": "Water", "color": "#1e90ff", "spectral_hint": "low_nir_high_blue"},
    {"id": 6, "name": "Barren", "color": "#d2b48c", "spectral_hint": "high_swir_low_ndvi"},
    {"id": 7, "name": "Wetland", "color": "#5f9ea0", "spectral_hint": "mixed"},
    {"id": 8, "name": "Snow/Ice", "color": "#e0f0ff", "spectral_hint": "high_all_bands"},
]


def auto_label_raster(filepath: str, n_classes: int = 6, min_segment_pixels: int = 50) -> dict:
    """
    Auto-label a GeoTIFF raster using spectral clustering.

    Returns GeoJSON FeatureCollection with labeled polygons + confidence scores.
    """
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import array_bounds
    from shapely.geometry import shape as shapely_shape, mapping
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    with rasterio.open(filepath) as src:
        # Read all bands
        data = src.read()  # (bands, height, width)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bands, height, width = data.shape
        bounds = src.bounds

    logger.info(f"Auto-label: {filepath} — {bands} bands, {width}x{height}, {n_classes} classes")

    # --- Step 1: Feature extraction ---
    # Reshape to (pixels, bands) for clustering
    pixels = data.reshape(bands, -1).T.astype(np.float32)  # (H*W, bands)

    # Build valid pixel mask (exclude nodata)
    if nodata is not None:
        valid_mask = ~np.any(np.isclose(pixels, nodata), axis=1)
    else:
        valid_mask = ~np.any(np.isnan(pixels), axis=1) & ~np.all(pixels == 0, axis=1)

    valid_pixels = pixels[valid_mask]

    if len(valid_pixels) < n_classes * 10:
        return _empty_result(bounds)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(valid_pixels)

    # Add spatial coordinates as weak features (helps spatial coherence)
    ys, xs = np.mgrid[0:height, 0:width]
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    valid_coords = coords[valid_mask]
    spatial_weight = 0.3  # Low weight — spectral dominates
    coord_scaled = valid_coords / np.array([width, height]) * spatial_weight
    features = np.hstack([features, coord_scaled])

    # --- Step 2: Segmentation via K-means ---
    n_classes = min(n_classes, len(valid_pixels) // 10)
    kmeans = MiniBatchKMeans(
        n_clusters=n_classes,
        batch_size=min(10000, len(valid_pixels)),
        n_init=3,
        random_state=42,
    )
    labels_flat = kmeans.fit_predict(features)

    # --- Step 3: Classification ---
    # Compute per-cluster spectral statistics
    cluster_stats = []
    for i in range(n_classes):
        mask = labels_flat == i
        if not np.any(mask):
            cluster_stats.append({"mean": np.zeros(bands), "std": np.zeros(bands), "count": 0})
            continue
        cluster_pixels = valid_pixels[mask]
        cluster_stats.append({
            "mean": cluster_pixels.mean(axis=0),
            "std": cluster_pixels.std(axis=0),
            "count": int(mask.sum()),
        })

    # Assign land cover class to each cluster based on spectral properties
    cluster_classes = _classify_clusters(cluster_stats, bands)

    # --- Step 4: Confidence scoring ---
    # Distance to cluster centroid (closer = higher confidence)
    distances = kmeans.transform(features)  # (n_pixels, n_clusters)
    min_distances = distances[np.arange(len(labels_flat)), labels_flat]
    # Normalize to 0-1 (closer = higher confidence)
    max_dist = np.percentile(min_distances, 95) if len(min_distances) > 0 else 1.0
    confidence_flat = np.clip(1.0 - min_distances / (max_dist + 1e-8), 0.1, 1.0)

    # --- Step 5: Vectorization ---
    # Set up CRS reprojection to WGS84 for GeoJSON output
    _reproject_poly = None
    if crs and str(crs) != "EPSG:4326":
        try:
            from pyproj import Transformer, CRS as ProjCRS
            from shapely.ops import transform as shapely_transform
            from functools import partial
            transformer = Transformer.from_crs(ProjCRS(crs), ProjCRS.from_epsg(4326), always_xy=True)
            _reproject_poly = partial(shapely_transform, transformer.transform)
        except Exception:
            pass

    # Rebuild full raster label map
    label_map = np.full(height * width, -1, dtype=np.int32)
    label_map[valid_mask] = labels_flat
    label_map = label_map.reshape(height, width)

    confidence_map = np.full(height * width, 0.0, dtype=np.float32)
    confidence_map[valid_mask] = confidence_flat
    confidence_map = confidence_map.reshape(height, width)

    # Convert raster segments to vector polygons
    features_list = []
    for geom, value in shapes(label_map.astype(np.int32), transform=transform):
        cluster_id = int(value)
        if cluster_id < 0 or cluster_id >= n_classes:
            continue

        poly = shapely_shape(geom)
        if poly.area < min_segment_pixels * abs(transform.a * transform.e):
            continue

        # Smooth blocky pixel boundaries — tight tolerance preserves accuracy
        # tolerance = 0.5 pixel keeps vertices within half-a-pixel of truth
        pixel_size = abs(transform.a)
        poly = poly.simplify(tolerance=pixel_size * 0.5, preserve_topology=True)
        if poly.is_empty or not poly.is_valid:
            # Try buffer(0) only if invalid
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

        lc = cluster_classes[cluster_id]
        # Average confidence for this segment
        seg_mask = label_map == cluster_id
        seg_confidence = float(confidence_map[seg_mask].mean())

        features_list.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "class_id": lc["id"],
                "class_name": lc["name"],
                "color": lc["color"],
                "confidence": round(seg_confidence, 3),
                "area_m2": round(poly.area, 1),
                "cluster_id": cluster_id,
                "pixel_count": int(seg_mask.sum()),
                "needs_review": seg_confidence < 0.6,
            },
        })

    # Sort by area (largest first)
    features_list.sort(key=lambda f: f["properties"]["area_m2"], reverse=True)

    # Summary statistics
    class_summary = {}
    total_area = sum(f["properties"]["area_m2"] for f in features_list)
    for f in features_list:
        name = f["properties"]["class_name"]
        if name not in class_summary:
            class_summary[name] = {"area_m2": 0, "count": 0, "avg_confidence": 0, "color": f["properties"]["color"]}
        class_summary[name]["area_m2"] += f["properties"]["area_m2"]
        class_summary[name]["count"] += 1
        class_summary[name]["avg_confidence"] += f["properties"]["confidence"]

    for name, s in class_summary.items():
        s["avg_confidence"] = round(s["avg_confidence"] / max(s["count"], 1), 3)
        s["percentage"] = round(s["area_m2"] / max(total_area, 1) * 100, 1)

    return {
        "type": "FeatureCollection",
        "features": features_list,
        "properties": {
            "total_features": len(features_list),
            "total_area_m2": round(total_area, 1),
            "n_classes": n_classes,
            "class_summary": class_summary,
            "needs_review_count": sum(1 for f in features_list if f["properties"]["needs_review"]),
            "avg_confidence": round(
                np.mean([f["properties"]["confidence"] for f in features_list]) if features_list else 0, 3
            ),
            "method": "spectral_kmeans",
            "model_version": "heuristic_v1",
        },
    }


def auto_label_vector(filepath: str, n_classes: int = 6) -> dict:
    """
    Auto-label vector features by clustering their properties.
    """
    import geopandas as gpd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    try:
        gdf = gpd.read_file(filepath)
    except Exception:
        gdf = gpd.read_file(filepath, driver="GeoJSON")
    if len(gdf) == 0:
        return {"type": "FeatureCollection", "features": [], "properties": {"total_features": 0}}

    # Select numeric columns for clustering
    numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # No numeric features — assign based on existing categorical columns
        return _label_by_categories(gdf)

    features = gdf[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_classes = min(n_classes, max(2, len(gdf) // 2))
    kmeans = KMeans(n_clusters=n_classes, n_init=3, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    distances = kmeans.transform(features_scaled)
    min_distances = distances[np.arange(len(labels)), labels]
    max_dist = np.percentile(min_distances, 95) if len(min_distances) > 0 else 1.0
    confidence = np.clip(1.0 - min_distances / (max_dist + 1e-8), 0.1, 1.0)

    # Assign colors per cluster
    colors = ["#228b22", "#f0e68c", "#808080", "#1e90ff", "#90ee90", "#d2b48c", "#5f9ea0", "#e0f0ff"]
    class_names = [f"Cluster {i+1}" for i in range(n_classes)]

    # Build GeoJSON manually to avoid geopandas serialization issues
    from shapely.geometry import mapping

    try:
        if gdf.crs and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    features_out = []
    non_geom = [c for c in gdf.columns if c != "geometry"]
    for idx in range(len(gdf)):
        row = gdf.iloc[idx]
        geom = row.geometry
        if geom is None:
            continue
        props = {c: _safe_val(row[c]) for c in non_geom}
        props["auto_class"] = class_names[labels[idx]]
        props["auto_color"] = colors[labels[idx] % len(colors)]
        props["confidence"] = round(float(confidence[idx]), 3)
        props["needs_review"] = bool(confidence[idx] < 0.6)
        features_out.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": props,
        })

    geojson: dict = {"type": "FeatureCollection", "features": features_out}
    geojson["properties"] = {
        "total_features": len(features_out),
        "n_classes": n_classes,
        "avg_confidence": round(float(confidence.mean()), 3),
        "needs_review_count": int((confidence < 0.6).sum()),
        "method": "property_kmeans",
        "model_version": "heuristic_v1",
    }
    return geojson


# --- Helpers ---


def _classify_clusters(cluster_stats: list[dict], n_bands: int) -> list[dict]:
    """Assign land cover class to each cluster based on spectral signature."""
    classes = []
    used_ids = set()

    for i, stats in enumerate(cluster_stats):
        if stats["count"] == 0:
            classes.append(LAND_COVER_CLASSES[0])
            continue

        mean = stats["mean"]

        if n_bands == 1:
            # Single band (e.g., NDVI) — classify by value ranges
            val = mean[0]
            if val > 0.6:
                lc = _get_class("Forest", used_ids)
            elif val > 0.3:
                lc = _get_class("Cropland", used_ids)
            elif val > 0.1:
                lc = _get_class("Grassland", used_ids)
            elif val > -0.1:
                lc = _get_class("Barren", used_ids)
            else:
                lc = _get_class("Water", used_ids)
        elif n_bands == 3:
            # RGB — use color heuristics
            r, g, b = mean[0], mean[1], mean[2]
            brightness = (r + g + b) / 3
            if g > r and g > b and brightness > 50:
                lc = _get_class("Forest", used_ids)
            elif brightness < 30:
                lc = _get_class("Water", used_ids)
            elif r > 150 and g > 150 and b > 150:
                lc = _get_class("Urban", used_ids)
            elif r > g:
                lc = _get_class("Barren", used_ids)
            else:
                lc = _get_class("Grassland", used_ids)
        elif n_bands >= 4:
            # Multispectral — use NDVI-like indices
            # Assume band order: B, G, R, NIR (common for Sentinel-2 / Landsat)
            nir = mean[min(3, n_bands - 1)]
            red = mean[min(2, n_bands - 1)]
            green = mean[min(1, n_bands - 1)]
            ndvi = (nir - red) / (nir + red + 1e-8)

            if ndvi > 0.5:
                lc = _get_class("Forest", used_ids)
            elif ndvi > 0.3:
                lc = _get_class("Cropland", used_ids)
            elif ndvi > 0.15:
                lc = _get_class("Grassland", used_ids)
            elif nir < green * 0.8:
                lc = _get_class("Water", used_ids)
            elif mean.mean() > np.percentile([s["mean"].mean() for s in cluster_stats if s["count"] > 0], 75):
                lc = _get_class("Urban", used_ids)
            else:
                lc = _get_class("Barren", used_ids)
        else:
            lc = LAND_COVER_CLASSES[i % len(LAND_COVER_CLASSES)]

        classes.append(lc)
    return classes


def _get_class(name: str, used_ids: set) -> dict:
    """Get a land cover class, avoiding duplicates when possible."""
    for lc in LAND_COVER_CLASSES:
        if lc["name"] == name:
            used_ids.add(lc["id"])
            return lc
    return LAND_COVER_CLASSES[0]


def _label_by_categories(gdf) -> dict:
    """Label vector features using existing categorical columns."""
    import geopandas as gpd

    cat_cols = gdf.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "geometry"]

    if not cat_cols:
        # No features to cluster — return as-is with neutral labels
        gdf = gdf.copy()
        gdf["auto_class"] = "Unclassified"
        gdf["confidence"] = 0.5
        gdf["needs_review"] = True
    else:
        col = cat_cols[0]
        unique = gdf[col].unique()
        colors = ["#228b22", "#f0e68c", "#808080", "#1e90ff", "#90ee90", "#d2b48c", "#5f9ea0", "#e0f0ff"]
        color_map = {v: colors[i % len(colors)] for i, v in enumerate(unique)}
        gdf = gdf.copy()
        gdf["auto_class"] = gdf[col].astype(str)
        gdf["auto_color"] = gdf[col].map(color_map)
        gdf["confidence"] = 0.8
        gdf["needs_review"] = False

    for c in gdf.columns:
        if gdf[c].dtype.kind in ("M", "m"):
            gdf[c] = gdf[c].astype(str)

    try:
        if gdf.crs and str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    geojson = json.loads(gdf.to_json())
    geojson["properties"] = {
        "total_features": len(gdf),
        "method": "categorical",
        "model_version": "heuristic_v1",
    }
    return geojson


def _safe_val(v: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _empty_result(bounds) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "total_features": 0,
            "error": "Insufficient valid pixels for segmentation",
        },
    }
