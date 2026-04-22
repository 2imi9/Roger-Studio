"""
SamGeo (Segment Anything Geospatial) labeler — pixel-accurate segmentation.

Uses Meta's SAM via segment-geospatial wrapper. Produces much finer boundaries
than TIPSv2 because SAM is a pure segmentation model trained on 11M images
with 1.1B masks.

Pipeline:
  1. Convert GeoTIFF → RGB uint8
  2. Run SamGeo.generate() for automatic mask generation
  3. Get per-segment polygons (already reprojected to WGS84)
  4. Classify each segment by spectral centroid using K-means
  5. Apply user's label classes to the clusters

Model: sam_vit_h (~2.4GB, auto-downloaded on first use)
GPU: optional, runs on CPU in ~30-60s per tile
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cache directory for SAM checkpoint
_SAM_CHECKPOINT_DIR = Path.home() / ".cache" / "samgeo"
_SAM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def auto_label_geotiff_samgeo(
    filepath: str,
    classes: list[dict] | None = None,
    n_clusters: int | None = None,
) -> dict:
    """
    Auto-label a GeoTIFF using SAM pixel-accurate segmentation.

    Args:
        filepath: Path to the GeoTIFF file
        classes: Optional label classes (name, prompt, color). If provided,
                 clusters are mapped to class names.
        n_clusters: Number of classes to group segments into. Defaults to
                    len(classes) or 6.

    Returns:
        GeoJSON FeatureCollection with polygons + confidence scores.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import shape as shapely_shape, mapping
    from sklearn.cluster import KMeans

    if classes is None:
        from app.services.tipsv2_labeler import DEFAULT_CLASSES
        classes = DEFAULT_CLASSES

    n_clusters = n_clusters or len(classes)

    # Load raster and convert to RGB uint8 for SAM
    with rasterio.open(filepath) as src:
        data = src.read()
        transform = src.transform
        src_crs = src.crs
        bands, height, width = data.shape

    if bands >= 3:
        rgb = data[:3].transpose(1, 2, 0)
    elif bands == 1:
        band = data[0]
        bmin, bmax = (
            np.nanpercentile(band[band != 0], [2, 98])
            if np.any(band != 0)
            else (0, 1)
        )
        normalized = np.clip(
            (band - bmin) / (bmax - bmin + 1e-8) * 255, 0, 255
        ).astype(np.uint8)
        rgb = np.stack([normalized] * 3, axis=-1)
    else:
        rgb = np.stack([data[0]] * 3, axis=-1)

    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            p2, p98 = (
                np.percentile(rgb[rgb > 0], [2, 98])
                if np.any(rgb > 0)
                else (0, 255)
            )
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(
                np.uint8
            )

    logger.info(f"SamGeo: input {width}x{height}, {bands} bands → RGB uint8")

    # Write RGB to a temp GeoTIFF (SamGeo needs a georeferenced file)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=src_crs,
            transform=transform,
        ) as dst:
            dst.write(rgb.transpose(2, 0, 1))

        # Run SamGeo automatic mask generation
        from samgeo import SamGeo

        logger.info("SamGeo: loading SAM model (may download ~2.4GB on first run)...")
        sam = SamGeo(
            model_type="vit_h",
            checkpoint=str(_SAM_CHECKPOINT_DIR / "sam_vit_h_4b8939.pth"),
            sam_kwargs=None,
        )

        logger.info("SamGeo: generating masks (this takes 30-60s on CPU)...")
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as mask_tmp:
            mask_path = mask_tmp.name

        try:
            sam.generate(tmp_path, output=mask_path, foreground=True, unique=True)

            # Convert mask raster to polygons
            vector_path = mask_path.replace(".tif", "_vector.geojson")
            sam.tiff_to_vector(mask_path, vector_path)

            import geopandas as gpd

            gdf = gpd.read_file(vector_path)
            logger.info(f"SamGeo: got {len(gdf)} segments")

            if len(gdf) == 0:
                return _empty_samgeo_result()

            # Reproject to WGS84 for frontend
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Compute spectral centroid per segment for clustering
            # (we need segment→pixel mapping, so use the mask raster again)
            with rasterio.open(mask_path) as msrc:
                mask_data = msrc.read(1)

            segment_ids = np.unique(mask_data)
            segment_ids = segment_ids[segment_ids > 0]

            if len(segment_ids) < n_clusters:
                n_clusters = max(2, len(segment_ids))

            # Get mean RGB per segment
            seg_features = []
            seg_order = []
            for sid in segment_ids:
                mask = mask_data == sid
                if mask.sum() < 5:
                    continue
                mean_rgb = [rgb[:, :, c][mask].mean() for c in range(3)]
                seg_features.append(mean_rgb)
                seg_order.append(int(sid))

            if len(seg_features) == 0:
                return _empty_samgeo_result()

            # Cluster segments into classes
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(seg_features)),
                n_init=5,
                random_state=42,
            )
            seg_labels = kmeans.fit_predict(np.array(seg_features))

            # Build segment_id → class_name mapping
            seg_to_class = {}
            for sid, cluster_idx in zip(seg_order, seg_labels):
                cls = classes[cluster_idx % len(classes)]
                seg_to_class[sid] = cls

            # Assign class properties to each GeoDataFrame feature
            features_list = []
            total_area_deg2 = 0.0
            class_summary: dict[str, dict[str, Any]] = {}

            for idx, row in gdf.iterrows():
                sid = int(row.get("value", row.get("id", idx + 1)))
                cls = seg_to_class.get(sid, classes[0])

                poly = row.geometry
                if poly is None or poly.is_empty:
                    continue

                # Tight simplify for clean boundaries (preserve accuracy)
                poly = poly.simplify(tolerance=0.00001, preserve_topology=True)
                if not poly.is_valid or poly.is_empty:
                    continue

                area_deg2 = poly.area
                total_area_deg2 += area_deg2

                # Approx area_m2 at mid-latitude
                lat = poly.centroid.y
                m_per_deg_lat = 111320
                m_per_deg_lng = 111320 * abs(np.cos(np.radians(lat)))
                area_m2 = area_deg2 * m_per_deg_lat * m_per_deg_lng

                props = {
                    "class_id": classes.index(cls) + 1 if cls in classes else 1,
                    "class_name": cls["name"],
                    "color": cls["color"],
                    "confidence": 0.85,  # SAM is strong; fixed high confidence
                    "area_m2": round(area_m2, 1),
                    "segment_id": sid,
                    "needs_review": False,
                }

                features_list.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": props,
                    }
                )

                name = cls["name"]
                if name not in class_summary:
                    class_summary[name] = {
                        "area_m2": 0.0,
                        "count": 0,
                        "color": cls["color"],
                    }
                class_summary[name]["area_m2"] += area_m2
                class_summary[name]["count"] += 1

            # Compute percentages
            total_area_m2 = sum(s["area_m2"] for s in class_summary.values())
            for s in class_summary.values():
                s["percentage"] = round(
                    s["area_m2"] / max(total_area_m2, 1) * 100, 1
                )

            features_list.sort(
                key=lambda f: f["properties"]["area_m2"], reverse=True
            )

            return {
                "type": "FeatureCollection",
                "features": features_list,
                "properties": {
                    "total_features": len(features_list),
                    "total_area_m2": round(total_area_m2, 1),
                    "n_classes": len(class_summary),
                    "class_summary": class_summary,
                    "needs_review_count": 0,
                    "avg_confidence": 0.85,
                    "method": "samgeo_sam",
                    "model_version": "sam_vit_h",
                },
            }

        finally:
            for p in (mask_path, mask_path.replace(".tif", "_vector.geojson")):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _empty_samgeo_result() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "total_features": 0,
            "method": "samgeo_sam",
            "model_version": "sam_vit_h",
            "error": "No segments produced by SAM",
        },
    }
