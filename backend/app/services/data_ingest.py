"""
Data ingest service — detect format, extract metadata, generate map preview.

Supported formats:
  Raster:  GeoTIFF, COG (Cloud Optimized GeoTIFF)
  Multidim: NetCDF, Zarr
  Vector:  GeoJSON, GeoPackage, Shapefile, GeoParquet, CSV
  Point cloud: LAS, LAZ
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from app.models.schemas import (
    BBox,
    CRS,
    DataFormat,
    DatasetInfo,
    MultidimInfo,
    PointCloudInfo,
    RasterInfo,
    VectorInfo,
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "geoenv_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Extension → format mapping
EXT_MAP: dict[str, DataFormat] = {
    ".tif": DataFormat.GEOTIFF,
    ".tiff": DataFormat.GEOTIFF,
    ".geotiff": DataFormat.GEOTIFF,
    ".nc": DataFormat.NETCDF,
    ".nc4": DataFormat.NETCDF,
    ".netcdf": DataFormat.NETCDF,
    ".zarr": DataFormat.ZARR,
    ".gpkg": DataFormat.GEOPACKAGE,
    ".geojson": DataFormat.GEOJSON,
    ".json": DataFormat.GEOJSON,
    ".shp": DataFormat.SHAPEFILE,
    ".zip": DataFormat.SHAPEFILE,  # zipped shapefile
    ".las": DataFormat.LAS,
    ".laz": DataFormat.LAZ,
    ".parquet": DataFormat.PARQUET,
    ".geoparquet": DataFormat.GEOPARQUET,
    ".csv": DataFormat.CSV,
}


def detect_format(filename: str, content_peek: bytes | None = None) -> DataFormat:
    ext = Path(filename).suffix.lower()
    fmt = EXT_MAP.get(ext, DataFormat.UNKNOWN)

    # Check if .zip contains shapefile components
    if ext == ".zip" and content_peek:
        try:
            import io
            with zipfile.ZipFile(io.BytesIO(content_peek)) as zf:
                names = zf.namelist()
                if any(n.endswith(".shp") for n in names):
                    return DataFormat.SHAPEFILE
        except zipfile.BadZipFile:
            pass

    # Check if .parquet is actually geoparquet (has geo metadata)
    if fmt == DataFormat.PARQUET and content_peek:
        try:
            import pyarrow.parquet as pq
            import io
            pf = pq.ParquetFile(io.BytesIO(content_peek))
            meta = pf.schema_arrow.metadata or {}
            if b"geo" in meta:
                return DataFormat.GEOPARQUET
        except Exception:
            pass

    return fmt


def sniff_format_by_magic(data: bytes) -> set[DataFormat]:
    """Return the set of ``DataFormat`` values whose magic bytes match ``data``.

    The audit caught this: ``detect_format`` trusted the filename extension.
    A user could rename ``evil.geojson`` to ``evil.tif`` and upload — the
    backend would accept it as GeoTIFF, later crash in ``inspect_file``, and
    leave the file on disk with the wrong stated format. Worse, a tif could
    be renamed .geojson and fed to downstream vector tooling. Sniffing
    magic bytes at ingest closes the gap.

    A format may map to multiple enums (Parquet/GeoParquet share the PAR1
    header; .las and .laz both start with ``LASF``) — the caller compares
    the extension-derived format to this set with ``in`` membership. An
    empty return means either the format has no canonical magic (CSV —
    it's plaintext with no signature) or the bytes are genuinely unknown;
    the caller decides whether that's fatal per-extension.
    """
    if len(data) < 4:
        return set()
    matches: set[DataFormat] = set()

    head4 = data[:4]
    head8 = data[:8]
    head16 = data[:16]

    if head4 in (b"CDF\x01", b"CDF\x02"):
        matches.add(DataFormat.NETCDF)
    elif head8 == b"\x89HDF\r\n\x1a\n":
        # HDF5 underpins NetCDF-4. Treat it as NetCDF for upload purposes
        # — anyone uploading raw HDF5 can stage via a .nc renaming.
        matches.add(DataFormat.NETCDF)
    elif head4 in (b"II*\x00", b"MM\x00*"):
        # TIFF (little-endian classic + big-endian classic). Covers GeoTIFF
        # and Cloud-Optimized GeoTIFF — they share the same file header.
        matches.add(DataFormat.GEOTIFF)
        matches.add(DataFormat.COG)
    elif head4 == b"PK\x03\x04":
        # ZIP: the upload flow accepts zipped shapefiles. Zarr stores are
        # directories (never uploaded as single files here), so a bare
        # .zip upload is treated as a candidate shapefile — the caller's
        # detect_format path cracks the zip and confirms a .shp inside.
        matches.add(DataFormat.SHAPEFILE)
    elif head4 == b"PAR1":
        matches.add(DataFormat.PARQUET)
        matches.add(DataFormat.GEOPARQUET)
    elif head16 == b"SQLite format 3\x00":
        matches.add(DataFormat.GEOPACKAGE)
    elif head4 == b"LASF":
        matches.add(DataFormat.LAS)
        matches.add(DataFormat.LAZ)
    else:
        # JSON/GeoJSON: strip leading whitespace and look for '{' or '['.
        # Anything else — CSV, unknown binary — gets an empty set so the
        # caller can allow extensions without a magic signature.
        stripped = data.lstrip(b" \t\r\n")
        if stripped and stripped[:1] in (b"{", b"["):
            matches.add(DataFormat.GEOJSON)

    return matches


# Extensions for which magic-byte sniffing returns nothing (plaintext / no
# canonical signature) — uploads with these extensions skip the mismatch
# check because there's no reliable way to tell content-vs-claim apart.
_NO_MAGIC_EXTS: set[str] = {".csv"}


def _extract_crs(crs_obj: Any) -> CRS | None:
    if crs_obj is None:
        return None
    try:
        from pyproj import CRS as ProjCRS
        if hasattr(crs_obj, "to_epsg"):
            epsg = crs_obj.to_epsg()
            return CRS(epsg=epsg, wkt=str(crs_obj))
        proj_crs = ProjCRS(crs_obj)
        return CRS(epsg=proj_crs.to_epsg(), wkt=proj_crs.to_wkt())
    except Exception:
        return CRS(wkt=str(crs_obj))


def _reproject_bbox_to_wgs84(bbox: BBox, crs_obj: Any) -> BBox:
    """Reproject bounding box to EPSG:4326 (WGS84 degrees) for map/globe display."""
    if crs_obj is None:
        return bbox
    try:
        from pyproj import Transformer, CRS as ProjCRS
        src_crs = ProjCRS(crs_obj) if not isinstance(crs_obj, ProjCRS) else crs_obj
        epsg = src_crs.to_epsg()
        if epsg == 4326:
            return bbox  # Already WGS84
        transformer = Transformer.from_crs(src_crs, ProjCRS.from_epsg(4326), always_xy=True)
        x1, y1 = transformer.transform(bbox.west, bbox.south)
        x2, y2 = transformer.transform(bbox.east, bbox.north)
        # Ensure south <= north, west <= east after reprojection
        return BBox(
            west=min(x1, x2), south=min(y1, y2),
            east=max(x1, x2), north=max(y1, y2),
        )
    except Exception:
        return bbox  # Can't reproject — return as-is


# --- Raster (GeoTIFF / COG) ---


def inspect_raster(filepath: str) -> DatasetInfo:
    import rasterio

    with rasterio.open(filepath) as src:
        bounds = src.bounds
        raw_bbox = BBox(west=bounds.left, south=bounds.bottom, east=bounds.right, north=bounds.top)
        # Always reproject to WGS84 for map/globe display
        bbox = _reproject_bbox_to_wgs84(raw_bbox, src.crs)

        # Check if COG
        is_cog = False
        if hasattr(src, "overviews") and src.overviews(1):
            is_cog = True

        band_names = [src.descriptions[i] or f"Band {i+1}" for i in range(src.count)]

        info = RasterInfo(
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            crs=_extract_crs(src.crs),
            bbox=bbox,
            resolution=(src.res[0], src.res[1]),
            band_names=band_names,
        )

    return DatasetInfo(
        filename=Path(filepath).name,
        format=DataFormat.COG if is_cog else DataFormat.GEOTIFF,
        size_bytes=os.path.getsize(filepath),
        raster=info,
        preview_geojson=_bbox_to_geojson(bbox),
    )


# --- Vector (GeoJSON, GeoPackage, Shapefile, GeoParquet) ---


def inspect_vector(filepath: str, fmt: DataFormat) -> DatasetInfo:
    import geopandas as gpd

    if fmt == DataFormat.GEOJSON:
        gdf = gpd.read_file(filepath, driver="GeoJSON")
    elif fmt == DataFormat.GEOPACKAGE:
        gdf = gpd.read_file(filepath, driver="GPKG")
    elif fmt == DataFormat.SHAPEFILE:
        if filepath.endswith(".zip"):
            gdf = gpd.read_file(f"zip://{filepath}")
        else:
            gdf = gpd.read_file(filepath)
    elif fmt in (DataFormat.PARQUET, DataFormat.GEOPARQUET):
        gdf = gpd.read_parquet(filepath)
    elif fmt == DataFormat.CSV:
        import pandas as pd
        df = pd.read_csv(filepath)
        # Try to find lat/lon columns
        lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude", "lat_dd", "y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "long_dd", "x")), None)
        if lat_col and lon_col:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(df)
    else:
        gdf = gpd.read_file(filepath)

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    raw_bbox = BBox(west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3])
    bbox = _reproject_bbox_to_wgs84(raw_bbox, gdf.crs)

    geom_type = gdf.geom_type.iloc[0] if len(gdf) > 0 and gdf.geometry is not None else "Unknown"
    non_geom_cols = [c for c in gdf.columns if c != "geometry"]

    # Sample properties for preview
    sample = None
    if len(gdf) > 0:
        row = gdf.iloc[0]
        sample = {}
        for c in non_geom_cols[:10]:
            try:
                sample[c] = _safe_serialize(row[c])
            except Exception:
                sample[c] = str(row[c])[:100]

    # Generate lightweight GeoJSON preview (limit features)
    preview = None
    if len(gdf) > 0:
        preview_gdf = gdf.head(500).copy()
        if preview_gdf.crs and str(preview_gdf.crs) != "EPSG:4326":
            preview_gdf = preview_gdf.to_crs("EPSG:4326")
        # Convert non-serializable columns to strings
        for col in preview_gdf.columns:
            if col == "geometry":
                continue
            dtype_kind = preview_gdf[col].dtype.kind
            if dtype_kind in ("M", "m"):  # datetime or timedelta
                preview_gdf[col] = preview_gdf[col].astype(str)
            elif dtype_kind == "O":  # object — may contain arrays, dicts, etc.
                preview_gdf[col] = preview_gdf[col].apply(
                    lambda v: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                )
        try:
            preview = json.loads(preview_gdf.to_json())
        except (TypeError, ValueError):
            # Fallback: drop non-geometry columns and just show shapes
            preview = json.loads(preview_gdf[["geometry"]].to_json())

    info = VectorInfo(
        geometry_type=geom_type,
        feature_count=len(gdf),
        crs=_extract_crs(gdf.crs),
        bbox=bbox,
        columns=non_geom_cols,
        sample_properties=sample,
    )

    return DatasetInfo(
        filename=Path(filepath).name,
        format=fmt,
        size_bytes=os.path.getsize(filepath),
        vector=info,
        preview_geojson=preview,
    )


# --- Multidimensional (NetCDF, Zarr) ---


def inspect_multidim(filepath: str, fmt: DataFormat) -> DatasetInfo:
    import xarray as xr

    if fmt == DataFormat.ZARR:
        ds = xr.open_zarr(filepath)
    else:
        ds = xr.open_dataset(filepath)

    dims = dict(ds.dims)
    variables = [str(v) for v in ds.data_vars]
    coords = [str(c) for c in ds.coords]

    # Try to extract spatial bounds
    bbox = None
    for lat_name in ("lat", "latitude", "y"):
        for lon_name in ("lon", "longitude", "x"):
            if lat_name in ds.coords and lon_name in ds.coords:
                lats = ds.coords[lat_name].values
                lons = ds.coords[lon_name].values
                bbox = BBox(
                    west=float(lons.min()),
                    south=float(lats.min()),
                    east=float(lons.max()),
                    north=float(lats.max()),
                )
                break
        if bbox:
            break

    # Time range
    time_range = None
    for t_name in ("time", "t", "datetime"):
        if t_name in ds.coords:
            times = ds.coords[t_name].values
            time_range = (str(times[0]), str(times[-1]))
            break

    # Global attributes
    attrs = {k: _safe_serialize(v) for k, v in list(ds.attrs.items())[:20]}

    ds.close()

    info = MultidimInfo(
        dimensions=dims,
        variables=variables,
        coords=coords,
        bbox=bbox,
        time_range=time_range,
        attrs=attrs,
    )

    return DatasetInfo(
        filename=Path(filepath).name,
        format=fmt,
        size_bytes=_dir_size(filepath) if os.path.isdir(filepath) else os.path.getsize(filepath),
        multidim=info,
        preview_geojson=_bbox_to_geojson(bbox) if bbox else None,
    )


# --- Point Cloud (LAS/LAZ) ---


def inspect_point_cloud(filepath: str) -> DatasetInfo:
    import laspy

    with laspy.open(filepath) as f:
        header = f.header
        point_count = header.point_count
        bbox = BBox(
            west=header.mins[0],
            south=header.mins[1],
            east=header.maxs[0],
            north=header.maxs[1],
        )
        point_format = header.point_format.id

        # Check available dimensions
        dim_names = [d.name for d in header.point_format.dimensions]
        has_color = "red" in dim_names and "green" in dim_names
        has_intensity = "intensity" in dim_names
        has_classification = "classification" in dim_names

    fmt = DataFormat.LAZ if filepath.lower().endswith(".laz") else DataFormat.LAS

    info = PointCloudInfo(
        point_count=point_count,
        bbox=bbox,
        has_color=has_color,
        has_intensity=has_intensity,
        has_classification=has_classification,
        point_format=point_format,
    )

    return DatasetInfo(
        filename=Path(filepath).name,
        format=fmt,
        size_bytes=os.path.getsize(filepath),
        point_cloud=info,
        preview_geojson=_bbox_to_geojson(bbox),
    )


# --- Dispatch ---


def inspect_file(filepath: str, filename: str | None = None) -> DatasetInfo:
    """Inspect any supported geoscience file and return structured metadata."""
    name = filename or Path(filepath).name
    fmt = detect_format(name)

    if fmt in (DataFormat.GEOTIFF, DataFormat.COG):
        return inspect_raster(filepath)
    elif fmt in (DataFormat.GEOJSON, DataFormat.GEOPACKAGE, DataFormat.SHAPEFILE,
                 DataFormat.PARQUET, DataFormat.GEOPARQUET, DataFormat.CSV):
        return inspect_vector(filepath, fmt)
    elif fmt in (DataFormat.NETCDF, DataFormat.ZARR):
        return inspect_multidim(filepath, fmt)
    elif fmt in (DataFormat.LAS, DataFormat.LAZ):
        return inspect_point_cloud(filepath)
    else:
        return DatasetInfo(
            filename=name,
            format=DataFormat.UNKNOWN,
            size_bytes=os.path.getsize(filepath),
        )


# --- Helpers ---


def _bbox_to_geojson(bbox: BBox) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox.west, bbox.north],
                [bbox.east, bbox.north],
                [bbox.east, bbox.south],
                [bbox.west, bbox.south],
                [bbox.west, bbox.north],
            ]],
        },
    }


def _safe_serialize(val: Any) -> Any:
    """Convert numpy/pandas/datetime types to JSON-safe Python types."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    # pandas Timestamp, Timedelta, etc.
    try:
        import pandas as pd
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            return str(val)
        if isinstance(val, pd.Series):
            return val.tolist()
    except ImportError:
        pass
    return str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total
