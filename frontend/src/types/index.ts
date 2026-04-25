export interface BBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface PolygonCoords {
  type: "Polygon";
  coordinates: number[][][];
}

export type AreaSelection = BBox | PolygonCoords;

export interface LandCoverClass {
  id: number;
  name: string;
  color: string;
  percentage: number;
}

export interface AnalysisRequest {
  area: AreaSelection;
  data_sources?: string[];
}

export interface OlmoEarthDatasetMeta {
  repo_id: string;
  description?: string;
  license?: string;
  task?: string;
  coverage?: string;
  size?: string;
  modalities?: string[];
  format?: string;
  docs?: string;
}

export interface OlmoEarthModelMeta {
  repo_id: string;
  size_tier?: "smallest" | "small" | "medium" | "largest";
  type?: "encoder";
  base?: string;
  task?: string;
  reason?: string;
}

export interface OlmoEarthSummary {
  highlight: {
    project_coverage: { repo_id: string; task?: string }[];
    recommended_model: OlmoEarthModelMeta;
    notes: string[];
  };
  datasets: OlmoEarthDatasetMeta[];
  models: OlmoEarthModelMeta[];
}

export interface AnalysisResult {
  land_cover: LandCoverClass[];
  bbox: BBox;
  area_km2: number;
  timestamp: string;
  suitability_scores?: Record<string, number>;
  olmoearth?: OlmoEarthSummary | null;
}

export interface EnvDataRequest {
  bbox: BBox;
  variables: string[];
  time?: string;
}

export interface EnvDataResult {
  wind?: { speed: number; direction: number };
  temperature?: number;
  solar_irradiance?: number;
  humidity?: number;
  /**
   * Populated when the backend's upstream Open-Meteo call failed
   * (timeout, unreachable, non-200 response, JSON parse error). Lets
   * the UI distinguish "ERA5/Open-Meteo outage" from "this variable
   * isn't available for this location" — both used to surface as plain
   * ``undefined`` with a 200 OK and the UX couldn't tell them apart.
   * Prefix identifies the failure class:
   *   ``open_meteo_timeout`` / ``open_meteo_unreachable`` /
   *   ``open_meteo_http_{status}`` / ``open_meteo_parse_error``
   */
  error?: string;
}

export interface ElevationResult {
  lats: number[];
  lons: number[];
  /**
   * Row-major grid of elevation samples in meters. Cells may be
   * ``null`` where Open-Meteo SRTM had no coverage (ocean interior,
   * extreme polar latitudes) — consumers MUST guard against null
   * before doing arithmetic, or the cell will coerce to 0 silently.
   * The type was previously ``number[][]`` and would crash at runtime
   * the first time a null cell landed (see architecture audit).
   */
  elevations: (number | null)[][];
  stats: { min: number; max: number; mean: number; range: number };
  resolution: number;
  bbox: BBox;
}

// "3d" was dropped — the Cesium-based globe was a heavy WebGL dependency
// that pegged the GPU even when idle and wasn't carrying scientific value
// the Map tab couldn't already provide. Removed cleanly: type, tab pill,
// App-level render branch, icon, package.json dep.
export type ViewMode = "map" | "analysis" | "olmoearth" | "gemma";
export type BasemapStyle = "osm" | "satellite" | "dark";

// --- Dataset upload types ---

export type DataFormat =
  | "geotiff"
  | "cog"
  | "netcdf"
  | "zarr"
  | "geopackage"
  | "geojson"
  | "shapefile"
  | "las"
  | "laz"
  | "parquet"
  | "geoparquet"
  | "csv"
  | "unknown";

export interface RasterInfo {
  width: number;
  height: number;
  bands: number;
  dtype: string;
  nodata: number | null;
  crs: { epsg: number | null; wkt: string | null } | null;
  bbox: BBox | null;
  resolution: [number, number] | null;
  band_names: string[] | null;
}

export interface VectorInfo {
  geometry_type: string;
  feature_count: number;
  crs: { epsg: number | null; wkt: string | null } | null;
  bbox: BBox | null;
  columns: string[];
  sample_properties: Record<string, unknown> | null;
}

export interface PointCloudInfo {
  point_count: number;
  bbox: BBox | null;
  has_color: boolean;
  has_intensity: boolean;
  has_classification: boolean;
  point_format: number | null;
}

export interface MultidimInfo {
  dimensions: Record<string, number>;
  variables: string[];
  coords: string[];
  bbox: BBox | null;
  time_range: [string, string] | null;
  attrs: Record<string, unknown> | null;
}

export interface DatasetInfo {
  filename: string;
  format: DataFormat;
  size_bytes: number;
  raster: RasterInfo | null;
  vector: VectorInfo | null;
  point_cloud: PointCloudInfo | null;
  multidim: MultidimInfo | null;
  preview_geojson: GeoJSON.Feature | GeoJSON.FeatureCollection | null;
}
