import type { DatasetInfo } from "../types";
import type { ImageryLayer } from "./MapView";

interface DatasetDetailProps {
  dataset: DatasetInfo;
  onClose: () => void;
  onAddImageryLayer?: (l: ImageryLayer) => void;
}

export function DatasetDetail({ dataset, onClose, onAddImageryLayer }: DatasetDetailProps) {
  const d = dataset;

  const addAsLayer = () => {
    if (!onAddImageryLayer) return;
    // Path-encode the filename so quirks like spaces in sample datasets still
    // resolve. The backend mounts /api at the root, so the tile route is
    // /api/datasets/{filename}/tiles/{z}/{x}/{y}.png.
    const safeName = encodeURIComponent(d.filename);
    onAddImageryLayer({
      id: `upload-${d.filename}`,
      tileUrl: `${window.location.origin}/api/datasets/${safeName}/tiles/{z}/{x}/{y}.png`,
      label: `${d.filename} · uploaded`,
      opacity: 0.85,
    });
  };

  return (
    <div className="bg-geo-surface border border-geo-border rounded-lg p-3 mb-3">
      <div className="flex justify-between items-center mb-2">
        <h4 className="m-0 text-[13px] font-semibold text-geo-text">{d.filename}</h4>
        <button
          onClick={onClose}
          className="bg-transparent border-none text-geo-dim cursor-pointer text-base hover:text-geo-text"
        >
          x
        </button>
      </div>

      <Row label="Format" value={d.format.toUpperCase()} />
      <Row label="Size" value={formatBytes(d.size_bytes)} />

      {/* Add-as-layer affordance — raster uploads can be served as XYZ tiles
          via /api/datasets/.../tiles/z/x/y.png, letting the user drop them
          on the map alongside an OlmoEarth inference for side-by-side compare. */}
      {onAddImageryLayer && d.raster && (
        <div className="mt-2">
          <button
            type="button"
            onClick={addAsLayer}
            data-testid="add-raster-as-layer"
            className="w-full px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider bg-geo-accent/10 text-geo-accent hover:bg-geo-accent hover:text-white border border-geo-accent/40 rounded transition-colors"
          >
            + Add as map layer
          </button>
        </div>
      )}

      {/* Raster details */}
      {d.raster && (
        <>
          <Divider />
          <Row label="Dimensions" value={`${d.raster.width} x ${d.raster.height}`} />
          <Row label="Bands" value={String(d.raster.bands)} />
          <Row label="Dtype" value={d.raster.dtype} />
          {d.raster.nodata !== null && <Row label="NoData" value={String(d.raster.nodata)} />}
          {d.raster.resolution && (
            <Row label="Resolution" value={`${d.raster.resolution[0].toFixed(1)} x ${d.raster.resolution[1].toFixed(1)}`} />
          )}
          {d.raster.crs?.epsg && <Row label="CRS" value={`EPSG:${d.raster.crs.epsg}`} />}
          {d.raster.band_names && d.raster.band_names.length > 0 && (
            <Row label="Band names" value={d.raster.band_names.join(", ")} />
          )}
        </>
      )}

      {/* Vector details */}
      {d.vector && (
        <>
          <Divider />
          <Row label="Geometry" value={d.vector.geometry_type} />
          <Row label="Features" value={d.vector.feature_count.toLocaleString()} />
          {d.vector.crs?.epsg && <Row label="CRS" value={`EPSG:${d.vector.crs.epsg}`} />}
          <Row label="Columns" value={d.vector.columns.slice(0, 8).join(", ") + (d.vector.columns.length > 8 ? ` (+${d.vector.columns.length - 8})` : "")} />
          {d.vector.sample_properties && (
            <div className="mt-1.5">
              <div className="text-[10px] text-geo-dim mb-1">Sample row:</div>
              {Object.entries(d.vector.sample_properties).slice(0, 6).map(([k, v]) => (
                <Row key={k} label={k} value={String(v)} />
              ))}
            </div>
          )}
        </>
      )}

      {/* Point cloud details */}
      {d.point_cloud && (
        <>
          <Divider />
          <Row label="Points" value={d.point_cloud.point_count.toLocaleString()} />
          <Row label="Color" value={d.point_cloud.has_color ? "Yes (RGB)" : "No"} />
          <Row label="Intensity" value={d.point_cloud.has_intensity ? "Yes" : "No"} />
          <Row label="Classification" value={d.point_cloud.has_classification ? "Yes" : "No"} />
        </>
      )}

      {/* Multidim details */}
      {d.multidim && (
        <>
          <Divider />
          <Row label="Variables" value={d.multidim.variables.slice(0, 5).join(", ") + (d.multidim.variables.length > 5 ? ` (+${d.multidim.variables.length - 5})` : "")} />
          <Row label="Dimensions" value={Object.entries(d.multidim.dimensions).map(([k, v]) => `${k}:${v}`).join(", ")} />
          <Row label="Coords" value={d.multidim.coords.join(", ")} />
          {d.multidim.time_range && (
            <Row label="Time" value={`${d.multidim.time_range[0]} to ${d.multidim.time_range[1]}`} />
          )}
        </>
      )}

      {/* BBox */}
      {(d.raster?.bbox || d.vector?.bbox || d.point_cloud?.bbox || d.multidim?.bbox) && (
        <>
          <Divider />
          <BBoxDisplay bbox={(d.raster?.bbox || d.vector?.bbox || d.point_cloud?.bbox || d.multidim?.bbox)!} />
        </>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-[11px] py-0.5">
      <span className="text-geo-muted">{label}</span>
      <span className="text-geo-text font-mono text-right max-w-[180px] overflow-hidden text-ellipsis whitespace-nowrap">
        {value}
      </span>
    </div>
  );
}

function Divider() {
  return <div className="border-t border-geo-border my-1.5" />;
}

function BBoxDisplay({ bbox }: { bbox: { west: number; south: number; east: number; north: number } }) {
  return (
    <div className="text-[10px] font-mono text-geo-dim">
      [{bbox.west.toFixed(4)}, {bbox.south.toFixed(4)}] to [{bbox.east.toFixed(4)}, {bbox.north.toFixed(4)}]
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
