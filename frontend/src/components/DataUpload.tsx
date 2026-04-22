import { useCallback, useRef, useState } from "react";
import type { DatasetInfo } from "../types";
import { uploadFile, deleteDataset } from "../api/client";

const FORMAT_LABELS: Record<string, { label: string; color: string }> = {
  geotiff: { label: "GeoTIFF", color: "#22c55e" },
  cog: { label: "COG", color: "#22c55e" },
  netcdf: { label: "NetCDF", color: "#3b82f6" },
  zarr: { label: "Zarr", color: "#3b82f6" },
  geopackage: { label: "GPKG", color: "#f59e0b" },
  geojson: { label: "GeoJSON", color: "#f59e0b" },
  shapefile: { label: "Shapefile", color: "#f59e0b" },
  las: { label: "LAS", color: "#a855f7" },
  laz: { label: "LAZ", color: "#a855f7" },
  parquet: { label: "Parquet", color: "#ec4899" },
  geoparquet: { label: "GeoParquet", color: "#ec4899" },
  csv: { label: "CSV", color: "#6b7280" },
};

const ACCEPTED = ".tif,.tiff,.nc,.nc4,.zarr,.gpkg,.geojson,.json,.shp,.zip,.las,.laz,.parquet,.csv";

interface DataUploadProps {
  datasets: DatasetInfo[];
  onUpload: (ds: DatasetInfo) => void;
  onDelete: (filename: string) => void;
  onSelect: (ds: DatasetInfo) => void;
}

export function DataUpload({ datasets, onUpload, onDelete, onSelect }: DataUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setUploading(true);
      setError(null);
      try {
        for (const file of Array.from(files)) {
          const ds = await uploadFile(file);
          onUpload(ds);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleRemove = useCallback(
    async (filename: string) => {
      // Auto-label overlays start with "auto-label-" — not in DB, just remove locally
      if (filename.startsWith("auto-label-")) {
        onDelete(filename);
        return;
      }
      try {
        await deleteDataset(filename);
      } catch {
        // API delete failed — still remove locally
      }
      onDelete(filename);
    },
    [onDelete]
  );

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const hasAutoLabels = datasets.some((d) => d.filename.startsWith("auto-label-"));

  const handleClearAllLabels = () => {
    datasets.forEach((d) => {
      if (d.filename.startsWith("auto-label-")) {
        onDelete(d.filename);
      }
    });
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <h3 className="m-0 text-[13px] text-geo-muted font-medium">
          Data Layers
        </h3>
        {hasAutoLabels && (
          <button
            onClick={handleClearAllLabels}
            className="text-[10px] bg-geo-danger/20 text-geo-danger rounded px-2 py-0.5 border-none cursor-pointer hover:bg-geo-danger/30 transition-colors"
          >
            Clear Labels
          </button>
        )}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors mb-2.5 ${
          dragOver
            ? "border-geo-accent bg-geo-accent/10"
            : "border-geo-border hover:border-geo-accent"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="text-xs text-geo-muted">
          {uploading ? "Uploading..." : "Drop files or click to upload"}
        </div>
        <div className="text-[10px] text-geo-dim mt-1">
          GeoTIFF, NetCDF, Zarr, GPKG, GeoJSON, Shapefile(.zip), LAS/LAZ, Parquet, CSV
        </div>
      </div>

      {error && (
        <div className="bg-red-900 text-red-300 rounded-lg p-2 text-xs mb-2">
          {error}
        </div>
      )}

      {/* Dataset list */}
      {datasets.map((ds) => {
        const fmt = FORMAT_LABELS[ds.format] || { label: ds.format, color: "#6b7280" };
        const summary = _datasetSummary(ds);
        return (
          <div
            key={ds.filename}
            onClick={() => onSelect(ds)}
            className="bg-geo-surface border-l-[3px] rounded-lg p-2 mb-1.5 cursor-pointer hover:bg-geo-elevated transition-colors"
            style={{ borderLeftColor: fmt.color }}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="text-[9px] font-bold py-0.5 px-1.5 rounded uppercase tracking-wider"
                style={{ background: fmt.color + "22", color: fmt.color }}
              >
                {fmt.label}
              </span>
              <span className="text-xs flex-1 overflow-hidden text-ellipsis whitespace-nowrap text-geo-text">
                {ds.filename}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); handleRemove(ds.filename); }}
                className="bg-transparent border-none text-geo-dim cursor-pointer text-sm px-0.5 leading-none hover:text-geo-danger"
                title="Remove"
              >
                x
              </button>
            </div>
            <div className="text-[10px] text-geo-dim mt-1">
              {formatSize(ds.size_bytes)} — {summary}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function _datasetSummary(ds: DatasetInfo): string {
  if (ds.raster) {
    return `${ds.raster.width}x${ds.raster.height}, ${ds.raster.bands} band${ds.raster.bands > 1 ? "s" : ""}, ${ds.raster.dtype}`;
  }
  if (ds.vector) {
    return `${ds.vector.feature_count.toLocaleString()} ${ds.vector.geometry_type}s, ${ds.vector.columns.length} cols`;
  }
  if (ds.point_cloud) {
    return `${ds.point_cloud.point_count.toLocaleString()} points${ds.point_cloud.has_color ? ", RGB" : ""}${ds.point_cloud.has_classification ? ", classified" : ""}`;
  }
  if (ds.multidim) {
    const vars = ds.multidim.variables.slice(0, 3).join(", ");
    const dims = Object.entries(ds.multidim.dimensions).map(([k, v]) => `${k}:${v}`).join(", ");
    return `${vars} [${dims}]`;
  }
  return "Unknown format";
}
