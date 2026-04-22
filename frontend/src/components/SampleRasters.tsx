import { useMemo, useState } from "react";
import type { DatasetInfo } from "../types";
import type { ImageryLayer } from "./MapView";
import { uploadFile, deleteDataset } from "../api/client";
import { StatusOnline, StatusLoading, DeleteShape, SampleDataset } from "./icons";

export interface RasterSample {
  name: string;
  description: string;
  filename: string;
  url: string;
  tags: string[];
  /** Optional hex tint applied to the layer label pill — purely cosmetic,
   * helps the user spot which sample a layer came from in the Added Layer
   * popover when several rasters are on the map at once. */
  accent?: string;
}

// Curated raster presets. Each one also exercises a distinct path through the
// tile server's renderer (single-band colormap vs multi-band RGB composite),
// so loading all three is a quick tour of how raster data surfaces in Studio.
export const SAMPLE_RASTERS: RasterSample[] = [
  {
    name: "Knoxville NDVI 2024",
    description: "Vegetation index from Landsat — 1-band, colormap gradient",
    filename: "knoxville_ndvi_2024.tif",
    url: "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/knoxville_ndvi_2024.tif",
    tags: ["GeoTIFF", "NDVI", "1-band"],
    accent: "#2f7a3d",
  },
  {
    name: "Knoxville Landsat 2024",
    description: "Multispectral Landsat scene — true-color RGB composite",
    filename: "knoxville_landsat_2024.tif",
    url: "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/knoxville_landsat_2024.tif",
    tags: ["GeoTIFF", "Landsat", "RGB"],
    accent: "#3a6690",
  },
  {
    name: "Wetland Prediction Mask",
    description: "Binary wetland classification — what a segmentation head outputs",
    filename: "wetland_prediction.tif",
    url: "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/wetland_prediction.tif",
    tags: ["GeoTIFF", "classification", "mask"],
    accent: "#5b8a52",
  },
];

interface SampleRastersProps {
  onLoad: (ds: DatasetInfo) => void;
  onDelete: (filename: string) => void;
  onAddImageryLayer: (l: ImageryLayer) => void;
  onRemoveImageryLayer: (id: string) => void;
  imageryLayers: ImageryLayer[];
  datasets: DatasetInfo[];
}

/** Build the layer id we use to key this sample's imagery layer. Matches the
 * convention DatasetDetail uses ("upload-<filename>") so a sample added here
 * and re-added via DatasetDetail's "+ Add as map layer" resolve to the same
 * layer (replacement, not duplication). */
function layerIdFor(filename: string): string {
  return `upload-${filename}`;
}

export function SampleRasters({
  onLoad,
  onDelete,
  onAddImageryLayer,
  onRemoveImageryLayer,
  imageryLayers,
  datasets,
}: SampleRastersProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);

  // A sample is "loaded" when its imagery layer is on the map. Dataset
  // presence alone isn't the right signal — the user could drop the layer
  // from the Added Layer popover without deleting the backing dataset.
  const onMapIds = useMemo(
    () => new Set(imageryLayers.map((l) => l.id)),
    [imageryLayers],
  );
  const datasetNames = useMemo(
    () => new Set(datasets.map((d) => d.filename)),
    [datasets],
  );

  const handleLoad = async (sample: RasterSample) => {
    setLoading(sample.filename);
    try {
      // If the dataset is already uploaded (e.g. user loaded it via Import
      // Data earlier), skip the fetch+upload round-trip and just promote it
      // to a map layer.
      let ds: DatasetInfo | null = null;
      if (!datasetNames.has(sample.filename)) {
        const res = await fetch(sample.url);
        const blob = await res.blob();
        const file = new File([blob], sample.filename, { type: blob.type });
        ds = await uploadFile(file);
      }
      const safeName = encodeURIComponent(sample.filename);
      const layerId = layerIdFor(sample.filename);
      const layer = {
        id: layerId,
        tileUrl: `${window.location.origin}/api/datasets/${safeName}/tiles/{z}/{x}/{y}.png`,
        label: `${sample.name} · sample raster`,
        opacity: 0.85,
      };
      onAddImageryLayer(layer);
      if (ds) onLoad(ds);
      // MapLibre quirk: a raster source added while the map is animating
      // (the flyTo `onLoad` triggers above) often doesn't schedule tile
      // fetches until the user interacts with the map — so the layer reads
      // as "loaded" but paints nothing. Manually toggling the layer off and
      // on in the Added Layer panel forces a fresh tile queue, which works.
      // Replicate that automatically: wait for the flyTo to settle, then
      // remove + re-add the layer to kick MapLibre into requesting tiles.
      window.setTimeout(() => {
        onRemoveImageryLayer(layerId);
        window.setTimeout(() => onAddImageryLayer(layer), 50);
      }, 1800);
    } catch (e) {
      console.error("Raster sample load failed:", e);
    } finally {
      setLoading(null);
    }
  };

  // Drop = symmetric undo: take it off the map AND remove the backing
  // dataset. Backend delete is best-effort (mirrors DataUpload's delete
  // flow) so UI state stays consistent even if the API call fails.
  const handleDrop = async (sample: RasterSample) => {
    setRemoving(sample.filename);
    onRemoveImageryLayer(layerIdFor(sample.filename));
    try {
      await deleteDataset(sample.filename);
    } catch {
      /* still clear local state even if backend delete fails */
    }
    onDelete(sample.filename);
    setRemoving(null);
  };

  return (
    <div>
      <h3 className="m-0 mb-2 text-[13px] text-geo-muted font-medium">
        Sample Rasters
      </h3>
      <p className="text-[10px] text-geo-dim mb-2 leading-snug">
        One click fetches the GeoTIFF, caches it on the backend, and drops it on
        the map as an imagery layer — the same path OlmoEarth inference outputs
        take. Find it in the Added Layer popover's <em>On map</em> tab.
      </p>
      {SAMPLE_RASTERS.map((s) => {
        const id = layerIdFor(s.filename);
        const loaded = onMapIds.has(id);
        const isLoading = loading === s.filename;
        const isRemoving = removing === s.filename;
        return (
          <button
            key={s.filename}
            onClick={() => !loaded && !isLoading && !isRemoving && handleLoad(s)}
            onDoubleClick={() => loaded && !isRemoving && handleDrop(s)}
            disabled={isLoading || isRemoving}
            title={
              loaded
                ? "Loaded on map — double-click to drop"
                : isLoading
                  ? "Loading…"
                  : "Click to load onto the map"
            }
            className={`flex items-start gap-2 w-full p-2.5 mb-1.5 rounded-lg text-left text-geo-text transition-colors border ${
              loaded
                ? "border-geo-success/30 bg-geo-success/5 cursor-pointer hover:bg-red-50 hover:border-red-300"
                : "bg-geo-surface border-transparent cursor-pointer hover:border-geo-border"
            } ${isLoading || isRemoving ? "opacity-60 cursor-default" : ""}`}
          >
            <span
              className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                loaded
                  ? "bg-geo-success/10"
                  : isRemoving
                    ? "bg-red-50"
                    : "bg-geo-border"
              }`}
              style={!loaded && !isRemoving && s.accent ? { background: `${s.accent}22` } : undefined}
            >
              {isRemoving ? (
                <DeleteShape className="w-4 h-4 [&_*]:!stroke-red-700" />
              ) : loaded ? (
                <StatusOnline size={18} className="[&_circle]:!fill-geo-success [&_circle]:!stroke-geo-success" />
              ) : isLoading ? (
                <StatusLoading className="w-4 h-4" />
              ) : (
                <SampleDataset className="w-4 h-4" />
              )}
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold">{s.name}</div>
              <div className="text-[10px] text-geo-dim mt-0.5">{s.description}</div>
              <div className="flex gap-1 mt-1 flex-wrap">
                {s.tags.map((t) => (
                  <span
                    key={t}
                    className="text-[9px] bg-geo-border rounded px-1.5 py-0.5 text-geo-muted"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
