import { useState } from "react";
import type { DatasetInfo } from "../types";
import { uploadFile, deleteDataset } from "../api/client";
import { StatusOnline, StatusLoading, DeleteShape, SampleDataset } from "./icons";

export interface Sample {
  name: string;
  description: string;
  filename: string;
  url: string;
  icon: string;
  tags: string[];
}

// Vector-only presets — loaded into the dataset list for labeling/analysis.
// Raster presets live in SampleRasters.tsx, where they take a different path
// (also promoted to imagery layers on load) so the two flows stay visibly
// distinct in the UI (different Map-tab sub-views).
export const SAMPLES: Sample[] = [
  {
    name: "SF Parks & Landmarks",
    description: "7 parks, waterfront areas, and natural features in San Francisco",
    filename: "sf_parks.geojson",
    url: "/samples/sf_parks.geojson",
    icon: "P",
    tags: ["GeoJSON", "vector", "urban"],
  },
  {
    name: "PA Karst Features",
    description: "Sinkholes, caves, and depressions — BAI/OlmoEarth case study",
    filename: "pa_karst_features.geojson",
    url: "/samples/pa_karst_features.geojson",
    icon: "K",
    tags: ["GeoJSON", "geology", "hazard"],
  },
  {
    name: "Solar Energy Sites",
    description: "Major solar farms in CA/NV/AZ with capacity and GHI data",
    filename: "solar_sites.geojson",
    url: "/samples/solar_sites.geojson",
    icon: "S",
    tags: ["GeoJSON", "energy", "infrastructure"],
  },
];

interface SampleDataProps {
  onLoad: (ds: DatasetInfo) => void;
  onDelete: (filename: string) => void;
  loadedNames: Set<string>;
}

/** Shared sample-fetch-and-upload helper. Kept as a module-level export so
 * MapView's Added Layer "Samples" tab can reuse the exact same flow without
 * re-implementing the fetch → File → uploadFile chain (and drifting when we
 * tweak the path, e.g. if we ever introduce auth tokens on /upload). */
export async function loadSampleDataset(sample: Sample): Promise<DatasetInfo> {
  const res = await fetch(sample.url);
  const blob = await res.blob();
  const file = new File([blob], sample.filename, { type: blob.type });
  return uploadFile(file);
}

export function SampleData({ onLoad, onDelete, loadedNames }: SampleDataProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);

  const handleLoad = async (sample: Sample) => {
    setLoading(sample.filename);
    try {
      const ds = await loadSampleDataset(sample);
      onLoad(ds);
    } catch (e) {
      console.error("Sample load failed:", e);
    } finally {
      setLoading(null);
    }
  };

  // Double-click a loaded sample to drop it from the map. Backend DELETE
  // first (best-effort — local removal still happens if API call fails),
  // then strip from App state. Mirrors the flow in DataUpload.
  const handleDrop = async (sample: Sample) => {
    setRemoving(sample.filename);
    try {
      await deleteDataset(sample.filename);
    } catch {
      /* still remove locally even if backend call failed */
    }
    onDelete(sample.filename);
    setRemoving(null);
  };

  return (
    <div>
      <h3 className="m-0 mb-2 text-[13px] text-geo-muted font-medium">
        Sample Data
      </h3>
      {SAMPLES.map((s) => {
        const loaded = loadedNames.has(s.filename);
        const isLoading = loading === s.filename;
        const isRemoving = removing === s.filename;
        return (
          <button
            key={s.filename}
            onClick={() => !loaded && !isLoading && !isRemoving && handleLoad(s)}
            onDoubleClick={() => loaded && !isRemoving && handleDrop(s)}
            // NOTE: only disable mid-load / mid-remove. We deliberately do
            // NOT disable when loaded — disabled buttons don't fire dblclick,
            // which would block the drop gesture. Single-click on a loaded
            // card is gated to a no-op in onClick instead.
            disabled={isLoading || isRemoving}
            title={
              loaded
                ? "Loaded — double-click to drop from the map"
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
              <div className="flex gap-1 mt-1">
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
