import { useState } from "react";
import type { BBox } from "../types";
import { getCompositeTileUrl } from "../api/client";
import type { ImageryLayer } from "./MapView";
import { Panel, SectionTitle } from "./ui/Panel";

interface StacImageryProps {
  selectedArea: BBox | null;
  imageryLayers: ImageryLayer[];
  onAdd: (layer: ImageryLayer) => void;
  onRemove: (id: string) => void;
}

// Rolling 3-month "recent cloud-free" window ending today. Good default for
// a quick demo and works year-round; user can refine later via custom range.
function defaultInterval(): string {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - 3);
  const fmt = (d: Date) => d.toISOString().slice(0, 10);
  return `${fmt(start)}/${fmt(end)}`;
}

export function StacImagery({ selectedArea, imageryLayers, onAdd, onRemove }: StacImageryProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [interval, setInterval] = useState(defaultInterval());

  const addSentinel2 = async () => {
    if (!selectedArea) return;
    setLoading(true);
    setError(null);
    try {
      const r = await getCompositeTileUrl({
        bbox: selectedArea,
        datetime: interval,
        collection: "sentinel-2-l2a",
        maxCloudCover: 20,
      });
      if (r.error) {
        setError(`${r.error}: ${(r as unknown as { detail?: string }).detail ?? ""}`);
        return;
      }
      onAdd({
        id: `s2-${r.search_id.slice(0, 8)}`,
        tileUrl: r.tile_url,
        label: `Sentinel-2 ${interval}`,
        opacity: 1,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <SectionTitle>Imagery</SectionTitle>
        <span
          className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-accent-soft text-geo-accent"
          title="Least-cloudy mosaic from Microsoft Planetary Computer."
        >
          Live
        </span>
      </div>

      <Panel border className="mb-3">
        <label className="block text-[11px] uppercase tracking-wider text-geo-muted mb-1">
          Date range (RFC 3339 interval)
        </label>
        <input
          type="text"
          value={interval}
          onChange={(e) => setInterval(e.target.value)}
          placeholder="2024-06-01/2024-09-01"
          className="w-full px-2 py-1 bg-geo-surface border border-geo-border rounded text-[12px] font-mono text-geo-text focus:outline-none focus:border-geo-accent"
        />
        <button
          onClick={addSentinel2}
          disabled={!selectedArea || loading}
          className={`mt-3 w-full py-2 rounded-lg text-xs font-semibold transition-all ${
            !selectedArea || loading
              ? "bg-geo-elevated text-geo-muted cursor-not-allowed"
              : "bg-gradient-primary text-white cursor-pointer hover:shadow-md"
          }`}
        >
          {loading ? "Fetching composite…" : "Add cloud-free Sentinel-2"}
        </button>
        {error && <p className="mt-2 text-xs" style={{ color: "#dc2626" }}>{error}</p>}
        {!selectedArea && (
          <p className="mt-2 text-[11px] text-geo-muted">
            Draw a rectangle or polygon first to scope the mosaic.
          </p>
        )}
      </Panel>

      {imageryLayers.length > 0 && (
        <div className="space-y-2">
          {imageryLayers.map((layer) => (
            <div
              key={layer.id}
              className="flex items-center gap-2 px-3 py-2 bg-geo-surface border border-geo-border rounded-lg text-xs"
            >
              <span className="flex-1 truncate" title={layer.label}>
                {layer.label ?? layer.id}
              </span>
              <button
                onClick={() => onRemove(layer.id)}
                aria-label={`Remove ${layer.id}`}
                className="w-5 h-5 flex items-center justify-center rounded-full bg-geo-elevated hover:bg-red-500 hover:text-white text-geo-muted cursor-pointer text-xs font-bold transition-colors"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
