import { useEffect, useState } from "react";
import type { BBox } from "../types";
import { getPolygonStats, type PolygonStatsResponse } from "../api/client";
import { Panel, SectionTitle } from "./ui/Panel";

interface PolygonStatsProps {
  selectedArea: BBox | null;
  // When the user draws an actual polygon via terra-draw, this carries the
  // real shape. Otherwise we fall back to a 4-vertex rectangle from bbox.
  selectedGeometry?: GeoJSON.Polygon | null;
  onClear?: () => void;
}

function bboxToPolygon(bbox: BBox): GeoJSON.Polygon {
  return {
    type: "Polygon",
    coordinates: [[
      [bbox.west, bbox.south],
      [bbox.east, bbox.south],
      [bbox.east, bbox.north],
      [bbox.west, bbox.north],
      [bbox.west, bbox.south],
    ]],
  };
}

export function PolygonStats({ selectedArea, selectedGeometry, onClear }: PolygonStatsProps) {
  const [stats, setStats] = useState<PolygonStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Key the effect on geometry shape (ring-length + stringified coords) so
  // StrictMode's double-mount in dev doesn't race and new polygons actually
  // re-fetch even if the bbox reference happens to match.
  const geometryKey = selectedGeometry
    ? JSON.stringify(selectedGeometry.coordinates[0] ?? [])
    : selectedArea
    ? `bbox:${selectedArea.west},${selectedArea.south},${selectedArea.east},${selectedArea.north}`
    : "";

  useEffect(() => {
    if (!geometryKey) {
      setStats(null);
      setError(null);
      return;
    }
    const geom: GeoJSON.Polygon =
      selectedGeometry ?? bboxToPolygon(selectedArea as BBox);
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPolygonStats(geom, { includeElevation: true, resolution: 15 })
      .then((r) => { if (!cancelled) setStats(r); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "failed"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometryKey]);

  if (!selectedArea) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <SectionTitle>Path or Polygon</SectionTitle>
        <div className="flex items-center gap-2">
          <span
            className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-accent-soft text-geo-accent"
            title="Geodesic perimeter + area + Open-Meteo elevation stats, computed server-side."
          >
            Live
          </span>
          {onClear && (
            <button
              onClick={onClear}
              aria-label="Clear selection"
              title="Clear selection"
              className="w-6 h-6 flex items-center justify-center rounded-full bg-geo-elevated hover:bg-red-500 hover:text-white text-geo-muted cursor-pointer text-xs font-bold transition-colors"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {loading && !stats && (
        <p className="text-xs text-geo-muted">Measuring…</p>
      )}

      {error && (
        <p className="text-xs text-geo-danger" style={{ color: "#dc2626" }}>
          {error}
        </p>
      )}

      {stats && !stats.error && (
        <>
          <Panel border className="mb-3">
            <div className="space-y-2">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-geo-muted">Perimeter</p>
                <p className="text-base font-mono">
                  {stats.perimeter_km < 1
                    ? `${(stats.perimeter_km * 1000).toFixed(0)} m`
                    : `${stats.perimeter_km.toFixed(2)} km`}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-wider text-geo-muted">Area</p>
                <p className="text-base font-mono">
                  {stats.area_km2 < 0.01
                    ? `${(stats.area_km2 * 1_000_000).toFixed(0)} m²`
                    : `${stats.area_km2.toFixed(2)} km²`}
                </p>
              </div>
            </div>
          </Panel>

          {stats.elevation ? (
            <Panel border>
              <p className="text-[11px] uppercase tracking-wider text-geo-muted mb-2">
                Elevation estimate
              </p>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-[10px] text-geo-muted">Min</p>
                  <p className="font-mono text-sm">{stats.elevation.min_m.toFixed(0)} m</p>
                </div>
                <div>
                  <p className="text-[10px] text-geo-muted">Median</p>
                  <p className="font-mono text-sm">{stats.elevation.median_m.toFixed(0)} m</p>
                </div>
                <div>
                  <p className="text-[10px] text-geo-muted">Max</p>
                  <p className="font-mono text-sm">{stats.elevation.max_m.toFixed(0)} m</p>
                </div>
              </div>
              <p className="mt-2 text-[10px] text-geo-muted text-center">
                {stats.elevation_sample_count} samples · range{" "}
                {stats.elevation.range_m.toFixed(0)} m · {stats.elevation.source}
              </p>
            </Panel>
          ) : (
            <p className="text-xs text-geo-muted">Elevation unavailable.</p>
          )}
        </>
      )}

      {stats?.error && (
        <p className="text-xs text-geo-muted">{stats.error}</p>
      )}
    </div>
  );
}
