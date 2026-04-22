import { useState } from "react";

export interface DataSourceConfig {
  sentinel2: boolean;
  sentinel1: boolean;
  landsat: boolean;
  naip: boolean;
  startDate: string;
  endDate: string;
}

const DEFAULT_CONFIG: DataSourceConfig = {
  sentinel2: true,
  sentinel1: false,
  landsat: false,
  naip: false,
  startDate: "2024-01-01",
  endDate: "2025-12-31",
};

const SOURCES = [
  { key: "sentinel2" as const, label: "Sentinel-2", detail: "Optical, 10m, 5-day", color: "#22c55e" },
  { key: "sentinel1" as const, label: "Sentinel-1", detail: "SAR, 10m, 12-day", color: "#3b82f6" },
  { key: "landsat" as const, label: "Landsat", detail: "Optical, 30m, 16-day", color: "#f59e0b" },
  { key: "naip" as const, label: "NAIP", detail: "Aerial RGB, 1m, US only", color: "#a855f7" },
];

interface DataSourcePickerProps {
  config: DataSourceConfig;
  onChange: (config: DataSourceConfig) => void;
}

export function DataSourcePicker({ config, onChange }: DataSourcePickerProps) {
  const [expanded, setExpanded] = useState(false);

  const toggle = (key: keyof DataSourceConfig) => {
    if (typeof config[key] === "boolean") {
      onChange({ ...config, [key]: !config[key] });
    }
  };

  const activeCount = SOURCES.filter((s) => config[s.key]).length;

  return (
    <div className="mb-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex justify-between items-center py-2 bg-transparent border-none text-geo-text cursor-pointer text-[13px] font-medium"
      >
        <span>Data Sources ({activeCount} selected)</span>
        <span className="text-[10px] text-geo-dim">{expanded ? "\u25B2" : "\u25BC"}</span>
      </button>

      {expanded && (
        <div className="bg-geo-surface border border-geo-border rounded-lg p-2.5">
          {/* Source toggles */}
          {SOURCES.map((src) => (
            <label
              key={src.key}
              className="flex items-center gap-2 py-1.5 cursor-pointer text-xs text-geo-text"
            >
              <input
                type="checkbox"
                checked={config[src.key] as boolean}
                onChange={() => toggle(src.key)}
                style={{ accentColor: src.color }}
              />
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: config[src.key] ? src.color : undefined }}
              />
              <span className="flex-1">{src.label}</span>
              <span className="text-geo-dim text-[10px]">{src.detail}</span>
            </label>
          ))}

          {/* Date range */}
          <div className="grid grid-cols-2 gap-2 mt-2.5 border-t border-geo-border pt-2.5">
            <div>
              <label className="text-[10px] text-geo-dim block mb-0.5">
                Start Date
              </label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => onChange({ ...config, startDate: e.target.value })}
                className="w-full px-1.5 py-1 bg-geo-bg border border-geo-border rounded text-geo-text text-[11px] box-border"
              />
            </div>
            <div>
              <label className="text-[10px] text-geo-dim block mb-0.5">
                End Date
              </label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => onChange({ ...config, endDate: e.target.value })}
                className="w-full px-1.5 py-1 bg-geo-bg border border-geo-border rounded text-geo-text text-[11px] box-border"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export { DEFAULT_CONFIG };
