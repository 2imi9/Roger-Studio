import { useMemo, useState } from "react";
import type { DatasetInfo } from "../types";
import { tipsv2Label, type TIPSv2Model } from "../api/client";
import { LabelClassEditor, PRESETS } from "./LabelClassEditor";
import type { LabelClass } from "./LabelClassEditor";

interface TIPSv2PanelProps {
  datasets: DatasetInfo[];
  /** Same callback the Map-tab AutoLabel uses — drops the result on the
   *  map as a GeoJSON overlay so it can be toggled / overlaid with the
   *  OlmoEarth raster output for human comparison. */
  onResult: (geojson: GeoJSON.FeatureCollection, meta: Record<string, unknown>) => void;
}

const MODEL_OPTIONS: { value: TIPSv2Model; label: string; vram: string }[] = [
  { value: "google/tipsv2-b14", label: "B/14", vram: "~300 MB · fast" },
  { value: "google/tipsv2-l14", label: "L/14", vram: "~600 MB · balanced" },
  { value: "google/tipsv2-g14", label: "g/14", vram: "~2 GB · best" },
];

/** TIPSv2 zero-shot semantic labeling tab. Mirrors the AutoLabel Map-tab
 *  affordance but is the dedicated workflow surface for TIPSv2 — exposes
 *  the model size + sliding-window knobs the lighter Map-tab block hides.
 *
 *  Standalone by design: no encoder fusion with OlmoEarth. Both pipelines
 *  produce overlays the user can toggle on the map and analyze
 *  individually via the LLM tab. */
export function TIPSv2Panel({ datasets, onResult }: TIPSv2PanelProps) {
  const [selectedFile, setSelectedFile] = useState("");
  const [model, setModel] = useState<TIPSv2Model>("google/tipsv2-l14");
  const [slidingWindow, setSlidingWindow] = useState(true);
  const [customClasses, setCustomClasses] = useState<LabelClass[]>([...PRESETS["Land Cover"]]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{ count: number; method: string } | null>(null);

  // Only raster GeoTIFFs are supported by the TIPSv2 inspect path.
  const eligible = useMemo(
    () => datasets.filter((d) => d.format === "geotiff" || d.format === "cog"),
    [datasets]
  );

  const handleRun = async () => {
    if (!selectedFile) return;
    setRunning(true);
    setError(null);
    setLastResult(null);
    try {
      const res = await tipsv2Label(selectedFile, {
        model,
        slidingWindow,
        classes: customClasses,
      });
      onResult(res, (res as { properties?: Record<string, unknown> }).properties || {});
      setLastResult({
        count: res.features?.length || 0,
        method: ((res as { properties?: { method?: string } }).properties?.method) || "tipsv2",
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "TIPSv2 label failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <h3 className="m-0 mb-1 text-sm font-semibold text-geo-text">
        TIPSv2 Semantic Label
      </h3>
      <p className="m-0 mb-3 text-[11px] text-geo-dim leading-relaxed">
        Zero-shot label any uploaded GeoTIFF with text prompts. Standalone
        from OlmoEarth — drop both outputs on the map to overlay them, or
        ask the LLM tab to interpret each one.
      </p>

      {eligible.length === 0 ? (
        <div className="text-xs text-geo-dim py-3 px-3 bg-geo-surface border border-geo-border rounded-lg">
          Upload a GeoTIFF first (Map tab → Import Data).
        </div>
      ) : (
        <>
          {/* Dataset picker */}
          <label className="block text-[11px] uppercase tracking-wider text-geo-muted mb-1">
            Raster
          </label>
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="w-full px-2 py-1.5 bg-geo-bg border border-geo-border rounded-lg text-geo-text text-xs mb-3"
          >
            <option value="">Select a GeoTIFF…</option>
            {eligible.map((d) => (
              <option key={d.filename} value={d.filename}>
                {d.filename}
              </option>
            ))}
          </select>

          {/* Model size */}
          <label className="block text-[11px] uppercase tracking-wider text-geo-muted mb-1">
            Model
          </label>
          <div className="flex gap-1.5 mb-3">
            {MODEL_OPTIONS.map((m) => (
              <button
                key={m.value}
                onClick={() => setModel(m.value)}
                className={`flex-1 py-2 rounded-lg cursor-pointer text-xs font-semibold border transition-all ${
                  model === m.value
                    ? "text-white border-transparent shadow-sm"
                    : "bg-gradient-panel text-geo-text border-geo-border hover:border-geo-accent"
                }`}
                style={
                  model === m.value
                    ? { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
                    : undefined
                }
                title={m.vram}
              >
                <div>{m.label}</div>
                <div className="text-[9px] opacity-80 mt-0.5">{m.vram}</div>
              </button>
            ))}
          </div>

          {/* Sliding-window toggle */}
          <label className="flex items-start gap-2 mb-3 cursor-pointer">
            <input
              type="checkbox"
              checked={slidingWindow}
              onChange={(e) => setSlidingWindow(e.target.checked)}
              className="mt-0.5 accent-geo-accent"
            />
            <div className="text-[11px] leading-relaxed">
              <div className="font-semibold text-geo-text">Sliding-window</div>
              <div className="text-geo-muted">
                Overlapping-tile inference for pixel-accurate boundaries.
                ~2× wall time, much sharper edges.
              </div>
            </div>
          </label>

          {/* Quick-start preset chips — researcher's path for "explore
              what TIPSv2 can label on diverse scenes". Clicking a chip
              swaps the class set in one go; LabelClassEditor stays
              available for custom edits underneath. */}
          <div className="mb-2">
            <div className="text-[11px] uppercase tracking-wider text-geo-muted mb-1">
              Try a preset
            </div>
            <div className="flex flex-wrap gap-1">
              {(["City Detail", "Tree Types", "City + Trees Mix"] as const).map((key) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setCustomClasses([...PRESETS[key]])}
                  className="px-2 py-1 text-[10px] bg-geo-surface text-geo-text border border-geo-border rounded cursor-pointer hover:border-geo-accent hover:bg-geo-bg transition-colors"
                  title={`Load ${key} prompts (${PRESETS[key].length} classes)`}
                >
                  {key}
                </button>
              ))}
            </div>
          </div>

          {/* Class prompts */}
          <LabelClassEditor classes={customClasses} onChange={setCustomClasses} />

          {/* Run */}
          <button
            type="button"
            onClick={handleRun}
            disabled={running || !selectedFile}
            className={`mt-3 w-full py-2.5 rounded-lg text-[13px] font-semibold transition-all ${
              running || !selectedFile
                ? "bg-geo-border text-geo-dim cursor-not-allowed"
                : "text-white cursor-pointer shadow-sm hover:shadow-lg hover:-translate-y-0.5 hover:brightness-110 active:translate-y-0 active:brightness-95"
            }`}
            style={
              running || !selectedFile
                ? undefined
                : { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
            }
          >
            {running ? "Running TIPSv2…" : "Run + add to map"}
          </button>

          {error && (
            <div className="mt-3 text-[11px] text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {lastResult && !error && (
            <div className="mt-3 text-[11px] text-geo-muted bg-geo-surface border border-geo-border rounded-lg px-3 py-2">
              Added <span className="font-semibold text-geo-text">{lastResult.count}</span>
              {" "}polygons via <span className="font-mono">{lastResult.method}</span>. Toggle on the map
              alongside any OlmoEarth output to compare.
            </div>
          )}
        </>
      )}
    </div>
  );
}
