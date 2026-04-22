import { useState, useMemo, useEffect } from "react";
import type { DatasetInfo } from "../types";
import { autoLabel, validateLabels, gemmaHealth, gemmaStart } from "../api/client";
import { LabelClassEditor, PRESETS } from "./LabelClassEditor";
import type { LabelClass } from "./LabelClassEditor";
import { ConfidenceDot } from "./ui/ConfidenceDot";
import { Badge } from "./ui/Badge";

interface AutoLabelProps {
  datasets: DatasetInfo[];
  onResult: (geojson: GeoJSON.FeatureCollection, meta: Record<string, unknown>) => void;
}

export function AutoLabel({ datasets, onResult }: AutoLabelProps) {
  const [selectedFile, setSelectedFile] = useState("");
  const [nClasses, setNClasses] = useState(6);
  const [method, setMethod] = useState<"auto" | "tipsv2" | "spectral" | "samgeo">("auto");
  const [customClasses, setCustomClasses] = useState<LabelClass[]>([...PRESETS["Land Cover"]]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const [running, setRunning] = useState(false);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rawResult, setRawResult] = useState<any>(null);
  const [gemmaOk, setGemmaOk] = useState<boolean | null>(null);
  const [dockerOk, setDockerOk] = useState<boolean>(false);
  const [starting, setStarting] = useState(false);

  const pollHealth = async () => {
    try {
      const h = await gemmaHealth();
      setGemmaOk(h.reachable);
      setDockerOk(h.docker_available);
      return h;
    } catch {
      setGemmaOk(false);
      return null;
    }
  };

  useEffect(() => {
    pollHealth();
  }, []);

  // Poll every 5 s while ``starting`` is true. Previously this loop had
  // no cap: if the vLLM container failed silently (image build looping,
  // port firewalled, model download hung) the UI sat on "starting…"
  // indefinitely and the Start button stayed disabled forever. Now we:
  //   1. Stop early if the backend reports ``last_start_error`` — the
  //      docker lifecycle gave up and we should surface why.
  //   2. Hard-cap at ``MAX_START_POLLS`` attempts (5 min @ 5 s each).
  //      Slow first-run model downloads can legitimately take a few
  //      minutes, but past that we're almost certainly wedged. When we
  //      hit the cap, flip ``starting`` back to false so the user can
  //      click Start again or investigate the docker logs.
  useEffect(() => {
    if (!starting) return;
    let attempts = 0;
    const MAX_START_POLLS = 60; // 60 × 5 s = 5 min
    const t = setInterval(async () => {
      attempts += 1;
      const h = await pollHealth();
      if (h?.reachable) {
        setStarting(false);
        return;
      }
      // Backend-reported start failure (image build crashed, model
      // download 404'd, etc.). Surface the message and stop polling.
      if (h?.last_start_error) {
        setError(`LLM start failed: ${h.last_start_error}`);
        setStarting(false);
        return;
      }
      if (attempts >= MAX_START_POLLS) {
        setError(
          `LLM failed to become reachable after ${MAX_START_POLLS * 5}s. ` +
          "Check ``docker logs`` for the container, or try again.",
        );
        setStarting(false);
      }
    }, 5000);
    return () => clearInterval(t);
  }, [starting]);

  const handleStart = async () => {
    // Clear any prior error banner so the user sees the fresh outcome
    // instead of a stale "failed" message carrying over from a previous
    // attempt.
    setError(null);
    try {
      await gemmaStart();
      setStarting(true);
    } catch (e) {
      // gemmaStart itself rejected — container lifecycle call didn't
      // even reach the docker daemon. ``starting`` stayed false, so
      // the Start button auto-re-enables; no stuck state to reset.
      setError(e instanceof Error ? e.message : "LLM start failed");
    }
  };

  const eligible = datasets.filter((d) =>
    ["geotiff", "cog", "geojson", "geopackage", "shapefile", "parquet", "geoparquet", "csv"].includes(d.format)
  );

  const isRaster =
    eligible.find((d) => d.filename === selectedFile)?.format === "geotiff" ||
    eligible.find((d) => d.filename === selectedFile)?.format === "cog";

  // Filter results by confidence threshold
  const filteredResult = useMemo(() => {
    if (!rawResult) return null;
    const meta = rawResult.properties || {};
    if (confidenceThreshold <= 0) return meta;

    const kept = rawResult.features?.filter(
      (f: any) => (f.properties?.confidence || 0) >= confidenceThreshold
    );
    return {
      ...meta,
      total_features: kept?.length || 0,
      needs_review_count: kept?.filter((f: any) => f.properties?.needs_review).length || 0,
    };
  }, [rawResult, confidenceThreshold]);

  const handleRun = async () => {
    if (!selectedFile) return;
    setRunning(true);
    setError(null);
    setRawResult(null);
    try {
      const res = await autoLabel(selectedFile, nClasses, method);
      setRawResult(res);

      // Apply confidence filter before sending to map
      const filtered = {
        ...res,
        features: (res.features || []).filter(
          (f: any) => (f.properties?.confidence || 0) >= confidenceThreshold
        ),
      };
      onResult(filtered, (res as any).properties || {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-label failed");
    } finally {
      setRunning(false);
    }
  };

  const handleValidate = async () => {
    if (!rawResult || !selectedFile) return;
    setValidating(true);
    setError(null);
    try {
      const pipeline =
        rawResult.properties?.method === "samgeo"
          ? "samgeo"
          : rawResult.properties?.method === "spectral_kmeans"
          ? "spectral"
          : "tipsv2";
      const validated = await validateLabels(selectedFile, rawResult, {
        pipeline,
        onlyLowConfidence: true,
        lowConfThreshold: 0.6,
      });
      setRawResult(validated);
      const filtered = {
        ...validated,
        features: (validated.features || []).filter(
          (f: any) => (f.properties?.confidence || 0) >= confidenceThreshold
        ),
      };
      onResult(filtered, (validated as any).properties || {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gemma validation failed");
    } finally {
      setValidating(false);
    }
  };

  // Re-filter when threshold changes and we have results
  const handleThresholdChange = (val: number) => {
    setConfidenceThreshold(val);
    if (rawResult) {
      const filtered = {
        ...rawResult,
        features: (rawResult.features || []).filter(
          (f: any) => (f.properties?.confidence || 0) >= val
        ),
      };
      onResult(filtered, rawResult.properties || {});
    }
  };

  return (
    <div>
      <h3 className="m-0 mb-1 text-sm font-semibold text-geo-text">
        Auto-Label
      </h3>
      <p className="m-0 mb-2 text-[11px] text-geo-dim">
        TIPSv2 zero-shot, spectral K-means, or SamGeo. LLM validates results.
      </p>

      {eligible.length === 0 ? (
        <div className="text-xs text-geo-dim py-2">
          Upload a GeoTIFF or vector file first
        </div>
      ) : (
        <>
          {/* Dataset selector */}
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="w-full px-2 py-1.5 bg-geo-bg border border-geo-border rounded-lg text-geo-text text-xs mb-2"
          >
            <option value="">Select dataset...</option>
            {eligible.map((d) => (
              <option key={d.filename} value={d.filename}>
                {d.filename} ({d.format})
              </option>
            ))}
          </select>

          {/* Method selector */}
          <div className="flex gap-1.5 mb-3">
            {(["auto", "tipsv2", "spectral", "samgeo"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMethod(m)}
                className={`flex-1 py-2.5 rounded-lg cursor-pointer text-xs font-semibold border transition-all ${
                  method === m
                    ? "bg-gradient-to-br from-sage-500 to-sage-700 text-white border-transparent shadow-sm"
                    : "bg-gradient-panel text-geo-text border-geo-border hover:border-geo-accent"
                }`}
                style={method === m ? { background: "linear-gradient(135deg, #7a9c6b 0%, #5d7e4e 100%)" } : undefined}
              >
                {m === "auto" ? "Auto" : m === "tipsv2" ? "TIPSv2" : m === "spectral" ? "Spectral" : "SamGeo"}
              </button>
            ))}
          </div>

          {/* Custom label classes editor */}
          <LabelClassEditor classes={customClasses} onChange={setCustomClasses} />

          {/* Class count for spectral */}
          {(method === "spectral" || !isRaster) && (
            <div className="flex items-center gap-2 mb-2">
              <label className="text-[11px] text-geo-muted">Clusters:</label>
              <input type="range" min={2} max={12} value={nClasses}
                onChange={(e) => setNClasses(parseInt(e.target.value))}
                className="flex-1 accent-geo-success"
              />
              <span className="text-xs text-geo-text font-mono w-5">{nClasses}</span>
            </div>
          )}

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!selectedFile || running}
            className={`w-full py-3 border-none rounded-xl text-sm font-semibold mb-2 transition-all ${
              !selectedFile || running
                ? "bg-geo-border text-geo-dim cursor-not-allowed"
                : "text-white cursor-pointer shadow-md hover:shadow-lg hover:-translate-y-0.5"
            }`}
            style={
              !selectedFile || running
                ? undefined
                : { background: "linear-gradient(135deg, #7a9c6b 0%, #5d7e4e 100%)" }
            }
          >
            {running ? "Segmenting..." : "Run Auto-Label"}
          </button>
        </>
      )}

      {error && (
        <div className="bg-red-900 text-red-300 rounded-lg p-2 text-xs mb-2">
          {error}
        </div>
      )}

      {/* Results + confidence filter */}
      {filteredResult && (
        <div className="bg-geo-surface border border-geo-border rounded-lg p-2.5">
          {/* Method badge */}
          <div className="mb-1.5">
            <Badge label={filteredResult.method || "unknown"} />
            {filteredResult.global_class && (
              <span className="text-[11px] text-geo-text ml-2">
                {filteredResult.global_class}{" "}
                <ConfidenceDot value={filteredResult.global_confidence || 0} showLabel />
              </span>
            )}
          </div>

          {/* Confidence threshold slider */}
          <div className="mb-2">
            <div className="flex justify-between text-[11px]">
              <span className="text-geo-muted">Min Confidence</span>
              <span className="text-geo-text font-mono">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range" min={0} max={0.9} step={0.05} value={confidenceThreshold}
              onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
              className="w-full accent-geo-warn"
            />
            <div className="flex justify-between text-[9px] text-geo-dim">
              <span>Show all</span>
              <span>Only confident</span>
            </div>
          </div>

          {/* Stats */}
          <div className="flex justify-between mb-1">
            <span className="text-xs text-geo-muted">Features shown</span>
            <span className="text-xs text-geo-text font-semibold">
              {filteredResult.total_features?.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between mb-1">
            <span className="text-xs text-geo-muted">Needs Review</span>
            <span className="text-xs text-geo-warn font-semibold">
              {filteredResult.needs_review_count}
            </span>
          </div>

          {/* Class breakdown */}
          {filteredResult.class_summary && (
            <div className="mt-1.5 border-t border-geo-border pt-1.5">
              {Object.entries(filteredResult.class_summary as Record<string, any>)
                .sort((a, b) => (b[1].percentage || 0) - (a[1].percentage || 0))
                .map(([name, info]) => (
                  <div key={name} className="flex items-center gap-1.5 mb-0.5">
                    <div
                      className="w-2 h-2 rounded-sm shrink-0"
                      style={{ background: info.color }}
                    />
                    <span className="flex-1 text-[11px] text-geo-text">{name}</span>
                    <span className="text-[10px] text-geo-muted font-mono">{info.percentage}%</span>
                  </div>
                ))}
            </div>
          )}

          <div className="text-[9px] text-geo-dim mt-1">
            {filteredResult.model_version}
          </div>

          {/* LLM validator */}
          <div className="mt-2 border-t border-geo-border pt-2">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-semibold text-geo-text">LLM Validator</span>
              <div className="flex items-center gap-1">
                <span
                  className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                    starting
                      ? "bg-blue-100 text-blue-700"
                      : gemmaOk === true
                      ? "bg-green-100 text-green-700"
                      : gemmaOk === false
                      ? "bg-amber-100 text-amber-700"
                      : "bg-gray-100 text-gray-500"
                  }`}
                  title={
                    starting
                      ? "LLM starting (first launch downloads model)"
                      : gemmaOk === true
                      ? "LLM endpoint reachable"
                      : !dockerOk
                      ? "Local runtime not installed"
                      : "LLM not running"
                  }
                >
                  {starting ? "starting…" : gemmaOk === true ? "online" : gemmaOk === false ? "offline" : "…"}
                </span>
                {gemmaOk === false && !starting && (
                  <button
                    onClick={handleStart}
                    disabled={!dockerOk}
                    className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-geo-accent text-white disabled:bg-geo-border disabled:text-geo-dim disabled:cursor-not-allowed"
                    style={
                      dockerOk
                        ? { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
                        : undefined
                    }
                    title={dockerOk ? "docker run (first launch downloads ~16GB)" : "Install Docker first"}
                  >
                    Start
                  </button>
                )}
              </div>
            </div>
            <button
              onClick={handleValidate}
              disabled={!gemmaOk || validating}
              className={`w-full py-2 border-none rounded-lg text-xs font-semibold transition-all ${
                !gemmaOk || validating
                  ? "bg-geo-border text-geo-dim cursor-not-allowed"
                  : "text-white cursor-pointer shadow-sm hover:shadow-md"
              }`}
              style={
                !gemmaOk || validating
                  ? undefined
                  : { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
              }
              title="Re-validate low-confidence polygons with the LLM using patch + elevation + weather + neighbor context"
            >
              {validating ? "Validating..." : "Validate with LLM"}
            </button>
            {filteredResult.gemma_validation && (
              <div className="mt-1.5 text-[10px] text-geo-muted">
                <div>
                  Reviewed {filteredResult.gemma_validation.validated_count} /{" "}
                  {filteredResult.gemma_validation.validated_count +
                    (filteredResult.gemma_validation.skipped_count || 0)}
                </div>
                {filteredResult.gemma_validation.action_counts && (
                  <div className="flex gap-2 flex-wrap mt-0.5">
                    {Object.entries(
                      filteredResult.gemma_validation.action_counts as Record<string, number>
                    ).map(([action, count]) => (
                      <span key={action} className="font-mono">
                        {action}: {count}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
