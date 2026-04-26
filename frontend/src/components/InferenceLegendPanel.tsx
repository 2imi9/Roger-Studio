/**
 * Floating popover legend(s) for one or more active OlmoEarth inference layers.
 *
 * Position: top-right corner of the map area (rendered as an absolute child of
 * App.tsx's <main className="relative">, beside MapView). One ~280 px popover
 * per active inference layer, vertically stacked.
 *
 * The backend packs a per-task ``legend`` block into the /api/olmoearth/infer
 * response — classification/segmentation jobs carry per-class hex colors
 * sourced from olmoearth_projects' rslearn configs, regression jobs carry a
 * gradient stop list + the predicted value in task units, embedding jobs
 * carry a colormap-gradient hint.
 *
 * Visual language ported from Claude Design 2026-04-26 v2 component3:
 * 280 px wide, display-font header + muted subtitle, swatch + name rows,
 * coverage row at bottom, and a GeoJSON export button for classification
 * heads (wires the existing /api/olmoearth/ft-classification/geojson endpoint).
 */
import { useState } from "react";
import type { ImageryLayer } from "./MapView";
import type { OlmoEarthInferenceResult, OlmoEarthLegend } from "../api/client";
import { downloadFtClassificationGeoJson } from "../api/client";

interface Props {
  imageryLayers: ImageryLayer[];
  onRemove?: (id: string) => void;
}

function prettyModelName(repoId: string): string {
  return repoId
    .replace(/^allenai\//, "")
    .replace(/^OlmoEarth-v1-FT-/, "")
    .replace(/^OlmoEarth-v1-/, "");
}

function taskSubtitle(meta: OlmoEarthInferenceResult): string {
  if (meta.kind === "stub") return "Stub fallback";
  if (meta.task_type === "embedding") {
    const dim = typeof meta.embedding_dim === "number" ? ` · ${meta.embedding_dim}-d` : "";
    return `Embedding · PCA false-color${dim}`;
  }
  if (meta.task_type === "regression") return "Regression · per-pixel value";
  if (meta.task_type === "classification" || meta.task_type === "segmentation") {
    const n = meta.legend && "classes" in meta.legend ? meta.legend.classes.length : null;
    return n ? `Classification · ${n} classes` : "Classification";
  }
  return meta.task_type ?? meta.kind ?? "Inference";
}

function sceneCoverageRow(meta: OlmoEarthInferenceResult): { label: string; value: string } | null {
  if (!meta.scene_id && !meta.scene_datetime) return null;
  const date = meta.scene_datetime?.split("T")[0] ?? "—";
  return { label: "Sentinel-2", value: date };
}

function isClassLegend(
  l: OlmoEarthLegend | undefined,
): l is Extract<OlmoEarthLegend, { classes: unknown }> {
  return !!l && "classes" in l && Array.isArray(l.classes);
}

function isRegressionLegend(
  l: OlmoEarthLegend | undefined,
): l is Extract<OlmoEarthLegend, { kind: "regression" }> {
  return !!l && "kind" in l && l.kind === "regression";
}

function GradientBar({ stops }: { stops: [string, number][] }) {
  const stopStr = stops
    .map(([hex, pos]) => `${hex} ${(pos * 100).toFixed(0)}%`)
    .join(", ");
  return (
    <div
      className="h-2 rounded-full border border-geo-border"
      style={{ backgroundImage: `linear-gradient(to right, ${stopStr})` }}
      role="img"
      aria-label="colormap gradient"
    />
  );
}

function LegendNote({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <p className="text-[10px] italic text-geo-muted leading-snug px-3 pb-2">
      {text}
    </p>
  );
}

function CloseButton({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="w-[22px] h-[22px] inline-flex items-center justify-center rounded-sm text-geo-muted hover:bg-geo-elevated hover:text-geo-text transition-colors flex-shrink-0"
    >
      <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <line x1="2" y1="2" x2="10" y2="10" />
        <line x1="10" y1="2" x2="2" y2="10" />
      </svg>
    </button>
  );
}

function GeoJsonIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 15v4h14v-4" />
      <polyline points="12,5 12,15" />
      <polyline points="8,11 12,15 16,11" />
    </svg>
  );
}

function ClassList({ classes }: { classes: { index: number; name: string; color: string }[] }) {
  return (
    <ul className="px-3 py-1 space-y-0 max-h-[260px] overflow-y-auto">
      {classes.map((c) => (
        <li
          key={c.index}
          className="flex items-center gap-2 py-1 text-[12.5px] text-geo-text"
        >
          <span
            className="inline-block w-3 h-3 rounded-sm border border-geo-border flex-shrink-0"
            style={{ backgroundColor: c.color }}
          />
          <span className="truncate" title={c.name}>{c.name}</span>
        </li>
      ))}
    </ul>
  );
}

function ClassExtras({ meta }: { meta: OlmoEarthInferenceResult }) {
  if (!meta.class_probs || !meta.class_names) return null;
  const top = [...meta.class_probs]
    .map((p, i) => ({ i, p, name: meta.class_names![i] ?? `class_${i}` }))
    .sort((a, b) => b.p - a.p)[0];
  if (!top) return null;
  return (
    <p className="px-3 pt-1 pb-2 text-[11px] text-geo-muted">
      Top class: <span className="font-mono text-geo-text">{top.name}</span>
      {" "}(score {top.p.toFixed(3)})
    </p>
  );
}

function RegressionBody({ meta }: { meta: OlmoEarthInferenceResult }) {
  if (!isRegressionLegend(meta.legend)) return null;
  const l = meta.legend;
  return (
    <div className="px-3 py-2 space-y-1.5">
      <p className="text-[11px] text-geo-muted">{l.label}</p>
      <GradientBar stops={l.stops} />
      {typeof l.value === "number" && (
        <p className="text-sm font-mono text-geo-text">
          {l.value.toFixed(2)}
          {l.units ? <span className="text-geo-muted"> {l.units}</span> : null}
        </p>
      )}
    </div>
  );
}

function EmbeddingBody({ meta }: { meta: OlmoEarthInferenceResult }) {
  const stops = meta.legend && "stops" in meta.legend ? meta.legend.stops : null;
  return (
    <div className="px-3 py-2 space-y-1.5">
      <p className="text-[11px] text-geo-muted">
        Encoder feature magnitude · PCA first component
      </p>
      {stops && <GradientBar stops={stops} />}
      {typeof meta.embedding_dim === "number" && (
        <p className="text-[10px] font-mono text-geo-muted">
          embedding dim: {meta.embedding_dim}
        </p>
      )}
    </div>
  );
}

function ExportRow({ layer }: { layer: ImageryLayer }) {
  const meta = layer.inferenceMetadata!;
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const isClassification =
    meta.task_type === "classification" || meta.task_type === "segmentation";
  if (!isClassification || meta.kind === "stub") return null;

  const handleGeoJson = async () => {
    setBusy(true);
    setMsg(null);
    try {
      // date_range + event_date aren't echoed back on the inference result; the
      // backend resolves them from the cached job when omitted, so we pass
      // undefined and let the server pick the matching cached run.
      const res = await downloadFtClassificationGeoJson({
        bbox: meta.bbox,
        modelRepoId: meta.model_repo_id,
      });
      setMsg(
        res.featureCount != null
          ? `${res.filename} · ${res.featureCount} polygons`
          : res.filename,
      );
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e);
      setMsg(`Failed: ${raw.slice(0, 120)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border-t border-geo-border bg-geo-bg px-2.5 py-2">
      <div className="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={handleGeoJson}
          disabled={busy}
          className="h-7 inline-flex items-center justify-center gap-1.5 text-[12px] font-medium border border-geo-border rounded bg-geo-surface text-geo-text hover:bg-geo-elevated disabled:opacity-50 disabled:cursor-wait"
        >
          <GeoJsonIcon />
          {busy ? "Exporting…" : "GeoJSON"}
        </button>
        <button
          type="button"
          disabled
          title="Per-class GeoTIFF export — not yet wired."
          className="h-7 inline-flex items-center justify-center gap-1.5 text-[12px] font-medium border border-geo-border rounded bg-geo-surface text-geo-dim opacity-50 cursor-not-allowed"
        >
          GeoTIFF
        </button>
      </div>
      {msg && (
        <p className="mt-1.5 text-[10.5px] text-geo-muted font-mono break-all">{msg}</p>
      )}
    </div>
  );
}

function LegendCard({ layer, onRemove }: { layer: ImageryLayer; onRemove?: (id: string) => void }) {
  const meta = layer.inferenceMetadata!;
  const isStub = meta.kind === "stub";
  const cov = sceneCoverageRow(meta);

  return (
    <article
      data-testid="inference-legend-popover"
      data-stub={isStub || undefined}
      className={
        "w-[280px] rounded-md overflow-hidden border shadow-geo-float bg-geo-surface " +
        (isStub ? "border-amber-300" : "border-geo-border")
      }
    >
      {/* Header */}
      <header className="flex items-start gap-2 px-3 py-2.5 border-b border-geo-border">
        <div className="flex-1 min-w-0">
          <h3 className="text-[13px] font-bold tracking-tight text-geo-text leading-tight font-display truncate" title={meta.model_repo_id}>
            {prettyModelName(meta.model_repo_id)}
          </h3>
          <p className="mt-0.5 text-[11px] text-geo-muted truncate">
            {taskSubtitle(meta)}
          </p>
        </div>
        {onRemove && (
          <CloseButton onClick={() => onRemove(layer.id)} title="Remove this inference layer" />
        )}
      </header>

      {/* Stub warning */}
      {isStub && (
        <div
          role="alert"
          data-testid="inference-stub-banner"
          className="flex items-start gap-2 px-3 py-2 text-[11px] text-amber-900 bg-amber-50 border-b border-amber-300"
        >
          <span className="text-[13px] leading-none flex-shrink-0">⚠</span>
          <div className="min-w-0">
            <span className="font-semibold">Synthetic raster — real inference failed.</span>
            {meta.stub_reason && (
              <span className="block mt-0.5 text-amber-800">{meta.stub_reason}</span>
            )}
          </div>
        </div>
      )}

      {/* Body */}
      {!isStub && isClassLegend(meta.legend) && (
        <>
          <ClassList classes={meta.legend.classes} />
          <ClassExtras meta={meta} />
          {meta.class_names_tentative && (
            <p className="px-3 pb-1 text-[10px] italic text-geo-muted">
              class names tentative — unverified against published labels
            </p>
          )}
        </>
      )}
      {!isStub && isRegressionLegend(meta.legend) && <RegressionBody meta={meta} />}
      {!isStub && meta.task_type === "embedding" && <EmbeddingBody meta={meta} />}

      {/* Coverage row */}
      {cov && (
        <div className="px-3 py-1.5 border-t border-geo-border-soft flex items-center justify-between text-[11px]">
          <span className="font-display font-bold tracking-wider uppercase text-[9.5px] text-geo-muted">
            {cov.label}
          </span>
          <span className="font-mono tabular-nums text-geo-text">{cov.value}</span>
        </div>
      )}

      <LegendNote text={meta.legend && "note" in meta.legend ? meta.legend.note : undefined} />

      {/* Export buttons */}
      <ExportRow layer={layer} />
    </article>
  );
}

export function InferenceLegendPanel({ imageryLayers, onRemove }: Props) {
  const inferLayers = imageryLayers.filter((l) => l.inferenceMetadata);
  if (inferLayers.length === 0) return null;

  return (
    <div
      data-testid="inference-legend-panel"
      className="absolute top-24 right-6 z-10 flex flex-col gap-2 max-h-[calc(100vh-160px)] overflow-y-auto pointer-events-auto"
    >
      {inferLayers.map((l) => (
        <LegendCard key={l.id} layer={l} onRemove={onRemove} />
      ))}
    </div>
  );
}
