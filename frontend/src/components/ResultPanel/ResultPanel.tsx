/**
 * Result Interpretation Panel — replaces the input form in the
 * OlmoEarth import sidebar after an inference completes, summarizing
 * what the model produced in human-readable terms.
 *
 * Ported from Claude Design v2 component1 (2026-04-26). Six states
 * collapse into ONE polymorphic component that branches on the
 * OlmoEarthInferenceResult shape:
 *
 *   * State 1 (Pristine classification) — task_type ∈ {classification,
 *       segmentation}, kind="pytorch", AOI inside training region
 *   * State 2 (Regression) — task_type="regression"
 *   * State 3 (Partial coverage) — backend ``notes`` mentions failed
 *       chunks, or a "<n>/<m> chunks" count appears in the result
 *   * State 4 (Off-distribution) — classification + AOI outside
 *       head's training region per HEAD_TRAINING_REGIONS
 *   * State 5 (Stub fallback) — kind="stub", surfaces stub_reason
 *   * State 6 (History) — out of scope for v1. The recent-inference
 *       caching + history strip would belong here in a future pass.
 *
 * The class summary block is sortable (% AOI desc by default, or
 * alphabetical), with empty classes collapsed under a disclosure.
 * Each row shows a colored swatch + class name + class index. We
 * compute %-of-AOI from class_probs when available; for sliding-
 * window jobs the backend may return per-window counts in the
 * future, but for now the scene-level class_probs gives a usable
 * proxy.
 *
 * Provenance footer surfaces: model repo id + version, scene id(s)
 * the inference ran over, scene datetime, sliding-window settings.
 * Wired entirely off the existing OlmoEarthInferenceResult shape;
 * no new endpoints required.
 */
import { useState } from "react";
import type {
  OlmoEarthInferenceResult,
  OlmoEarthLegend,
} from "../../api/client";
import { downloadFtClassificationGeoJson } from "../../api/client";
import type { BBox } from "../../types";
import {
  HEAD_TRAINING_REGIONS,
  isAoiInsideTrainingRegion,
} from "../../constants/headTrainingRegions";

interface Props {
  result: OlmoEarthInferenceResult;
  /** When the user clicks "← Back to input", flip OlmoEarthImport
   *  back to the form mode. */
  onBackToInput: () => void;
  /** Time the inference finished (ms epoch). Used to render the
   *  "ran X ago" pill in the panel header. */
  ranAt: number;
  /** Wall time spent in the run, in ms. */
  tookMs: number;
}

interface ClassRow {
  index: number;
  name: string;
  color: string;
  /** softmax score, when available */
  prob: number | null;
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

function prettyModelName(repoId: string): string {
  return repoId
    .replace(/^allenai\//, "")
    .replace(/^OlmoEarth-v1-FT-/, "")
    .replace(/^OlmoEarth-v1-/, "");
}

function formatRanAgo(ranAt: number, now: number): string {
  const diffSec = Math.max(0, Math.round((now - ranAt) / 1000));
  if (diffSec < 5) return "just now";
  if (diffSec < 60) return `${diffSec} s ago`;
  const m = Math.floor(diffSec / 60);
  const s = diffSec % 60;
  if (m < 60) return s ? `${m} min ${s} s ago` : `${m} min ago`;
  return `${Math.floor(m / 60)} h ago`;
}

function formatTook(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}

function bboxKm(b: BBox): { ew: number; ns: number } {
  // Quick-and-dirty: 1° lat ≈ 111 km, 1° lon ≈ 111 cos(lat) km.
  const midLat = (b.south + b.north) / 2;
  const ew = Math.round(Math.abs(b.east - b.west) * 111 * Math.cos((midLat * Math.PI) / 180));
  const ns = Math.round(Math.abs(b.north - b.south) * 111);
  return { ew, ns };
}

function StatusPill({ tone, children }: { tone: "success" | "warn" | "danger"; children: React.ReactNode }) {
  const cls =
    tone === "success" ? "bg-geo-success-soft text-geo-success border-geo-success/30"
    : tone === "warn" ? "bg-geo-warn-soft text-geo-warn border-geo-warn/30"
    : "bg-geo-danger-soft text-geo-danger border-geo-danger/30";
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-display font-bold text-[11px] tracking-wide border ${cls}`}>
      {children}
    </span>
  );
}

function SectionLabel({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between px-4 mb-2">
      <span className="font-display font-bold text-[11px] uppercase tracking-wider text-geo-muted">
        {children}
      </span>
      {right}
    </div>
  );
}

function ClassSummary({ rows, hasMutedTreatment = false }: { rows: ClassRow[]; hasMutedTreatment?: boolean }) {
  const [emptyOpen, setEmptyOpen] = useState(false);
  const [sort, setSort] = useState<"pct" | "name">("pct");

  const present = rows.filter((r) => (r.prob ?? 0) > 0);
  const empty = rows.filter((r) => !((r.prob ?? 0) > 0));
  const sorted = [...present].sort((a, b) =>
    sort === "pct"
      ? (b.prob ?? 0) - (a.prob ?? 0)
      : a.name.localeCompare(b.name),
  );

  const Row = ({ row, dim }: { row: ClassRow; dim?: boolean }) => (
    <div
      className={`grid items-center gap-2 px-2 py-1 ${dim ? "opacity-40" : ""}`}
      style={{ gridTemplateColumns: "14px 1fr 70px 52px" }}
    >
      <span
        className="w-3 h-3 rounded-sm border border-geo-border flex-shrink-0"
        style={{ backgroundColor: row.color }}
      />
      <span className="text-[12.5px] text-geo-text truncate" title={row.name}>
        {row.name}
      </span>
      <span className="text-right font-mono tabular-nums text-[11px] text-geo-muted">
        {row.prob != null ? row.prob.toFixed(3) : "—"}
      </span>
      <span className="text-right font-mono tabular-nums text-[12px] text-geo-text">
        {row.prob != null ? `${(row.prob * 100).toFixed(1)}%` : "—"}
      </span>
    </div>
  );

  return (
    <div className={`px-4 ${hasMutedTreatment ? "opacity-82" : ""}`}>
      <div
        className="grid gap-2 px-2 pb-1.5 mb-0.5 border-b border-geo-border-soft"
        style={{ gridTemplateColumns: "14px 1fr 70px 52px" }}
      >
        <span />
        <button
          type="button"
          onClick={() => setSort("name")}
          className={`text-left font-display font-bold text-[10px] uppercase tracking-wider cursor-pointer ${
            sort === "name" ? "text-geo-text" : "text-geo-muted"
          }`}
        >
          Class
        </button>
        <span className="text-right font-display font-bold text-[10px] uppercase tracking-wider text-geo-muted">
          Score
        </span>
        <button
          type="button"
          onClick={() => setSort("pct")}
          className={`text-right font-display font-bold text-[10px] uppercase tracking-wider cursor-pointer ${
            sort === "pct" ? "text-geo-text" : "text-geo-muted"
          }`}
        >
          % AOI
        </button>
      </div>
      {sorted.map((r) => <Row key={r.index} row={r} />)}
      {empty.length > 0 && (
        <div className="mt-1">
          <button
            type="button"
            onClick={() => setEmptyOpen(!emptyOpen)}
            className="px-2 py-1.5 text-[12px] text-geo-muted inline-flex items-center gap-1.5 hover:text-geo-text transition-colors"
          >
            <span className="text-[10px]">{emptyOpen ? "▾" : "▸"}</span>
            + {empty.length} empty class{empty.length === 1 ? "" : "es"}
          </button>
          {emptyOpen && empty.map((r) => <Row key={`e${r.index}`} row={r} dim />)}
        </div>
      )}
    </div>
  );
}

function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const tone = value >= 0.75 ? "success" : value >= 0.5 ? "warn" : "danger";
  const fillCls =
    tone === "success" ? "bg-geo-success" : tone === "warn" ? "bg-geo-warn" : "bg-geo-danger";
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-geo-muted">{label ?? "Top-class softmax score"}</span>
        <span className="font-display font-bold text-[16px] tabular-nums text-geo-text">
          {value.toFixed(2)}
        </span>
      </div>
      <div className="h-2 rounded-full bg-geo-elevated overflow-hidden">
        <div className={`h-full ${fillCls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function StubBlock({ result }: { result: OlmoEarthInferenceResult }) {
  return (
    <div className="px-4 space-y-3">
      <div className="rounded-md border border-geo-danger/35 bg-geo-danger-soft p-3 space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="text-geo-danger leading-none">⚠</span>
          <span className="font-display font-bold text-[14px] text-geo-danger">
            No real raster was produced
          </span>
        </div>
        <p className="text-[12px] text-geo-text/85 leading-snug">
          The map tile is a deterministic synthetic stub, not model output. The
          backend fell back because real inference failed — the reason is below.
        </p>
      </div>
      <div className="rounded border border-geo-border bg-geo-elevated p-2.5 font-mono text-[12px] leading-relaxed break-words">
        <span className="text-geo-muted">stub_reason: </span>
        <span className="text-geo-danger">"{result.stub_reason ?? "unknown"}"</span>
      </div>
      <p className="text-[11.5px] text-geo-muted leading-snug">
        Class summary, confidence, and export are unavailable for stub results.
      </p>
    </div>
  );
}

function PartialCoverageBanner({ note }: { note: string }) {
  return (
    <div className="mx-4 rounded-md border border-geo-warn/40 bg-geo-warn-soft px-3 py-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-geo-warn leading-none">⚠</span>
        <span className="font-display font-bold text-[13px] text-geo-warn">
          Partial coverage
        </span>
      </div>
      <p className="text-[12px] text-geo-text/85 leading-snug">{note}</p>
    </div>
  );
}

function ProvenanceFooter({ result }: { result: OlmoEarthInferenceResult }) {
  return (
    <div className="border-t border-geo-border bg-geo-bg/50 px-4 py-3">
      <div className="font-display font-bold text-[10px] uppercase tracking-wider text-geo-muted mb-1.5">
        Provenance
      </div>
      <dl className="grid grid-cols-[68px_1fr] gap-x-2 gap-y-1 text-[11px]">
        <dt className="text-geo-muted">Model</dt>
        <dd className="font-mono text-geo-text break-all">{result.model_repo_id}</dd>
        {result.scene_id && (
          <>
            <dt className="text-geo-muted">Scene</dt>
            <dd className="font-mono text-geo-text break-all">{result.scene_id}</dd>
          </>
        )}
        {result.scene_datetime && (
          <>
            <dt className="text-geo-muted">Date</dt>
            <dd className="font-mono tabular-nums text-geo-text">{result.scene_datetime.split("T")[0]}</dd>
          </>
        )}
        {typeof result.scene_cloud_cover === "number" && (
          <>
            <dt className="text-geo-muted">Cloud</dt>
            <dd className="font-mono tabular-nums text-geo-text">{result.scene_cloud_cover.toFixed(1)}%</dd>
          </>
        )}
        {result.sliding_window && (
          <>
            <dt className="text-geo-muted">Window</dt>
            <dd className="font-mono tabular-nums text-geo-text">
              {result.window_size ? `${result.window_size} px` : "sliding"}
            </dd>
          </>
        )}
        <dt className="text-geo-muted">Job</dt>
        <dd className="font-mono text-geo-text break-all">{result.job_id}</dd>
      </dl>
    </div>
  );
}

function ExportRow({ result }: { result: OlmoEarthInferenceResult }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const isClassification =
    result.task_type === "classification" || result.task_type === "segmentation";
  if (!isClassification || result.kind === "stub") return null;

  const handleGeoJson = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await downloadFtClassificationGeoJson({
        bbox: result.bbox,
        modelRepoId: result.model_repo_id,
      });
      setMsg(
        r.featureCount != null
          ? `${r.filename} · ${r.featureCount} polygons`
          : r.filename,
      );
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e);
      setMsg(`Failed: ${raw.slice(0, 120)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="px-4 space-y-2">
      <button
        type="button"
        onClick={handleGeoJson}
        disabled={busy}
        className="w-full h-9 inline-flex items-center justify-center gap-2 text-[13px] font-medium rounded bg-geo-accent text-white hover:bg-geo-accent-deep transition-colors disabled:opacity-60 disabled:cursor-wait shadow-geo-button"
      >
        {busy ? "Exporting…" : "Download as GeoJSON"}
      </button>
      <p className="text-[11px] text-geo-muted px-1 leading-snug">
        Polygons with class labels + area · opens in QGIS, ArcGIS, Google Earth.
      </p>
      <button
        type="button"
        disabled
        title="Per-class GeoTIFF export — not yet wired."
        className="w-full h-8 inline-flex items-center justify-center gap-2 text-[12px] font-medium rounded border border-geo-border bg-geo-surface text-geo-dim opacity-60 cursor-not-allowed"
      >
        Download as GeoTIFF · soon
      </button>
      {msg && (
        <p className="text-[10.5px] text-geo-muted font-mono break-all">{msg}</p>
      )}
    </div>
  );
}

function buildClassRows(result: OlmoEarthInferenceResult): ClassRow[] {
  if (!isClassLegend(result.legend)) return [];
  const probs = result.class_probs;
  const present = result.present_class_ids ?? null;
  return result.legend.classes.map((c) => {
    const prob = probs && probs[c.index] != null ? probs[c.index] : null;
    const isPresent = !present || present.includes(c.index);
    return {
      index: c.index,
      name: c.name,
      color: c.color,
      prob: isPresent ? prob : null,
    };
  });
}

function topClass(rows: ClassRow[]): { name: string; prob: number } | null {
  const sorted = [...rows].filter((r) => r.prob != null).sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
  const t = sorted[0];
  if (!t || t.prob == null) return null;
  return { name: t.name, prob: t.prob };
}

export function ResultPanel({ result, onBackToInput, ranAt, tookMs }: Props) {
  const now = Date.now();
  const isStub = result.kind === "stub";

  // Derive partial-coverage note from result.notes — backend appends
  // chunk-failure messages there. Not all responses carry it.
  const partialNote = (result.notes ?? []).find((n) =>
    /chunk|partial|failed|timeout|gateway/i.test(n),
  );
  const isPartial = !!partialNote && !isStub;

  // Off-distribution check using the same registry as Component 2.
  const region = HEAD_TRAINING_REGIONS[result.model_repo_id];
  const isOffDist = !!region && !isAoiInsideTrainingRegion(result.bbox, region.bbox);

  const isClassification =
    result.task_type === "classification" || result.task_type === "segmentation";
  const isRegression = result.task_type === "regression";

  const classRows = isClassification ? buildClassRows(result) : [];
  const top = topClass(classRows);

  const statusTone: "success" | "warn" | "danger" = isStub
    ? "danger"
    : isPartial || isOffDist
      ? "warn"
      : "success";
  const statusLabel = isStub
    ? "Stub fallback"
    : isPartial
      ? "Partial coverage"
      : isOffDist
        ? "Off-distribution"
        : "Real result";

  const km = bboxKm(result.bbox);

  return (
    <div data-testid="result-panel" className="flex flex-col h-full overflow-hidden border border-geo-border rounded-md bg-geo-surface shadow-geo-panel">
      {/* Back-to-input bar */}
      <button
        type="button"
        onClick={onBackToInput}
        className="flex-shrink-0 flex items-center gap-2 px-4 py-2 text-[12px] text-geo-muted hover:text-geo-text hover:bg-geo-elevated transition-colors border-b border-geo-border-soft cursor-pointer text-left"
      >
        <span>←</span>
        <span>Back to input</span>
      </button>

      {/* Panel header */}
      <header className="flex-shrink-0 px-4 py-3 border-b border-geo-border bg-geo-bg/40">
        <div className="flex items-start justify-between gap-2">
          <h3 className="font-display font-bold text-[16px] tracking-tight text-geo-text leading-tight" title={result.model_repo_id}>
            {prettyModelName(result.model_repo_id)}
          </h3>
          <StatusPill tone={statusTone}>{statusLabel}</StatusPill>
        </div>
        <p className="mt-1 text-[11.5px] text-geo-muted">
          AOI · <span className="font-mono tabular-nums">{km.ew} × {km.ns} km</span>
        </p>
        <p className="mt-0.5 text-[11px] text-geo-muted font-mono tabular-nums">
          ran {formatRanAgo(ranAt, now)} · took {formatTook(tookMs)}
        </p>
      </header>

      {/* Body */}
      <div className="flex-1 overflow-y-auto py-3 space-y-4">
        {isStub && <StubBlock result={result} />}

        {isPartial && partialNote && <PartialCoverageBanner note={partialNote} />}

        {!isStub && isClassification && (
          <section>
            <SectionLabel
              right={
                isOffDist ? (
                  <span className="text-[10.5px] italic text-geo-warn">
                    rendered at 80% saturation
                  </span>
                ) : null
              }
            >
              Class summary
            </SectionLabel>
            <ClassSummary rows={classRows} hasMutedTreatment={isOffDist} />
          </section>
        )}

        {!isStub && isClassification && top && (
          <section>
            <SectionLabel>Confidence</SectionLabel>
            <div className="px-4">
              <ConfidenceBar value={top.prob} label={`Top class · ${top.name}`} />
              {result.class_names_tentative && (
                <p className="mt-1 text-[10px] italic text-geo-muted">
                  class names tentative — unverified against published labels
                </p>
              )}
            </div>
          </section>
        )}

        {!isStub && isRegression && isRegressionLegend(result.legend) && (
          <section>
            <SectionLabel>Predicted value</SectionLabel>
            <div className="px-4 space-y-2">
              <p className="text-[12px] text-geo-muted">{result.legend.label}</p>
              {typeof result.legend.value === "number" && (
                <p className="font-display font-bold text-[28px] tabular-nums text-geo-text leading-none">
                  {result.legend.value.toFixed(2)}
                  {result.legend.units && (
                    <span className="ml-1 text-[14px] text-geo-muted font-medium">
                      {result.legend.units}
                    </span>
                  )}
                </p>
              )}
              {result.legend.note && (
                <p className="text-[11px] italic text-geo-muted leading-snug">
                  {result.legend.note}
                </p>
              )}
            </div>
          </section>
        )}

        {!isStub && (
          <section>
            <SectionLabel>Export</SectionLabel>
            <ExportRow result={result} />
          </section>
        )}

        <ProvenanceFooter result={result} />
      </div>
    </div>
  );
}
