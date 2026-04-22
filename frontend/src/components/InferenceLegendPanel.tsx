/**
 * Renders legends for one or more active OlmoEarth inference layers.
 *
 * The backend packs a per-task ``legend`` block into the /api/olmoearth/infer
 * response — classification/segmentation jobs carry per-class hex colors
 * sourced from olmoearth_projects' rslearn configs, regression jobs carry a
 * gradient stop list + the predicted value in task units.
 *
 * This panel reads ``imageryLayers`` and renders a compact card for each
 * inference layer, so users can tell at a glance what each map overlay
 * means without digging through a separate modal.
 */
import type { ImageryLayer } from "./MapView";
import type { OlmoEarthInferenceResult, OlmoEarthLegend } from "../api/client";

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

function sceneLine(meta: OlmoEarthInferenceResult): string | null {
  if (!meta.scene_id) return null;
  const when = meta.scene_datetime?.split("T")[0] ?? "";
  const cc =
    typeof meta.scene_cloud_cover === "number"
      ? ` · ${meta.scene_cloud_cover.toFixed(1)}% cloud`
      : "";
  return `${meta.scene_id.slice(0, 24)}… · ${when}${cc}`;
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

/** Small honesty footnote that surfaces the backend's calibration /
 *  interpretation guidance to the user. Empty string → render nothing.
 *  Styled as italic muted text so it reads as a caveat, not a header.
 *  Covers the audit-flagged scientific-credibility risk: previous legend
 *  copy ("Mangrove probability") implied a calibrated 0-1 Bayesian
 *  probability when the value is actually a raw uncalibrated softmax
 *  score. Each task now ships a ``note`` explaining what the number
 *  actually means. */
function LegendNote({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <p className="text-[10px] italic text-geo-muted leading-snug">
      {text}
    </p>
  );
}

function GradientBar({ stops }: { stops: [string, number][] }) {
  const stopStr = stops
    .map(([hex, pos]) => `${hex} ${(pos * 100).toFixed(0)}%`)
    .join(", ");
  return (
    <div
      className="h-2 rounded-full"
      style={{ backgroundImage: `linear-gradient(to right, ${stopStr})` }}
      role="img"
      aria-label="colormap gradient"
    />
  );
}

function LegendBody({ meta }: { meta: OlmoEarthInferenceResult }) {
  if (meta.task_type === "embedding") {
    const note = meta.legend && "note" in meta.legend ? meta.legend.note : undefined;
    return (
      <div className="space-y-2">
        <p className="text-[11px] text-geo-muted">
          Encoder feature magnitude · PCA first component
        </p>
        {meta.legend && "stops" in meta.legend && (
          <GradientBar stops={meta.legend.stops} />
        )}
        {typeof meta.embedding_dim === "number" && (
          <p className="text-[11px] font-mono text-geo-muted">
            embedding dim: {meta.embedding_dim}
          </p>
        )}
        <LegendNote text={note} />
      </div>
    );
  }

  if (isRegressionLegend(meta.legend)) {
    const l = meta.legend;
    return (
      <div className="space-y-2">
        <p className="text-[11px] text-geo-muted">{l.label}</p>
        <GradientBar stops={l.stops} />
        {typeof l.value === "number" && (
          <p className="text-sm font-mono">
            {l.value.toFixed(2)}
            {l.units ? <span className="text-geo-muted"> {l.units}</span> : null}
          </p>
        )}
        <LegendNote text={l.note} />
      </div>
    );
  }

  if (isClassLegend(meta.legend)) {
    const top =
      meta.class_probs && meta.class_names
        ? [...meta.class_probs]
            .map((p, i) => ({ i, p, name: meta.class_names![i] ?? `class_${i}` }))
            .sort((a, b) => b.p - a.p)[0]
        : null;
    return (
      <div className="space-y-2">
        {top && (
          // Previously displayed as a percentage ("83.2%") which users
          // read as "83% likely mangrove". It's actually a raw softmax
          // score — kept the numeric display but dropped the "%" sign
          // and relabeled so the reader sees "score" not "probability".
          // The LegendNote below reinforces the interpretation.
          <p className="text-[11px] text-geo-muted">
            Top class:{" "}
            <span className="font-mono text-geo-text">{top.name}</span>{" "}
            (score {top.p.toFixed(3)})
          </p>
        )}
        <ul className="space-y-1 max-h-40 overflow-y-auto pr-1">
          {meta.legend.classes.map((c) => (
            <li
              key={c.index}
              className="flex items-center gap-2 text-[11px]"
            >
              <span
                className="inline-block w-3 h-3 rounded-sm border border-geo-border flex-shrink-0"
                style={{ backgroundColor: c.color }}
              />
              <span className="font-mono text-geo-muted flex-shrink-0">
                {c.index}
              </span>
              <span className="truncate" title={c.name}>
                {c.name}
              </span>
            </li>
          ))}
        </ul>
        {meta.class_names_tentative && (
          <p className="text-[10px] italic text-geo-muted">
            class names tentative — unverified against published labels
          </p>
        )}
        <LegendNote text={meta.legend.note} />
      </div>
    );
  }

  // Stub fallback — no legend, just render the reason.
  return (
    <p className="text-[11px] italic text-amber-600">
      Stub output — {meta.stub_reason ?? "unknown reason"}
    </p>
  );
}

export function InferenceLegendPanel({ imageryLayers, onRemove }: Props) {
  const inferLayers = imageryLayers.filter((l) => l.inferenceMetadata);
  if (inferLayers.length === 0) return null;

  return (
    <section
      className="bg-gradient-panel rounded-xl p-4 border border-geo-border shadow-soft space-y-3"
      data-testid="inference-legend-panel"
    >
      <header className="flex items-center justify-between">
        <h3 className="text-[11px] uppercase tracking-wider text-geo-muted">
          OlmoEarth inference
        </h3>
        <span className="text-[10px] font-mono text-geo-muted">
          {inferLayers.length} active
        </span>
      </header>
      <div className="space-y-3">
        {inferLayers.map((l) => {
          const meta = l.inferenceMetadata!;
          const taskBadge = meta.task_type ?? meta.kind;
          const isStub = meta.kind === "stub";
          return (
            <article
              key={l.id}
              // Amber-tinted card when the backend fell back to stub mode
              // (S2 fetch failed, FT head state-dict didn't match, etc.).
              // Makes it visually obvious the raster is SYNTHETIC so the
              // user doesn't mistake it for real model output.
              className={
                "rounded-lg border p-3 space-y-2 " +
                (isStub
                  ? "bg-amber-50 border-amber-300"
                  : "bg-geo-surface border-geo-border")
              }
              data-stub={isStub || undefined}
            >
              {/* Prominent stub banner — unmissable at a glance, points the
                  user at the actual failure reason so they know WHY the
                  real pipeline didn't run (bad AOI, S2 scene 0-hit,
                  unrecognized FT head). Previously the only signal was a
                  small "stub" pill in the corner which users read past. */}
              {isStub && (
                <div
                  className="flex items-start gap-2 rounded-md bg-amber-100 border border-amber-300 px-2 py-1.5 text-[11px] text-amber-900"
                  role="alert"
                  data-testid="inference-stub-banner"
                >
                  <span className="text-[13px] leading-none flex-shrink-0">⚠</span>
                  <div className="min-w-0">
                    <span className="font-semibold">Stub fallback — this raster is synthetic.</span>{" "}
                    <span className="text-amber-800">
                      {meta.stub_reason
                        ? `Real inference failed: ${meta.stub_reason}`
                        : "Real inference failed; a placeholder colormap is shown so you can see extent."}
                    </span>
                  </div>
                </div>
              )}
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs font-mono truncate" title={meta.model_repo_id}>
                    {prettyModelName(meta.model_repo_id)}
                  </p>
                  {sceneLine(meta) && (
                    <p
                      className="text-[10px] text-geo-muted font-mono truncate"
                      title={meta.scene_id ?? undefined}
                    >
                      {sceneLine(meta)}
                    </p>
                  )}
                </div>
                <span
                  className={
                    "text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded flex-shrink-0 " +
                    (isStub
                      ? "bg-amber-200 text-amber-800 border border-amber-300"
                      : "bg-geo-accent/10 text-geo-accent")
                  }
                >
                  {isStub ? "stub" : taskBadge}
                </span>
              </div>
              <LegendBody meta={meta} />
              {onRemove && (
                <button
                  type="button"
                  onClick={() => onRemove(l.id)}
                  className="text-[10px] uppercase tracking-wider text-geo-muted hover:text-red-600"
                >
                  remove layer
                </button>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
