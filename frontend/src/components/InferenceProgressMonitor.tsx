/**
 * Live inference progress monitor — fills the OlmoEarthImport sidebar
 * slot between Form (before Run) and ResultPanel (after Run completes).
 *
 * Renders:
 *   * Display-font model header + status pill ("Running")
 *   * Live elapsed timer (mono tabular nums, ticks every 250 ms)
 *   * Backend-supplied stage label + per-chunk message
 *   * Real progress bar (0–100%) driven by chunks_done / chunks_total
 *   * ETA (refines as chunks complete)
 *
 * Data flow:
 *   * Caller passes the deterministic ``jobId`` (resolved up-front via
 *     ``previewInferenceJobId``) and the start timestamp.
 *   * This component polls ``GET /jobs/{jobId}/progress`` every 1 s.
 *   * The first few polls may 404 — the orchestrator hasn't registered
 *     the job yet — those return null and the monitor falls back to
 *     "Starting…" until backend state lands.
 *   * Caller takes responsibility for unmounting once the long-running
 *     POST resolves. The monitor doesn't need to know when to stop.
 *
 * Visual language matches Result Panel / Off-Distribution Banner:
 * warm-amber while running, success-green soft on completion of the
 * last chunk before unmount, danger-red soft if all chunks fail.
 */
import { useEffect, useRef, useState } from "react";
import {
  getInferenceProgress,
  type InferenceProgress,
} from "../api/client";

interface Props {
  /** Deterministic backend job_id, obtained ahead of /infer via
   *  ``previewInferenceJobId`` so polling can start immediately. */
  jobId: string;
  /** Pretty model label for the header (e.g. "Mangrove"). */
  modelLabel: string;
  /** Full repo id, shown in muted mono under the label. */
  modelRepoId: string;
  /** Wall-clock start timestamp (ms epoch) — drives the live elapsed
   *  timer. The backend supplies its own elapsed_ms but client-side
   *  ticking is smoother and more honest about latency to the API. */
  startedAt: number;
  /** Optional: AOI dimensions to display in the header subline. */
  aoiKm?: { ew: number; ns: number };
}

const POLL_MS = 1000;
const TICK_MS = 250;

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}

function stageLabel(stage: string): string {
  switch (stage) {
    case "queued": return "Queued";
    case "resolving_scenes": return "Resolving scenes";
    case "processing_chunks": return "Processing chunks";
    case "stitching": return "Stitching tiles";
    default: return stage.replace(/_/g, " ");
  }
}

export function InferenceProgressMonitor({
  jobId,
  modelLabel,
  modelRepoId,
  startedAt,
  aoiKm,
}: Props) {
  const [elapsed, setElapsed] = useState<number>(() => Date.now() - startedAt);
  const [progress, setProgress] = useState<InferenceProgress | null>(null);
  const cancelledRef = useRef(false);

  // Smooth client-side elapsed tick — keeps the timer visibly moving even
  // when the backend poll is between intervals.
  useEffect(() => {
    cancelledRef.current = false;
    const id = setInterval(() => {
      if (!cancelledRef.current) setElapsed(Date.now() - startedAt);
    }, TICK_MS);
    return () => {
      cancelledRef.current = true;
      clearInterval(id);
    };
  }, [startedAt]);

  // Backend progress poll. Runs on a 1 s cadence; each poll is best-
  // effort — 404s and network blips are swallowed so the timer keeps
  // moving even when the API is unreachable for a beat.
  useEffect(() => {
    let stopped = false;
    const fetchOnce = async () => {
      try {
        const p = await getInferenceProgress(jobId);
        if (!stopped && p !== null) setProgress(p);
      } catch {
        // network / 5xx — keep ticking, retry next interval
      }
    };
    fetchOnce();
    const id = setInterval(fetchOnce, POLL_MS);
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, [jobId]);

  const total = progress?.chunks_total ?? 0;
  const done = progress?.chunks_done ?? 0;
  const failed = progress?.chunks_failed ?? 0;
  // Two parts to the percentage: a nominal "we've started" floor of 5%
  // before the first chunk lands, then linear in done/total. Without the
  // floor the bar is empty for the first 30+ s of any cold run while
  // scenes resolve, which feels broken.
  let pct = 0;
  if (total > 0) {
    pct = Math.max(5, Math.round((done / total) * 100));
  } else if (progress) {
    pct = 5;
  }
  const stage = progress ? stageLabel(progress.stage) : "Starting…";
  const message = progress?.message ?? "Talking to backend…";

  // ETA: prefer backend's estimate (uses observed per-chunk wall time),
  // fall back to "—" before chunks complete.
  const etaText =
    progress?.est_remaining_ms != null
      ? `≈ ${formatDuration(progress.est_remaining_ms)} remaining`
      : "calculating ETA…";

  // Tone for the progress bar — flips to amber when any chunk fails so
  // partial-coverage is signaled the moment it happens, even before
  // completion. Stays accent-blue otherwise.
  const fillCls =
    failed > 0 ? "bg-geo-warn" : "bg-geo-accent";
  const fillSoftCls =
    failed > 0 ? "bg-geo-warn-soft" : "bg-geo-elevated";

  return (
    <div
      data-testid="inference-progress-monitor"
      className="flex flex-col gap-3 p-4 border border-geo-border rounded-md bg-geo-surface shadow-geo-panel"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <h3
            className="font-display font-bold text-[15px] tracking-tight text-geo-text leading-tight"
            title={modelRepoId}
          >
            {modelLabel}
          </h3>
          <p className="mt-0.5 text-[11px] text-geo-muted font-mono truncate" title={modelRepoId}>
            {modelRepoId}
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-display font-bold text-[11px] tracking-wide border bg-geo-warn-soft text-geo-warn border-geo-warn/30">
          <span className="w-1.5 h-1.5 rounded-full bg-geo-warn animate-pulse" aria-hidden="true" />
          Running
        </span>
      </div>

      {/* Stage + message */}
      <div className="space-y-0.5">
        <div className="flex items-baseline justify-between">
          <span className="font-display font-bold text-[11px] uppercase tracking-wider text-geo-muted">
            {stage}
          </span>
          <span className="text-[11px] font-mono tabular-nums text-geo-muted">
            {pct}%
          </span>
        </div>
        <p className="text-[12px] text-geo-text/85 leading-snug">{message}</p>
      </div>

      {/* Progress bar */}
      <div className={`h-2.5 rounded-full overflow-hidden ${fillSoftCls}`}>
        <div
          className={`h-full ${fillCls} transition-[width] duration-300 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Chunk + timing breakdown */}
      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div className="space-y-0.5">
          <div className="font-display font-bold text-[10px] uppercase tracking-wider text-geo-muted">
            Chunks
          </div>
          <div className="font-mono tabular-nums text-geo-text">
            {done} / {total || "?"}
            {failed > 0 && (
              <span className="text-geo-warn"> · {failed} failed</span>
            )}
          </div>
        </div>
        <div className="space-y-0.5">
          <div className="font-display font-bold text-[10px] uppercase tracking-wider text-geo-muted">
            Elapsed
          </div>
          <div className="font-mono tabular-nums text-geo-text">
            {formatDuration(elapsed)}
          </div>
        </div>
        <div className="space-y-0.5">
          <div className="font-display font-bold text-[10px] uppercase tracking-wider text-geo-muted">
            ETA
          </div>
          <div className="font-mono tabular-nums text-geo-text/85">
            {etaText}
          </div>
        </div>
      </div>

      {/* Optional AOI footnote */}
      {aoiKm && (
        <p className="text-[10.5px] text-geo-muted font-mono tabular-nums border-t border-geo-border-soft pt-2">
          AOI · {aoiKm.ew} × {aoiKm.ns} km · job {jobId.slice(0, 8)}…
        </p>
      )}
    </div>
  );
}
