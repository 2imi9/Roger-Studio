import { useState } from "react";
import {
  loadOlmoEarthRepo,
  startOlmoEarthInference,
  type OlmoEarthRepoStatus,
  type OlmoEarthInferenceResult,
} from "../api/client";
import type { BBox } from "../types";
import type { ImageryLayer } from "./MapView";

/**
 * Shared OlmoEarth / OlmoEarth-FT import + run-to-layer form.
 *
 * Used in two places (single source of truth):
 *   1. MapView's Added Layer popover → "Import" tab — quick in-map access.
 *   2. Sidebar's Map → Import Data sub-view — alongside DataUpload so users
 *      who live in the sidebar see the OE import path in context.
 *
 * The user's intent: pick an FT head (or base encoder) and produce a RASTER
 * LAYER on the map — not merely download weights to cache. So the primary
 * action is `Run + add to map`, which fires ``startOlmoEarthInference`` for
 * the selected area and pushes the returned tile_url back as a new
 * ImageryLayer via `onAddImageryLayer`. A secondary `Load weights only`
 * action remains for users who want to pre-warm the cache before running
 * inference (useful on a slow network where you'd rather download once
 * and try the model later).
 *
 * Model catalog here mirrors backend ``olmoearth_inference._COLORMAPS`` —
 * 5 FT heads + 4 base encoders. Adding a new head requires bumping both
 * places (the backend routes colormaps by repo_id; this list surfaces it
 * in the UI).
 */

interface ModelOption {
  repoId: string;
  label: string;
  kind: "ft" | "base";
  task: string;
  /** False when the backend FT-loader
   * (``backend/app/services/olmoearth_ft._infer_head_spec``) doesn't yet
   * recognize this checkpoint's decoder shape. The UI lets you see the
   * entry but blocks the Run action with a clear explanation — otherwise
   * users hit a cryptic "preview stub — ValueError: no recognized FT
   * head found" banner after waiting 30 s for inference. When loader
   * support lands, flip to true. LFMC ships a 6-layer Conv3×3 stack
   * (``model.decoder.0.layers.0.{0,3,5,8,10,12}.weight``) that the three
   * flat-Linear / Conv1×1 patterns the loader handles today don't cover.
   * ForestLossDriver marked unknown until we've verified the pattern —
   * cache it locally and add to the pattern-detector if the demo pair
   * lands in a stub. */
  supported: boolean;
  unsupportedReason?: string;
}

const FT_HEADS: ModelOption[] = [
  {
    repoId: "allenai/OlmoEarth-v1-FT-LFMC-Base",
    label: "LFMC",
    kind: "ft",
    task: "Live fuel moisture (regression)",
    // Conv-stack regression loader wired 2026-04-21. LFMC now produces
    // per-pixel moisture rasters at encoder-patch resolution instead of
    // the prior "no recognized FT head" stub.
    supported: true,
  },
  {
    repoId: "allenai/OlmoEarth-v1-FT-Mangrove-Base",
    label: "Mangrove",
    kind: "ft",
    task: "Mangrove extent (classification)",
    supported: true,
  },
  {
    repoId: "allenai/OlmoEarth-v1-FT-AWF-Base",
    label: "AWF land-use",
    kind: "ft",
    task: "Southern-Kenya land-use (classification)",
    supported: true,
  },
  {
    repoId: "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base",
    label: "Forest-loss driver",
    kind: "ft",
    task: "Driver classification (pre/post S2 pair)",
    // Loader wired 2026-04-21 (conv_pool_fc_classification head).
    // Weights load cleanly — the blocker is upstream: the decoder
    // expects 1536-channel input = concatenation of pre-event + post-
    // event Sentinel-2 feature maps (768 × 2). Our current inference
    // pipeline fetches a single S2 composite and can't produce the
    // required pair. Gated until we add pre/post date_range support.
    supported: false,
    unsupportedReason:
      "ForestLossDriver is a pre/post change-detection head — it expects two Sentinel-2 scenes (before + after the forest-loss event) concatenated along the feature dim. The weights load fine via the new conv_pool_fc loader, but the inference pipeline only supports a single scene today. Needs pipeline support for paired dates.",
  },
  {
    repoId: "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base",
    label: "Ecosystem type",
    kind: "ft",
    task: "110-class argmax",
    supported: true,
  },
];

const BASE_ENCODERS: ModelOption[] = [
  // Base encoders produce raw embedding features (no task head), which the
  // inference pipeline renders via a PCA-on-first-component gradient.
  // Works today for all four sizes — same forward path as the encoder
  // step inside the FT inference.
  { repoId: "allenai/OlmoEarth-v1-Nano", label: "Nano", kind: "base", task: "Smallest encoder · embedding output", supported: true },
  { repoId: "allenai/OlmoEarth-v1-Tiny", label: "Tiny", kind: "base", task: "Small encoder · embedding output", supported: true },
  { repoId: "allenai/OlmoEarth-v1-Base", label: "Base", kind: "base", task: "Medium encoder · embedding output", supported: true },
  { repoId: "allenai/OlmoEarth-v1-Large", label: "Large", kind: "base", task: "Largest encoder · embedding output", supported: true },
];

export function OlmoEarthImport({
  olmoCache,
  compact,
  initialRepoId,
  selectedArea,
  onAddImageryLayer,
}: {
  olmoCache?: Record<string, OlmoEarthRepoStatus>;
  /** Render without outer Panel chrome (popover usage). */
  compact?: boolean;
  /** Pre-select a repo id (e.g. OlmoEarth tab's "import" button passes
   * the clicked repo in so the Import tab opens with the right selection). */
  initialRepoId?: string;
  /** Current AOI. Required for `Run + add to map` — disabled until the
   * user has drawn a rectangle or polygon on the map. */
  selectedArea: BBox | null;
  /** Attach the inference result as an ImageryLayer. Wired straight to
   * App.handleAddImageryLayer — App's imageryLayers render on MapView. */
  onAddImageryLayer?: (layer: ImageryLayer) => void;
}) {
  // Default selection: first FT head unless initialRepoId matched something.
  const allOptions = [...FT_HEADS, ...BASE_ENCODERS];
  const initialOption =
    allOptions.find((m) => m.repoId === initialRepoId) ?? FT_HEADS[0];
  const [repoId, setRepoId] = useState(initialOption.repoId);
  const [hfToken, setHfToken] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState<null | "infer" | "cache">(null);

  const selected = allOptions.find((m) => m.repoId === repoId) ?? FT_HEADS[0];
  const live = olmoCache?.[repoId];
  const cached = live?.status === "cached";

  const handleRun = async () => {
    if (!selectedArea) return;
    if (!selected.supported) return;
    setBusy("infer");
    setStatus(null);
    try {
      const res: OlmoEarthInferenceResult = await startOlmoEarthInference({
        bbox: selectedArea,
        modelRepoId: repoId,
      });
      if (onAddImageryLayer) {
        onAddImageryLayer({
          id: `olmoearth-${res.job_id}`,
          tileUrl: res.tile_url,
          label: `${selected.label} · ${res.job_id.slice(0, 6)}`,
          inferenceMetadata: res,
        });
        setStatus(
          res.kind === "stub"
            ? `preview stub — real inference failed (${res.stub_reason ?? "unknown"})`
            : "added to map — check the On map tab",
        );
      } else {
        setStatus(`done — tile URL ready (job ${res.job_id})`);
      }
    } catch (e) {
      setStatus(`failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  const handleLoadCache = async () => {
    setBusy("cache");
    setStatus(null);
    try {
      const res = await loadOlmoEarthRepo({
        repoId,
        repoType: "model",
        hfToken: hfToken.trim() || undefined,
      });
      setStatus(res.error ? `error: ${res.error}` : "queued — watch status below");
    } catch (e) {
      setStatus(`failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  const body = (
    <div className="space-y-3 text-[12px]">
      <div>
        <div className="font-semibold text-geo-text mb-1">
          Import OlmoEarth / OlmoEarth-FT data
        </div>
        <div className="text-[10px] text-geo-muted leading-snug">
          Pick an OlmoEarth FT head or base encoder. <b>Run + add to map</b>{" "}
          executes inference over your currently-selected area and adds the
          resulting raster as a map layer. <b>Load weights only</b> just
          caches the model on disk for later.
        </div>
      </div>

      <div>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-geo-muted">
          Model
        </span>
        <select
          value={repoId}
          onChange={(e) => setRepoId(e.target.value)}
          disabled={busy !== null}
          className="w-full mt-0.5 px-2 py-1.5 text-[12px] bg-geo-surface border border-geo-border rounded focus:border-geo-accent focus:outline-none cursor-pointer"
        >
          <optgroup label="Fine-tuned heads (task-specific)">
            {FT_HEADS.map((m) => (
              <option key={m.repoId} value={m.repoId}>
                {m.label} — {m.task}
                {!m.supported ? " (loader not supported yet)" : ""}
              </option>
            ))}
          </optgroup>
          <optgroup label="Base encoders (embedding output)">
            {BASE_ENCODERS.map((m) => (
              <option key={m.repoId} value={m.repoId}>
                {m.label} — {m.task}
              </option>
            ))}
          </optgroup>
        </select>
        <div className="mt-1 text-[10px] text-geo-muted font-mono truncate" title={selected.repoId}>
          {selected.repoId}
        </div>
        {/* Support banner. Loader gaps are honest up-front instead of
            hiding them and letting the user wait 30 s for a stub. When
            the loader is extended, flip ``supported: true`` in the
            registry above and this banner disappears automatically. */}
        {!selected.supported && (
          <div className="mt-2 px-2 py-1.5 rounded bg-amber-50 border border-amber-300 text-amber-900 text-[10px] leading-snug">
            <div className="font-bold mb-0.5">Loader not yet supported</div>
            <div>{selected.unsupportedReason}</div>
          </div>
        )}
        {/* Coverage hint — FT heads have geographical training-distribution
            bias even though the Sentinel-2 + encoder stack works globally.
            Keyed off the repo id so the note reflects the SELECTED model,
            not a generic disclaimer. */}
        {selected.kind === "ft" && (
          <div className="mt-1 text-[9px] text-geo-muted italic">
            FT heads run globally on S2, but class labels / regression targets
            reflect each head's training distribution (Mangrove → tropical
            belt, AWF → southern Kenya, LFMC → fire-prone regions,
            ForestLossDriver → pantropical). Interpret outputs accordingly.
          </div>
        )}
      </div>

      {/* AOI status — the Run button needs a selected area. Surfacing this
          inline (rather than letting Run just no-op) avoids the "I clicked
          the button and nothing happened" reaction. */}
      <div className="px-2 py-1.5 rounded bg-geo-bg border border-geo-border text-[10px] leading-snug">
        <div className="text-geo-muted font-semibold uppercase tracking-wider">
          Area of interest
        </div>
        {selectedArea ? (
          <div className="text-geo-text font-mono">
            {selectedArea.west.toFixed(3)}, {selectedArea.south.toFixed(3)} → {selectedArea.east.toFixed(3)},{" "}
            {selectedArea.north.toFixed(3)}
          </div>
        ) : (
          <div className="text-geo-danger">
            No area selected — draw a rectangle or polygon on the map first.
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={handleRun}
        disabled={busy !== null || !selectedArea || !selected.supported}
        className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
          busy !== null || !selectedArea || !selected.supported
            ? "border-geo-border text-geo-muted cursor-not-allowed"
            : "border-geo-accent bg-geo-accent text-white hover:bg-geo-accent-hover cursor-pointer"
        }`}
        title={
          !selected.supported
            ? "This FT head's decoder shape isn't supported by the loader yet. Pick a supported head (Mangrove, AWF, Ecosystem, or any base encoder)."
            : selectedArea
              ? "Run inference on the selected area and add the result as a map layer"
              : "Draw an area on the map first"
        }
      >
        {busy === "infer" ? "Running inference…" : "Run + add to map"}
      </button>

      {/* Secondary: load-weights-only. Smaller / less prominent styling so
          it's clearly the advanced-user path, not the default. */}
      <details className="border border-geo-border rounded">
        <summary className="px-2 py-1.5 text-[11px] cursor-pointer hover:bg-geo-bg/60">
          Advanced: load weights only
        </summary>
        <div className="p-2 space-y-2 border-t border-geo-border">
          <label className="block">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-geo-muted">
              HF token{" "}
              <span className="font-normal normal-case">(only for gated repos)</span>
            </span>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="w-full mt-0.5 px-2 py-1.5 text-[11px] font-mono bg-geo-surface border border-geo-border rounded focus:border-geo-accent focus:outline-none"
              disabled={busy !== null}
            />
          </label>
          <button
            type="button"
            onClick={handleLoadCache}
            disabled={busy !== null}
            className={`w-full px-3 py-1.5 text-[11px] font-semibold rounded border transition-colors ${
              busy !== null
                ? "border-geo-border text-geo-muted cursor-not-allowed"
                : "border-geo-border text-geo-text hover:border-geo-accent hover:text-geo-accent cursor-pointer"
            }`}
          >
            {busy === "cache" ? "Queuing…" : "Load weights to disk"}
          </button>
        </div>
      </details>

      {/* Live cache status from the 2s /olmoearth/cache-status poll. */}
      {live && (
        <div
          className={`px-2 py-1.5 rounded border text-[10px] leading-snug ${
            cached
              ? "bg-geo-success/10 border-geo-success/40 text-geo-success"
              : live.status === "error"
                ? "bg-geo-danger/10 border-geo-danger/40 text-geo-danger"
                : "bg-geo-accent/10 border-geo-accent/40 text-geo-accent"
          }`}
        >
          <div className="font-semibold uppercase tracking-wider opacity-80">
            Cache status
          </div>
          <div>
            {cached
              ? `cached · ${
                  live.size_bytes
                    ? `${(live.size_bytes / 1_000_000).toFixed(0)} MB`
                    : "disk"
                } · ready to run`
              : live.status === "loading"
                ? "downloading from HuggingFace…"
                : live.status === "error"
                  ? `error: ${live.error ?? "unknown"}`
                  : String(live.status)}
          </div>
        </div>
      )}

      {status && (
        <div
          className={`text-[10px] leading-snug ${
            status.startsWith("error") || status.startsWith("failed") || status.startsWith("preview stub")
              ? "text-geo-danger"
              : "text-geo-text"
          }`}
        >
          {status}
        </div>
      )}
    </div>
  );

  if (compact) return body;
  return (
    <div className="bg-geo-surface border border-geo-border rounded-lg p-4">
      {body}
    </div>
  );
}
