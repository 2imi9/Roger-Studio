import { useState } from "react";
import {
  downloadEmbeddingExport,
  downloadFtClassificationGeoJson,
  exportOlmoEarthEmbedding,
  loadOlmoEarthRepo,
  runOlmoEarthEmbeddingPCARgb,
  runOlmoEarthEmbeddingSimilarity,
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

/**
 * Pull the human-readable detail out of an API error.
 *
 * The api/client wrappers throw ``new Error(`API ${status}: ${responseText}`)``
 * and the FastAPI handlers return JSON like ``{"detail": "Circuit breaker
 * tripped: …"}``. So a raw error message looks like:
 *
 *   API 503: {"detail":"Circuit breaker tripped: 3 chunks failed in a row…"}
 *
 * That's hostile to read. The detail string itself is already crafted to be
 * user-actionable (mentions OE_MAX_CHUNKS, suggests retry, etc.). We just
 * need to unwrap it. Returns the original string if parsing fails so we
 * never swallow useful info.
 */
function formatApiError(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  const m = raw.match(/^API (\d+): ([\s\S]+)$/);
  if (!m) return raw;
  const [, status, body] = m;
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === "string") return `${status} — ${parsed.detail}`;
  } catch {
    // Body wasn't JSON; fall through.
  }
  return raw;
}

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
  const [busy, setBusy] = useState<null | "infer" | "cache" | "embed" | "pca" | "sim" | "geojson">(null);
  // Subtab split: Run inference vs. Embedding tools. The panel had grown to
  // 1 inference action + 3 embedding actions + 1 advanced action in a flat
  // list, which pushed the primary Run button below the fold on short
  // popovers. Two tabs keep each workflow focused.
  const [activeTab, setActiveTab] = useState<"inference" | "embedding">("inference");

  const selected = allOptions.find((m) => m.repoId === repoId) ?? FT_HEADS[0];
  const live = olmoCache?.[repoId];
  const cached = live?.status === "cached";

  // Embedding tools require raw per-patch vectors, which only base encoders
  // expose. When the user switches to the Embedding tab while an FT head is
  // selected, auto-swap to Tiny so the tools are usable without an extra
  // click. Switching back to Inference keeps the current base encoder —
  // inference works on both kinds (base encoders render via PCA-on-first-
  // component).
  const switchTab = (next: "inference" | "embedding") => {
    setActiveTab(next);
    if (next === "embedding" && selected.kind === "ft") {
      setRepoId("allenai/OlmoEarth-v1-Tiny");
    }
  };

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
      setStatus(`failed: ${formatApiError(e)}`);
    } finally {
      setBusy(null);
    }
  };

  // FT classification → GeoJSON polygon download. Only meaningful for
  // classification heads (Mangrove, AWF, Ecosystem) — regression (LFMC)
  // doesn't have discrete regions to vectorise. Reuses the same job_id
  // as the map-tile run, so if the user already clicked "Run + add to map"
  // the GeoJSON download is near-instant (no second forward pass).
  // Output is a standard application/geo+json file ready to drop into
  // Google Earth Pro, QGIS, ArcGIS, leaflet, etc.
  const handleDownloadGeoJson = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "ft") return;
    if (!selected.supported) return;
    setBusy("geojson");
    setStatus("Downloading classification as GeoJSON — reuses cached inference if available, else runs a fresh job (may take a few minutes)…");
    try {
      const res = await downloadFtClassificationGeoJson({
        bbox: selectedArea,
        modelRepoId: repoId,
      });
      const count = res.featureCount;
      const countStr = count != null ? `${count} polygon${count === 1 ? "" : "s"}` : "polygons";
      setStatus(`downloaded ${res.filename} · ${countStr}`);
    } catch (e) {
      setStatus(`GeoJSON download failed: ${formatApiError(e)}`);
    } finally {
      setBusy(null);
    }
  };

  // Embedding export — available only for base encoders. Streams the int8
  // COG as a browser download; no map layer is added because embeddings
  // are per-dimension bands, not a single thematic raster. Downstream
  // analysis (similarity search, few-shot segmentation, change detection,
  // PCA) runs off the downloaded file.
  const handleExportEmbedding = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "base") return;
    setBusy("embed");
    setStatus("Exporting embeddings — chunked fetch + encoder forward, may take several minutes…");
    try {
      const result = await exportOlmoEarthEmbedding({
        bbox: selectedArea,
        modelRepoId: repoId,
      });
      downloadEmbeddingExport(result);
      const parts: string[] = [
        `downloaded ${result.filename}`,
        result.embeddingDim != null ? `${result.embeddingDim} dims` : null,
        result.patchGsdM != null ? `${result.patchGsdM} m/pixel` : null,
        result.chunksProcessed != null && result.chunksFailed != null
          ? `${result.chunksProcessed} chunks ok, ${result.chunksFailed} failed`
          : null,
      ].filter((s): s is string => s !== null);
      setStatus(parts.join(" · "));
    } catch (e) {
      setStatus(`export failed: ${formatApiError(e)}`);
    } finally {
      setBusy(null);
    }
  };

  // Embedding tool: PCA false-color visualization. Reuses the same
  // chunked fetch + base encoder forward as the export, but instead of
  // returning a COG file it projects the embedding to top-3 PCs and
  // registers a tile job — drops onto the map like any other inference
  // result. Works globally (no FT-head region lock).
  const handlePCARgb = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "base") return;
    setBusy("pca");
    setStatus("Computing embedding + PCA false-color — this is the same chunked fetch as Export, just with PCA on top.");
    try {
      const res = await runOlmoEarthEmbeddingPCARgb({
        bbox: selectedArea,
        modelRepoId: repoId,
      });
      if (onAddImageryLayer) {
        onAddImageryLayer({
          id: `olmoearth-pca-${res.job_id}`,
          tileUrl: res.tile_url,
          label: `${selected.label} · PCA · ${res.job_id.slice(0, 6)}`,
          inferenceMetadata: res,
        });
        setStatus(
          res.kind === "stub"
            ? `preview stub — PCA failed (${res.stub_reason ?? "unknown"})`
            : "added to map — top-3 PCs as RGB. Similar embeddings = similar colors.",
        );
      } else {
        setStatus(`done — tile URL ready (job ${res.job_id})`);
      }
    } catch (e) {
      setStatus(`PCA failed: ${formatApiError(e)}`);
    } finally {
      setBusy(null);
    }
  };

  // Embedding tool: cosine similarity heatmap. v1 uses the AOI center
  // as the query (zero clicks, instant demo). A future iteration will
  // expose a click-on-map pixel-pick UI for arbitrary query points.
  const handleSimilarity = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "base") return;
    setBusy("sim");
    setStatus("Computing embedding + cosine similarity vs AOI center — bright pixels = looks like the middle of your area.");
    try {
      const res = await runOlmoEarthEmbeddingSimilarity({
        bbox: selectedArea,
        modelRepoId: repoId,
      });
      if (onAddImageryLayer) {
        onAddImageryLayer({
          id: `olmoearth-sim-${res.job_id}`,
          tileUrl: res.tile_url,
          label: `${selected.label} · similarity · ${res.job_id.slice(0, 6)}`,
          inferenceMetadata: res,
        });
        setStatus(
          res.kind === "stub"
            ? `preview stub — similarity failed (${res.stub_reason ?? "unknown"})`
            : "added to map — bright = similar to AOI center, dark = unrelated.",
        );
      } else {
        setStatus(`done — tile URL ready (job ${res.job_id})`);
      }
    } catch (e) {
      setStatus(`Similarity failed: ${formatApiError(e)}`);
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
      setStatus(`failed: ${formatApiError(e)}`);
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
          {activeTab === "inference" ? (
            <>
              Pick an FT head (task-specific) or base encoder (PCA-rendered
              embeddings). <b>Run + add to map</b> executes inference over
              your AOI and drops the resulting raster as a map layer.
            </>
          ) : (
            <>
              Pick a base encoder to produce raw per-patch embeddings over
              your AOI, then visualize them as a false-color layer, find
              similar patches, or export as a COG for downstream analysis.
            </>
          )}
        </div>
      </div>

      {/* Subtab pills. Styled to match the main Map/Analysis/3D Globe/
          OlmoEarth/LLM tab bar so the visual language is consistent. */}
      <div className="flex gap-1 p-0.5 bg-geo-bg border border-geo-border rounded">
        <button
          type="button"
          onClick={() => switchTab("inference")}
          disabled={busy !== null}
          className={`flex-1 px-2 py-1 text-[11px] font-semibold rounded transition-colors ${
            activeTab === "inference"
              ? "bg-geo-accent text-white cursor-default"
              : busy !== null
                ? "text-geo-muted cursor-not-allowed"
                : "text-geo-muted hover:text-geo-text cursor-pointer"
          }`}
        >
          Run inference
        </button>
        <button
          type="button"
          onClick={() => switchTab("embedding")}
          disabled={busy !== null}
          className={`flex-1 px-2 py-1 text-[11px] font-semibold rounded transition-colors ${
            activeTab === "embedding"
              ? "bg-geo-accent text-white cursor-default"
              : busy !== null
                ? "text-geo-muted cursor-not-allowed"
                : "text-geo-muted hover:text-geo-text cursor-pointer"
          }`}
        >
          Embedding tools
        </button>
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
          {/* FT heads are hidden on the Embedding tab — they project
              embeddings to task outputs and don't expose raw vectors, so
              none of the embedding tools could run against them. */}
          {activeTab === "inference" && (
            <optgroup label="Fine-tuned heads (task-specific)">
              {FT_HEADS.map((m) => (
                <option key={m.repoId} value={m.repoId}>
                  {m.label} — {m.task}
                  {!m.supported ? " (loader not supported yet)" : ""}
                </option>
              ))}
            </optgroup>
          )}
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

      {activeTab === "inference" && (
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
      )}

      {/* GeoJSON polygon download — only rendered for FT classification
          heads (Mangrove, AWF, Ecosystem). Reuses the same job_id as the
          map-tile run, so if "Run + add to map" already fired this is
          near-instant. Output drops straight into Google Earth Pro,
          QGIS, ArcGIS, leaflet, etc. — no in-app rendering required. */}
      {activeTab === "inference"
        && selected.kind === "ft"
        && selected.supported
        && selected.task.toLowerCase().includes("classification") && (
        <>
          <button
            type="button"
            onClick={handleDownloadGeoJson}
            disabled={busy !== null || !selectedArea}
            className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
              busy !== null || !selectedArea
                ? "border-geo-border text-geo-muted cursor-not-allowed"
                : "border-geo-accent text-geo-accent hover:bg-geo-accent hover:text-white cursor-pointer"
            }`}
            title={
              selectedArea
                ? "Vectorise the FT classification raster into polygons and download as a GeoJSON file. Reuses cached inference if you already ran this AOI."
                : "Draw an area on the map first"
            }
          >
            {busy === "geojson" ? "Downloading GeoJSON…" : "Download as GeoJSON (for Google Earth / QGIS)"}
          </button>
          <div className="text-[10px] text-geo-muted leading-snug -mt-1">
            One polygon per contiguous class region. Properties carry
            class id, name, color, and area (m²). Loads natively in
            Google Earth Pro, QGIS, ArcGIS, My Maps. Editable as
            vector — refine boundaries or delete misclassifications
            before downstream use.
          </div>
        </>
      )}

      {/* Embedding tools — only rendered on the Embedding subtab. Each
          tool shares the same chunked fetch + base encoder forward; the
          difference is what we do with the resulting embedding tensor:
            * PCA false-color: top-3 PCs → RGB → map layer (in-UI)
            * Similarity to AOI center: cosine heatmap → map layer
            * Export as COG: int8 quantize → download (offline analysis)
          The `selected.kind === "base"` guard is now belt-and-suspenders:
          the model selector on this tab already hides FT heads. */}
      {activeTab === "embedding" && selected.kind === "base" && (
        <>
          {/* Order matters here. We deliberately put Similarity first and
              Export second — those are the workflows scientists actually
              use to produce results. PCA false-color is demoted to an
              "advanced" collapsed section because, while it makes a
              colorful map, the colors aren't comparable across runs (each
              call computes a fresh per-chunk PCA basis) and don't map to
              named classes. PCA's job is "encoder sanity check" + "where
              should I draw labels", not "the answer". */}

          {/* PRIMARY: Similarity search — interpretable, query-driven, the
              most direct path from "I drew an AOI" to "show me more like
              the centre". v1 uses AOI center as the query; click-on-map
              pixel-pick ships in the next pass. */}
          <button
            type="button"
            onClick={handleSimilarity}
            disabled={busy !== null || !selectedArea}
            className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
              busy !== null || !selectedArea
                ? "border-geo-border text-geo-muted cursor-not-allowed"
                : "border-geo-accent bg-geo-accent text-white hover:bg-geo-accent-hover cursor-pointer"
            }`}
            title={
              selectedArea
                ? "Compute embeddings + cosine-similarity heatmap vs the AOI center. Bright = looks like the middle of your area."
                : "Draw an area on the map first"
            }
          >
            {busy === "sim" ? "Computing similarity…" : "Similarity to AOI center"}
          </button>
          <div className="text-[10px] text-geo-muted leading-snug -mt-1">
            Cosine similarity heatmap to the embedding at your AOI center.
            Bright pixels look like the middle of your area — directly
            interpretable. Pixel-pick UI ships next.
          </div>

          {/* SECONDARY: Export as COG — the science-grade exit. Once the
              user has labels, they want this output to feed sklearn /
              QGIS / a notebook for proper supervised analysis. */}
          <button
            type="button"
            onClick={handleExportEmbedding}
            disabled={busy !== null || !selectedArea}
            className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
              busy !== null || !selectedArea
                ? "border-geo-border text-geo-muted cursor-not-allowed"
                : "border-geo-accent text-geo-accent hover:bg-geo-accent hover:text-white cursor-pointer"
            }`}
            title={
              selectedArea
                ? "Compute per-patch embeddings over the selected area and download as an int8 COG"
                : "Draw an area on the map first"
            }
          >
            {busy === "embed" ? "Exporting embeddings…" : "Export embeddings as COG"}
          </button>
          <div className="text-[10px] text-geo-muted leading-snug -mt-1">
            Multi-band int8 GeoTIFF (one band per dim, nodata = -128). Use
            with <code className="font-mono">dequantize_embeddings</code>{" "}
            from <code className="font-mono">olmoearth_pretrain</code> for
            similarity search, few-shot segmentation, or change detection.
          </div>

          {/* TERTIARY: PCA false-color — kept for encoder QA + label-
              picking, but collapsed by default so it doesn't crowd the
              real workflows. Caveat in the body makes its limits clear. */}
          <details className="border border-geo-border rounded">
            <summary className="px-2 py-1.5 text-[11px] cursor-pointer hover:bg-geo-bg/60">
              Advanced: encoder sanity check (PCA false-color)
            </summary>
            <div className="p-2 space-y-2 border-t border-geo-border">
              <button
                type="button"
                onClick={handlePCARgb}
                disabled={busy !== null || !selectedArea}
                className={`w-full px-3 py-1.5 text-[11px] font-semibold rounded border transition-colors ${
                  busy !== null || !selectedArea
                    ? "border-geo-border text-geo-muted cursor-not-allowed"
                    : "border-geo-border text-geo-text hover:border-geo-accent hover:text-geo-accent cursor-pointer"
                }`}
                title={
                  selectedArea
                    ? "Sanity-check the encoder by mapping top-3 embedding PCs to RGB. Useful for spotting where to draw labels; NOT a classification."
                    : "Draw an area on the map first"
                }
              >
                {busy === "pca" ? "Computing PCA false-color…" : "Run PCA false-color"}
              </button>
              <div className="text-[10px] text-geo-muted leading-snug">
                Top-3 PCs of the per-patch embedding → R/G/B. Useful for
                eyeballing landscape diversity and deciding where to draw
                labels.{" "}
                <b>Not a classification</b> — colors aren't stable across
                runs (per-chunk PCA basis), and visible chunk seams are
                expected. For science output use Similarity (above) or
                Export-as-COG + sklearn.
              </div>
            </div>
          </details>
        </>
      )}

      {/* Secondary: load-weights-only. Inference-tab only — pre-warming
          the disk cache is a knob you pull before running inference.
          Smaller / less prominent styling so it's clearly the advanced-
          user path, not the default. */}
      {activeTab === "inference" && (
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
      )}

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
            // Catch "failed", "error", or "preview stub" anywhere — not just
            // as a prefix. Status strings now look like "PCA failed: …",
            // "Similarity failed: …", "export failed: …" so a startsWith
            // check missed them and the actionable error rendered in grey.
            /\b(failed|error|preview stub)\b/i.test(status)
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
