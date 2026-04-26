import { useEffect, useRef, useState } from "react";
import {
  downloadEmbeddingExport,
  downloadFtClassificationGeoJson,
  exportOlmoEarthEmbedding,
  loadOlmoEarthRepo,
  runOlmoEarthEmbeddingFewShot,
  runOlmoEarthEmbeddingPCARgb,
  runOlmoEarthEmbeddingSimilarity,
  startOlmoEarthInference,
  type OlmoEarthRepoStatus,
  type OlmoEarthInferenceResult,
} from "../api/client";
import type { BBox } from "../types";
import type { ImageryLayer } from "./MapView";
import { OffDistributionBanner } from "./OffDistributionBanner";

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
    // Pre/post pipeline wired 2026-04-24. Backend fetches a pre group
    // (~event - 300d, 4 scenes) and a post group (~event + 7d, 4
    // scenes), concatenates encoder outputs along the feature dim
    // (768 + 768 → 1536), and feeds the conv_pool_fc head. Requires the
    // user to supply an ``event_date`` (the post-event date is the
    // natural anchor) — surfaced as a date picker below the model
    // selector when this head is active.
    supported: true,
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

/**
 * Per-model demo AOI registry. Each entry picks a small (~3 km × 3 km)
 * bbox over a region where the model's training distribution actually
 * applies, so a one-click run produces a meaningful output. The button
 * also pans the map there, so the user immediately sees what the model
 * was looking at.
 *
 * Coverage rationale:
 *   * LFMC — Riverside County chaparral (fire-prone, dry fuel)
 *   * Mangrove — Florida Keys (well-mapped tropical mangrove)
 *   * AWF — Tsavo East, Kenya (head's training-distribution region)
 *   * ForestLossDriver — Pará, Brazilian Amazon (verified end-to-end with
 *     event_date=2022-08-15 → "agriculture" 32% / "burned" 17% softmax)
 *   * EcosystemTypeMapping — Tunisia (head trained on north Africa group)
 *   * Base encoders — Monterey Bay (default, works anywhere; chosen for
 *     visual diversity in PCA false-color + similarity)
 */
const DEMO_AOIS: Record<string, { bbox: BBox; eventDate?: string; label: string }> = {
  "allenai/OlmoEarth-v1-FT-LFMC-Base": {
    bbox: { west: -117.30, south: 33.75, east: -117.27, north: 33.78 },
    label: "Riverside chaparral",
  },
  "allenai/OlmoEarth-v1-FT-Mangrove-Base": {
    bbox: { west: -81.45, south: 24.62, east: -81.42, north: 24.65 },
    label: "Florida Keys",
  },
  "allenai/OlmoEarth-v1-FT-AWF-Base": {
    bbox: { west: 38.50, south: -3.10, east: 38.53, north: -3.07 },
    label: "Tsavo East, Kenya",
  },
  "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base": {
    bbox: { west: -55.05, south: -9.05, east: -55.02, north: -9.02 },
    eventDate: "2022-08-15",
    label: "Pará, Brazilian Amazon",
  },
  "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": {
    bbox: { west: 10.10, south: 35.50, east: 10.13, north: 35.53 },
    label: "Tunisia (training region)",
  },
};
const DEMO_AOI_FALLBACK: { bbox: BBox; label: string } = {
  bbox: { west: -121.92, south: 36.60, east: -121.89, north: 36.63 },
  label: "Monterey Bay",
};

export function OlmoEarthImport({
  olmoCache,
  compact,
  initialRepoId,
  selectedArea,
  onAddImageryLayer,
  onSelectArea,
  queryPixel,
  pickQueryActive,
  onStartPickQuery,
  onClearQueryPixel,
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
  /** Set the AOI from inside this component (used by the per-model demo
   * AOI button). Wired to App.handleDemoAreaSelect which both updates
   * ``selectedArea`` and pans the map to the demo location. Optional —
   * the button hides if not provided (e.g. read-only contexts). */
  onSelectArea?: (bbox: BBox) => void;
  /** Embedding similarity query pixel state — picked location, whether
   * pixel-pick mode is currently armed, and handlers to start / clear.
   * App-lifted so the same picked pixel renders on the map AND the
   * Similarity tool sees it on the next click. Optional — pixel-pick UI
   * just hides when handlers aren't provided. */
  queryPixel?: { lon: number; lat: number } | null;
  pickQueryActive?: boolean;
  onStartPickQuery?: () => void;
  onClearQueryPixel?: () => void;
}) {
  // Default selection: first FT head unless initialRepoId matched something.
  const allOptions = [...FT_HEADS, ...BASE_ENCODERS];
  const initialOption =
    allOptions.find((m) => m.repoId === initialRepoId) ?? FT_HEADS[0];
  const [repoId, setRepoId] = useState(initialOption.repoId);
  const [hfToken, setHfToken] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState<null | "infer" | "cache" | "embed" | "pca" | "sim" | "geojson" | "fewshot">(null);
  // Ref to the model <select> so the off-distribution banner's
  // "Pick a different head" button can scroll/focus it.
  const modelSelectRef = useRef<HTMLSelectElement | null>(null);

  // Few-shot semantic segmentation state. Three classes pre-seeded with
  // distinct accent colours so the user can start clicking immediately
  // without first picking a palette. ``pickingForClass`` tracks which
  // class's "+ point" button is currently armed; null means no
  // few-shot pick is in progress (so we don't accidentally consume a
  // similarity-pick committed pixel).
  const [fewShotClasses, setFewShotClasses] = useState<{
    name: string;
    color: string;
    points: { lon: number; lat: number }[];
  }[]>([
    { name: "Class 1", color: "#ef4444", points: [] },
    { name: "Class 2", color: "#3b82f6", points: [] },
    { name: "Class 3", color: "#22c55e", points: [] },
  ]);
  const [pickingForClass, setPickingForClass] = useState<number | null>(null);
  // Identity-track which queryPixel objects we've already consumed so
  // the effect below never double-appends or fires on stale state. Each
  // fresh pick produces a new object reference from App, so a strict
  // identity check distinguishes "new pick" from "same pick re-rendered".
  const consumedQueryPixelRef = useRef<typeof queryPixel>(null);

  useEffect(() => {
    if (!queryPixel || pickingForClass === null) return;
    if (consumedQueryPixelRef.current === queryPixel) return;
    consumedQueryPixelRef.current = queryPixel;
    setFewShotClasses((prev) => prev.map((c, i) =>
      i === pickingForClass
        ? { ...c, points: [...c.points, queryPixel] }
        : c,
    ));
    setPickingForClass(null);
    onClearQueryPixel?.();
  }, [queryPixel, pickingForClass, onClearQueryPixel]);

  // Reset few-shot picks when the AOI changes — labels outside the
  // new AOI are nodata-bound and would silently no-op the prototype
  // computation, so it's clearer to wipe the slate than carry stale
  // points forward.
  useEffect(() => {
    setFewShotClasses((prev) => prev.map((c) => ({ ...c, points: [] })));
    setPickingForClass(null);
  }, [selectedArea]);
  // Event date for pre/post change-detection FT heads (ForestLossDriver).
  // Only surfaced when such a head is selected. Default is one year ago —
  // recent enough to have S2 coverage on both pre and post windows, old
  // enough that a 300-day pre offset doesn't fall outside the S2 archive.
  const [eventDate, setEventDate] = useState<string>(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().slice(0, 10);
  });
  // Sliding-window inference toggle. When ON, the chunked orchestrator
  // runs the FT head over a grid of ``window_size``-pixel windows per
  // chunk instead of one scene-level forward — turns ForestLossDriver's
  // "one driver class per 5 km chunk" into "one class per ~64 px
  // window" (16× finer for FLD's predict_window_px=64). Default ON for
  // classification heads where it matters; off for segmentation /
  // regression where the head already produces per-pixel output. Window
  // size is chosen by the backend from the head's metadata
  // (``predict_window_px``) when this is on.
  const [slidingWindow, setSlidingWindow] = useState<boolean>(true);
  // Subtab split: Run inference vs. Embedding tools. The panel had grown to
  // 1 inference action + 3 embedding actions + 1 advanced action in a flat
  // list, which pushed the primary Run button below the fold on short
  // popovers. Two tabs keep each workflow focused.
  const [activeTab, setActiveTab] = useState<"inference" | "embedding">("inference");

  const selected = allOptions.find((m) => m.repoId === repoId) ?? FT_HEADS[0];
  const live = olmoCache?.[repoId];
  const cached = live?.status === "cached";
  // Pre/post change-detection heads need an event_date. Today only
  // ForestLossDriver fits; check by repo id rather than threading a flag
  // through ModelOption since the list is short.
  const isPrePostHead = selected.repoId === "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base";
  // Pick the demo AOI keyed off the currently selected model. Falls back
  // to Monterey Bay for any model not in the registry (mostly base
  // encoders, which work anywhere).
  const demoAoi = DEMO_AOIS[selected.repoId] ?? DEMO_AOI_FALLBACK;

  // One-click "set AOI to a region this model can actually predict on +
  // pan map there + auto-set event_date for ForestLossDriver". This
  // closes the gap where Try Demo (Kenyan Coast) gave a 44 km AOI —
  // useful for AWF, useless for everything else and would take 10+ min
  // to run.
  const handlePickDemoAoi = () => {
    if (!onSelectArea) return;
    onSelectArea(demoAoi.bbox);
    if (isPrePostHead && demoAoi.eventDate) {
      setEventDate(demoAoi.eventDate);
    }
    setStatus(`Demo AOI set to ${demoAoi.label} — click Run to execute inference`);
  };

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
    if (isPrePostHead && !eventDate) return;
    setBusy("infer");
    setStatus(null);
    try {
      const res: OlmoEarthInferenceResult = await startOlmoEarthInference({
        bbox: selectedArea,
        modelRepoId: repoId,
        eventDate: isPrePostHead ? eventDate : undefined,
        // Only request sliding-window for FT heads — base encoders
        // run via the embedding-tools endpoints, not /infer, so
        // sliding_window has no effect there. Sending it on a base
        // encoder's /infer call is silently ignored by the dispatcher.
        slidingWindow: selected.kind === "ft" ? slidingWindow : undefined,
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
    if (isPrePostHead && !eventDate) return;
    setBusy("geojson");
    setStatus("Downloading classification as GeoJSON — reuses cached inference if available, else runs a fresh job (may take a few minutes)…");
    try {
      const res = await downloadFtClassificationGeoJson({
        bbox: selectedArea,
        modelRepoId: repoId,
        eventDate: isPrePostHead ? eventDate : undefined,
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

  // Embedding tool: cosine similarity heatmap. The query pixel is
  // either user-picked (via the "Pick query pixel" button → map click)
  // or defaults to the AOI center when none is picked. The picked
  // pixel persists across runs so users can iterate on the same query
  // location while toggling models / dates.
  const handleSimilarity = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "base") return;
    setBusy("sim");
    const queryLabel = queryPixel
      ? `picked pixel (${queryPixel.lat.toFixed(4)}, ${queryPixel.lon.toFixed(4)})`
      : "AOI center";
    setStatus(
      `Computing embedding + cosine similarity vs ${queryLabel} — bright pixels = looks like the query.`,
    );
    try {
      const res = await runOlmoEarthEmbeddingSimilarity({
        bbox: selectedArea,
        modelRepoId: repoId,
        queryLon: queryPixel?.lon,
        queryLat: queryPixel?.lat,
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

  // Few-shot: arm pick for a specific class. Clearing any in-flight
  // similarity pick first prevents the useEffect above from picking up
  // a stale queryPixel as the first labelled point.
  const handlePickPointForClass = (idx: number) => {
    if (!selectedArea) return;
    onClearQueryPixel?.();
    setPickingForClass(idx);
    onStartPickQuery?.();
  };

  // Few-shot: clear all labelled points for a class. The user
  // typically does this after seeing a noisy result and wanting to
  // re-label more carefully.
  const handleClearClassPoints = (idx: number) => {
    setFewShotClasses((prev) => prev.map((c, i) =>
      i === idx ? { ...c, points: [] } : c,
    ));
  };

  // Few-shot: rename a class. Inline rename via prompt() — this is a
  // dev-grade UX, not the polished version, but it lets the user move
  // beyond "Class 1 / Class 2" without a full settings panel.
  const handleRenameClass = (idx: number) => {
    const current = fewShotClasses[idx]?.name ?? "";
    const next = window.prompt("Class name:", current);
    if (next != null && next.trim()) {
      setFewShotClasses((prev) => prev.map((c, i) =>
        i === idx ? { ...c, name: next.trim().slice(0, 64) } : c,
      ));
    }
  };

  // Few-shot: run the classification pipeline. Requires at least 2
  // classes with at least 1 point each (matches backend validation).
  const handleRunFewShot = async () => {
    if (!selectedArea) return;
    if (selected.kind !== "base") return;
    const usable = fewShotClasses.filter((c) => c.points.length > 0);
    if (usable.length < 2) return;
    setBusy("fewshot");
    setStatus(
      `Running few-shot — ${usable.length} classes, ` +
      `${usable.reduce((n, c) => n + c.points.length, 0)} labelled points total. ` +
      "Same chunked encoder pass as PCA / Similarity, plus a per-class prototype + nearest-class assignment.",
    );
    try {
      const res = await runOlmoEarthEmbeddingFewShot({
        bbox: selectedArea,
        modelRepoId: repoId,
        classes: usable,
      });
      if (onAddImageryLayer) {
        onAddImageryLayer({
          id: `olmoearth-fewshot-${res.job_id}`,
          tileUrl: res.tile_url,
          label: `${selected.label} · few-shot · ${res.job_id.slice(0, 6)}`,
          inferenceMetadata: res,
        });
        setStatus(
          res.kind === "stub"
            ? `preview stub — few-shot failed (${res.stub_reason ?? "unknown"})`
            : "added to map — argmax cosine similarity to your labelled prototypes.",
        );
      } else {
        setStatus(`done — tile URL ready (job ${res.job_id})`);
      }
    } catch (e) {
      setStatus(`Few-shot failed: ${formatApiError(e)}`);
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
          ref={modelSelectRef}
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
      </div>

      {/* Off-distribution banner — replaces the prior italic gray
          coverage hint. Renders only when:
            (a) we're on the Inference tab (not Embedding tools),
            (b) selected head is an FT head,
            (c) AOI is set AND falls outside the head's training region.
          When all three hit, the banner shows the headline, body copy
          tailored per head, an inline world map, footer actions, and an
          advisory that the user can still run the model. The banner is
          internally responsible for its own conditional rendering — it
          renders nothing for in-distribution AOIs. */}
      {activeTab === "inference" && selected.kind === "ft" && (
        <OffDistributionBanner
          aoi={selectedArea}
          modelRepoId={selected.repoId}
          modelLabel={selected.label}
          onPickAnotherHead={() => {
            const el = modelSelectRef.current;
            if (!el) return;
            el.scrollIntoView({ behavior: "smooth", block: "center" });
            el.focus();
          }}
        />
      )}

      {/* Event date picker — only for pre/post change-detection heads
          (ForestLossDriver today). The backend treats this as the
          post-event anchor: pre window ~event - 300d, post window
          ~event + 7d. Without a date the request falls back to the
          legacy single-scene path which produces off-distribution
          output, so we make the field required for pre/post heads. */}
      {activeTab === "inference" && isPrePostHead && (
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-geo-muted">
            Event date <span className="text-geo-danger">*</span>
          </span>
          <input
            type="date"
            value={eventDate}
            onChange={(e) => setEventDate(e.target.value)}
            disabled={busy !== null}
            className="w-full mt-0.5 px-2 py-1.5 text-[12px] bg-geo-surface border border-geo-border rounded focus:border-geo-accent focus:outline-none cursor-pointer"
          />
          <div className="mt-1 text-[10px] text-geo-muted leading-snug">
            Approximate date of the forest-loss event. The model fetches
            ~4 Sentinel-2 scenes from the year before (pre window,
            ~event − 300 d) and ~4 scenes immediately after (post
            window, ~event + 7 d). Pick a recent enough date that
            S2 has coverage on both sides — late 2017 onward is safe.
          </div>
        </div>
      )}

      {/* Sliding-window toggle — only for FT heads. When ON, the
          backend runs the head over a grid of ~64 px windows per chunk
          instead of one scene-level forward. Turns ForestLossDriver's
          "one class per 5 km chunk" into "one class per ~64 px window"
          (16× finer); for already-segmentation heads (Mangrove, AWF,
          Ecosystem) it gives finer per-window confidence variation.
          Adds wall time roughly proportional to (chunk_size_px /
          window_size)² extra forward passes per chunk; for the default
          5 km / 64 px ratio that's ~49 forwards per chunk vs 1. */}
      {activeTab === "inference" && selected.kind === "ft" && (
        <label className="flex items-start gap-2 px-2 py-1.5 rounded bg-geo-bg border border-geo-border cursor-pointer">
          <input
            type="checkbox"
            checked={slidingWindow}
            onChange={(e) => setSlidingWindow(e.target.checked)}
            disabled={busy !== null}
            className="mt-0.5 cursor-pointer"
          />
          <div className="flex-1 text-[10px] leading-snug">
            <div className="font-semibold text-geo-text">
              Sliding-window inference
            </div>
            <div className="text-geo-muted">
              Run the head per ~64 px window instead of once per chunk.
              Turns scene-level outputs (ForestLossDriver, classification
              FTs) into a real per-window class map; ~50× more forward
              passes per chunk so wall time roughly doubles.
            </div>
          </div>
        </label>
      )}

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
        {/* Per-model "demo AOI" shortcut. Sets a small (~3 km) bbox over
            a region in the model's training distribution and pans the
            map there. For ForestLossDriver, also auto-fills event_date.
            Hidden when the host didn't wire the callback (e.g. read-only
            preview). Subtle styling so users know it's a shortcut, not
            the primary path. */}
        {onSelectArea && activeTab === "inference" && (
          <button
            type="button"
            onClick={handlePickDemoAoi}
            disabled={busy !== null}
            className={`mt-1 text-[10px] underline-offset-2 hover:underline transition-colors ${
              busy !== null
                ? "text-geo-muted cursor-not-allowed"
                : "text-geo-accent cursor-pointer"
            }`}
            title={`Pick a small demo AOI (~3 km × 3 km) over ${demoAoi.label} — runs in ~1-3 min instead of 10+`}
          >
            ↳ Use demo AOI ({demoAoi.label})
          </button>
        )}
      </div>

      {activeTab === "inference" && (
        <button
          type="button"
          onClick={handleRun}
          disabled={busy !== null || !selectedArea || !selected.supported || (isPrePostHead && !eventDate)}
          className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
            busy !== null || !selectedArea || !selected.supported || (isPrePostHead && !eventDate)
              ? "border-geo-border text-geo-muted cursor-not-allowed"
              : "border-geo-accent bg-geo-accent text-white hover:bg-geo-accent-hover cursor-pointer"
          }`}
          title={
            !selected.supported
              ? "This FT head's decoder shape isn't supported by the loader yet. Pick a supported head (Mangrove, AWF, Ecosystem, or any base encoder)."
              : isPrePostHead && !eventDate
                ? "Pick an event date first — pre/post heads need it to fetch the before/after S2 pair"
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

          {/* Pixel-pick UI for the Similarity query. Three states:
                a) idle, no pick yet — show "Pick query pixel" button
                   (disabled until AOI is set)
                b) picking armed — show status + Cancel; map cursor is
                   crosshair, next click writes the pixel
                c) pick committed — show coords + × clear
              Hidden entirely when the App-level handlers aren't wired
              (e.g. read-only preview contexts). */}
          {onStartPickQuery && onClearQueryPixel && (
            <div className="px-2 py-1.5 rounded bg-geo-bg border border-geo-border text-[10px] leading-snug">
              <div className="text-geo-muted font-semibold uppercase tracking-wider">
                Query pixel
              </div>
              {pickQueryActive ? (
                <div className="flex items-center justify-between gap-2 mt-0.5">
                  <span className="text-geo-accent">
                    Click anywhere on the map to set the query pixel…
                  </span>
                  <button
                    type="button"
                    onClick={onClearQueryPixel}
                    className="text-geo-muted hover:text-geo-text underline-offset-2 hover:underline cursor-pointer"
                  >
                    Cancel
                  </button>
                </div>
              ) : queryPixel ? (
                <div className="flex items-center justify-between gap-2 mt-0.5">
                  <span className="text-geo-text font-mono">
                    {queryPixel.lat.toFixed(4)}, {queryPixel.lon.toFixed(4)}
                  </span>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={onStartPickQuery}
                      disabled={busy !== null || !selectedArea}
                      className={`underline-offset-2 hover:underline ${
                        busy !== null || !selectedArea
                          ? "text-geo-muted cursor-not-allowed"
                          : "text-geo-accent cursor-pointer"
                      }`}
                    >
                      Re-pick
                    </button>
                    <button
                      type="button"
                      onClick={onClearQueryPixel}
                      className="text-geo-muted hover:text-geo-danger cursor-pointer"
                      title="Clear picked pixel — Similarity will fall back to AOI center"
                    >
                      × clear
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-between gap-2 mt-0.5">
                  <span className="text-geo-muted">
                    Default: AOI center
                  </span>
                  <button
                    type="button"
                    onClick={onStartPickQuery}
                    disabled={busy !== null || !selectedArea}
                    className={`underline-offset-2 hover:underline ${
                      busy !== null || !selectedArea
                        ? "text-geo-muted cursor-not-allowed"
                        : "text-geo-accent cursor-pointer"
                    }`}
                    title={
                      selectedArea
                        ? "Click then pick a pixel on the map to set the similarity query location"
                        : "Draw an AOI on the map first"
                    }
                  >
                    Pick query pixel
                  </button>
                </div>
              )}
            </div>
          )}

          {/* PRIMARY: Similarity search — interpretable, query-driven, the
              most direct path from "I drew an AOI" to "show me more like
              the query". Query is either user-picked (via the block
              above) or AOI center when none is picked. */}
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
                ? `Compute embeddings + cosine-similarity heatmap vs the ${queryPixel ? "picked query pixel" : "AOI center (default — pick a pixel above for a custom query)"}. Bright = looks like the query.`
                : "Draw an area on the map first"
            }
          >
            {busy === "sim"
              ? "Computing similarity…"
              : queryPixel
                ? "Similarity to picked pixel"
                : "Similarity to AOI center"}
          </button>
          <div className="text-[10px] text-geo-muted leading-snug -mt-1">
            Cosine similarity heatmap to the embedding at the query
            pixel (use the picker above) or AOI center (default).
            Bright pixels look like the query — directly interpretable.
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

          {/* Few-shot semantic segmentation — the science workflow.
              Click N example pixels per class on the map; backend
              computes a prototype embedding per class + argmax cosine
              similarity per pixel = a class raster. Reuses the same
              pixel-pick mechanic as Similarity. Result registers as a
              classification job so the existing GeoJSON download works
              on it.

              Hidden when the App-level pick handlers aren't wired
              (read-only contexts) — the picker is mandatory. */}
          {onStartPickQuery && onClearQueryPixel && (
            <details className="border border-geo-border rounded" open>
              <summary className="px-2 py-1.5 text-[11px] cursor-pointer hover:bg-geo-bg/60">
                Few-shot classify ({fewShotClasses.filter((c) => c.points.length > 0).length} of {fewShotClasses.length} classes labelled)
              </summary>
              <div className="p-2 space-y-2 border-t border-geo-border">
                <div className="text-[10px] text-geo-muted leading-snug">
                  Click <b>+ point</b> for a class, then click anywhere
                  on the map to label that pixel. Repeat to add more
                  examples per class. Run when at least 2 classes have
                  ≥1 point each — backend computes per-class prototype
                  embeddings + assigns each AOI pixel to its nearest
                  prototype by cosine similarity. <b>No fine-tuning.</b>
                </div>
                <div className="space-y-1">
                  {fewShotClasses.map((c, i) => {
                    const isPicking = pickingForClass === i;
                    return (
                      <div
                        key={i}
                        className="flex items-center gap-2 px-2 py-1.5 rounded bg-geo-bg border border-geo-border"
                      >
                        <span
                          className="inline-block w-3 h-3 rounded-full border border-geo-border flex-shrink-0"
                          style={{ backgroundColor: c.color }}
                          title={`Class ${i} colour`}
                        />
                        <button
                          type="button"
                          onClick={() => handleRenameClass(i)}
                          className="text-[11px] font-semibold text-geo-text hover:text-geo-accent cursor-pointer text-left flex-1 truncate"
                          title="Click to rename"
                          disabled={busy !== null}
                        >
                          {c.name}
                        </button>
                        <span className="text-[10px] text-geo-muted font-mono">
                          {c.points.length} pt{c.points.length === 1 ? "" : "s"}
                        </span>
                        <button
                          type="button"
                          onClick={() => handlePickPointForClass(i)}
                          disabled={busy !== null || !selectedArea || isPicking || pickQueryActive}
                          className={`text-[10px] underline-offset-2 hover:underline ${
                            busy !== null || !selectedArea || isPicking || pickQueryActive
                              ? "text-geo-muted cursor-not-allowed"
                              : "text-geo-accent cursor-pointer"
                          }`}
                          title={
                            !selectedArea
                              ? "Draw an AOI first"
                              : pickQueryActive && !isPicking
                                ? "Cancel the active pick or wait for it to commit"
                                : isPicking
                                  ? "Click anywhere on the map to add a point to this class"
                                  : "Arm pick mode + add the next clicked map pixel to this class"
                          }
                        >
                          {isPicking ? "armed — click map" : "+ point"}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleClearClassPoints(i)}
                          disabled={busy !== null || c.points.length === 0}
                          className={`text-[10px] ${
                            busy !== null || c.points.length === 0
                              ? "text-geo-muted cursor-not-allowed"
                              : "text-geo-muted hover:text-geo-danger cursor-pointer"
                          }`}
                          title="Clear all labelled points for this class"
                        >
                          ×
                        </button>
                      </div>
                    );
                  })}
                </div>
                <button
                  type="button"
                  onClick={handleRunFewShot}
                  disabled={
                    busy !== null
                    || !selectedArea
                    || fewShotClasses.filter((c) => c.points.length > 0).length < 2
                  }
                  className={`w-full px-3 py-2 text-[12px] font-semibold rounded border transition-colors ${
                    busy !== null
                      || !selectedArea
                      || fewShotClasses.filter((c) => c.points.length > 0).length < 2
                      ? "border-geo-border text-geo-muted cursor-not-allowed"
                      : "border-geo-accent bg-geo-accent text-white hover:bg-geo-accent-hover cursor-pointer"
                  }`}
                  title={
                    !selectedArea
                      ? "Draw an AOI first"
                      : fewShotClasses.filter((c) => c.points.length > 0).length < 2
                        ? "Label at least 2 classes (≥1 point each) before running"
                        : "Compute prototypes + assign every AOI pixel to its nearest prototype"
                  }
                >
                  {busy === "fewshot" ? "Running few-shot…" : "Run few-shot classification"}
                </button>
                <div className="text-[10px] text-geo-muted leading-snug">
                  Output is a classification raster — same shape as
                  Mangrove / AWF / Ecosystem FT outputs, so the
                  GeoJSON download (Run inference tab) works on this
                  job too. Pixels with no embedding (chunks that
                  failed) render transparent rather than being forced
                  into the nearest class.
                </div>
              </div>
            </details>
          )}

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
                runs (PCA basis depends on the AOI's embedding
                distribution). Chunk-grid seams are smoothed via a small
                Gaussian blur (configurable via{" "}
                <code className="font-mono">OE_PCA_SMOOTH_SIGMA</code>);
                the basis itself is computed once globally on the stitched
                tensor. For science output use Similarity (above) or
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
