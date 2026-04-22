import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import MaplibreCompare from "@maplibre/maplibre-gl-compare";
import "@maplibre/maplibre-gl-compare/style.css";
import type { BBox, BasemapStyle } from "../types";
import type { ImageryLayer } from "./MapView";
import {
  getLoadedOlmoEarthModels,
  getOlmoEarthDemoPairs,
  prebakeOlmoEarthDemo,
  type OlmoEarthDemoPair,
  type OlmoEarthLegendHint,
} from "../api/client";

interface SplitMapProps {
  selectedArea: BBox | null;
  imageryLayers: ImageryLayer[];
  /** Optional seed camera carried from MapView. When present, both
   * split-maps init at this center+zoom instead of the default
   * ``zoom=8`` at the selected-area centroid — so a user who zoomed
   * in on MapView before clicking Compare lands at the same detail
   * level, not back at a country-scale view. Propagated by App via
   * ``MapView.onCameraChange``. */
  initialCamera?: { center: [number, number]; zoom: number } | null;
  /** Lift an example A/B pair into App state so the split's two sides have
   * something to display. Called by the "Load example" presets when the
   * user wants a one-click demo (mirrors OlmoEarth Studio's "Prediction
   * Nigeria Mangrove 2018 vs 2024" compare). */
  onLoadExamplePair?: (a: ImageryLayer, b: ImageryLayer) => void;
  /** Drop a layer id from App's imageryLayers state. Used by the
   * example-pair cards' double-click-to-drop gesture (same pattern as
   * SampleData.tsx — single-click load, double-click drop). */
  onRemoveImageryLayer?: (id: string) => void;
  onExit: () => void;
}

// NOTE: The old static NASA-GIBS EXAMPLE_COMPARE_PAIRS list lived here.
// Deleted — compare demos now stream from the backend's
// ``/api/olmoearth/demo-pairs`` registry, which runs real FT inference
// (Mangrove / AWF / EcosystemTypeMapping) on Sentinel-2 L2A composites.
// That replaces generic satellite imagery with color-coded model outputs
// that actually demonstrate Roger Studio's value prop. See
// ``backend/app/services/olmoearth_demos.py`` for the spec registry.

const BASEMAP_URL: Record<BasemapStyle, string> = {
  osm: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  dark: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
};

function baseStyle(basemap: BasemapStyle): maplibregl.StyleSpecification {
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: [BASEMAP_URL[basemap]],
        tileSize: 256,
        attribution: basemap === "satellite" ? "Esri" : basemap === "dark" ? "CartoDB" : "OSM",
      },
    },
    layers: [{ id: "osm-tiles", type: "raster", source: "osm", minzoom: 0, maxzoom: 19 }],
  };
}

// Fit the initial camera to the selected bbox, or center-of-world if none.
function initialCamera(bbox: BBox | null): { center: [number, number]; zoom: number } {
  if (!bbox) return { center: [0, 0], zoom: 2 };
  return {
    center: [(bbox.west + bbox.east) / 2, (bbox.south + bbox.north) / 2],
    zoom: 8,
  };
}

/** Compact legend card for one side of the split view. Renders a gradient
 * bar assembled from the colormap's ``stops`` plus the axis label. Full
 * honesty-note copy (e.g. "Raw post-softmax score — not a calibrated
 * probability") is surfaced via the ``title`` tooltip so it's discoverable
 * without cluttering the map. ``side`` is the A / B badge — matches the
 * dropdown labels in the top-left controls, so users don't have to
 * mentally map a legend to a raster. */
function MiniLegend({
  side,
  legend,
  stub,
}: {
  side: "A" | "B";
  legend: OlmoEarthLegendHint;
  /** Populated when the tile response carried ``X-Inference-Kind: stub``.
   * The raster is a preview gradient + watermark, not real FT output —
   * the user needs to know before drawing conclusions from the compare. */
  stub?: { reason: string } | null;
}) {
  // Compose a CSS linear-gradient from the stops. Stops are [color, pos]
  // pairs with pos in [0, 1]; CSS wants percentages. Sorted defensively
  // in case the backend ever emits unsorted stops (current _COLORMAP_LEGEND
  // is already ordered ascending, but costs nothing to guarantee).
  const sorted = [...legend.stops].sort((a, b) => a[1] - b[1]);
  const gradient = `linear-gradient(to right, ${sorted
    .map(([color, pos]) => `${color} ${(pos * 100).toFixed(0)}%`)
    .join(", ")})`;
  // Prefer the backend-supplied semantic anchors. Earlier the card showed
  // only "low / high" under the gradient, which tells a user literally
  // nothing for a mangrove score — what's low? Now a mangrove legend
  // reads "non-mangrove → mangrove" under the cyan gradient, so the
  // color→meaning mapping is obvious without reading the tooltip.
  const lowLabel = legend.low_label ?? "low";
  const highLabel = legend.high_label ?? "high";
  // Mid-color swatches help the "what color means what" question when the
  // gradient passes through distinct hues (e.g. landuse: gold→green→cyan).
  // Show a swatch + its stop position for every stop.
  return (
    <div
      className={`pointer-events-auto backdrop-blur text-geo-text px-2.5 py-2 rounded-lg shadow-md max-w-[240px] border ${
        stub
          ? "bg-amber-50/95 border-amber-400"
          : "bg-gradient-panel border-geo-border"
      }`}
      title={legend.note ?? undefined}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-geo-accent text-white text-[9px] font-bold">
          {side}
        </span>
        <span className="text-[10px] font-semibold text-geo-text truncate">
          {legend.label ?? legend.colormap}
        </span>
      </div>
      {stub && (
        // Amber banner signals "this is not a real prediction". Matches
        // the InferenceLegendPanel stub-banner palette from item #8 so
        // the visual language across single-view and compare is consistent.
        <div
          className="mb-1.5 px-1.5 py-1 rounded bg-amber-100 border border-amber-300 text-amber-900 text-[9px] leading-snug"
          data-testid={`split-legend-${side}-stub`}
        >
          <div className="font-bold">PREVIEW STUB — not real inference</div>
          <div className="text-[9px]">
            {stub.reason.slice(0, 120)}
            {stub.reason.length > 120 ? "…" : ""}
          </div>
          <div className="text-[9px] italic mt-0.5">Drop + re-click the pair to retry.</div>
        </div>
      )}
      <div
        className="h-2 rounded-sm border border-geo-border/60"
        style={{ backgroundImage: gradient }}
        aria-hidden="true"
      />
      <div className="flex justify-between text-[9px] text-geo-muted mt-0.5 mb-1">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
      {/* Explicit per-stop swatches — clearer than relying on the
          continuous gradient alone, especially for 3-stop palettes where
          the middle color carries its own meaning (e.g. AWF green =
          rangeland, sitting between gold cropland and cyan water). */}
      <div className="flex gap-2 flex-wrap text-[9px] text-geo-text">
        {sorted.map(([color, pos], i) => {
          const anchor =
            pos <= 0.01 ? lowLabel : pos >= 0.99 ? highLabel : `mid (${(pos * 100).toFixed(0)}%)`;
          return (
            <span key={`${color}-${i}`} className="inline-flex items-center gap-1">
              <span
                className="w-2.5 h-2.5 rounded-sm border border-geo-border/60 flex-shrink-0"
                style={{ backgroundColor: color }}
                aria-hidden="true"
              />
              <span className="truncate">{anchor}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

function applyImageryLayer(
  map: maplibregl.Map,
  layer: ImageryLayer | null,
  opacity: number = 1,
) {
  // Fast path: same layer already attached → just tweak paint opacity so
  // the user's slider moves translate to an instant visual change (no
  // source tear-down / re-fetch). We key on source id (`compare-imagery`)
  // plus the current tiles[0] URL to detect no-op layer swaps.
  const existingSrc = map.getSource("compare-imagery") as
    | { tiles?: string[] }
    | undefined;
  const existingUrl = existingSrc?.tiles?.[0];
  if (layer && existingUrl === layer.tileUrl && map.getLayer("compare-imagery")) {
    map.setPaintProperty("compare-imagery", "raster-opacity", opacity);
    return;
  }
  if (map.getLayer("compare-imagery")) map.removeLayer("compare-imagery");
  if (map.getSource("compare-imagery")) map.removeSource("compare-imagery");
  if (!layer) return;
  map.addSource("compare-imagery", {
    type: "raster",
    tiles: [layer.tileUrl],
    tileSize: 256,
  });
  map.addLayer({
    id: "compare-imagery",
    type: "raster",
    source: "compare-imagery",
    paint: { "raster-opacity": opacity },
  });
}

export function SplitMap({ selectedArea, imageryLayers, initialCamera: seedCamera, onLoadExamplePair, onRemoveImageryLayer, onExit }: SplitMapProps) {
  // Backend-sourced demos (OlmoEarth FT inference pairs) — fetched lazily
  // at mount. Shown ABOVE the static GIBS presets so users see the
  // OlmoEarth-branded demos first. Failure to fetch (e.g. backend off)
  // silently falls back to just the GIBS list — compare still usable.
  const [olmoDemos, setOlmoDemos] = useState<OlmoEarthDemoPair[]>([]);
  useEffect(() => {
    let cancelled = false;
    getOlmoEarthDemoPairs()
      .then((r) => {
        if (!cancelled) setOlmoDemos(r.pairs ?? []);
      })
      .catch((e) => {
        // Non-fatal — log and keep the GIBS fallback list. Happens when
        // backend is offline or the olmoearth_demos module isn't loaded.
        console.warn("OlmoEarth demo-pairs fetch failed:", e);
      });
    return () => { cancelled = true; };
  }, []);

  // Warm-cache snapshot for the ready-timer badges. Populated on mount +
  // after each pair finishes polling. A repo_id in this set means the next
  // inference on that repo skips the 2–10 s safetensors re-read, so we can
  // honestly tell users "warm (~3 s)" vs "cold (~30 s)" on the button
  // before they click. Backend restart empties it, matching reality.
  const [loadedModels, setLoadedModels] = useState<Set<string>>(new Set());
  const refreshLoadedModels = async () => {
    try {
      const r = await getLoadedOlmoEarthModels();
      setLoadedModels(new Set(r.loaded));
    } catch (e) {
      // Non-fatal — the badge just won't update. The loading spinner
      // still covers the "is it glitched?" question.
      console.warn("loaded-models fetch failed:", e);
    }
  };
  useEffect(() => {
    refreshLoadedModels();
  }, []);

  // Per-pair loading state for the demo buttons. The prebake-then-poll
  // flow can take ~30 s on a cold backend (FT weights load + S2 STAC fetch
  // + forward pass). Without a visible loading indicator the UI looks
  // frozen and users assume the app glitched. We track:
  //   - `a` / `b` booleans for which side is still pending tiles
  //   - `startedAt` ms timestamp so we can show elapsed seconds
  // An entry is present while either side is pending; removed when both
  // are ready OR the 5-min polling budget expires.
  type PairLoadState = { a: boolean; b: boolean; startedAt: number };
  const [loadingPairs, setLoadingPairs] = useState<Record<string, PairLoadState>>({});

  // Stub-state map keyed by layer id. When pollReady sees
  // ``X-Inference-Kind: stub`` on the tile response, that side's raster is
  // the ``_render_stub_tile`` gradient-with-watermark — NOT real FT output.
  // Users otherwise see a colored tile, assume it's real, and get
  // confusing conclusions from the compare. The MiniLegend surfaces a
  // "PREVIEW STUB — inference failed, click to retry" banner when set.
  type StubInfo = { reason: string };
  const [stubByLayerId, setStubByLayerId] = useState<Record<string, StubInfo>>({});
  // Monotonic tick so elapsed-seconds text re-renders without re-running
  // the poll loop. Only fires while at least one pair is loading.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (Object.keys(loadingPairs).length === 0) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [loadingPairs]);

  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const leftMapRef = useRef<maplibregl.Map | null>(null);
  const rightMapRef = useRef<maplibregl.Map | null>(null);
  const compareRef = useRef<MaplibreCompare | null>(null);

  // Default: show the first imagery layer on the left, basemap-only on the
  // right — easiest "before vs after" demo on first click.
  const [leftLayerId, setLeftLayerId] = useState<string | null>(
    imageryLayers[0]?.id ?? null,
  );
  const [rightLayerId, setRightLayerId] = useState<string | null>(null);
  // Per-side opacity for layer comparison. Default both to 1.0 (fully
  // opaque). Sliders in the A/B row let users fade one side to see the
  // basemap + other side showing through — useful for subtle differences
  // in true-color imagery where a hard vertical divider obscures context.
  const [leftOpacity, setLeftOpacity] = useState<number>(1);
  const [rightOpacity, setRightOpacity] = useState<number>(1);

  useEffect(() => {
    if (!leftRef.current || !rightRef.current || !wrapperRef.current) return;
    // Prefer the caller-supplied seed camera (carried over from MapView
    // via App's onCameraChange) so the user's current zoom level is
    // preserved when they click Compare. Fall back to the centroid-at-
    // zoom-8 default only if no seed was given (e.g. direct deep-link
    // into compare mode, or first-run when MapView hasn't moved yet).
    const camera = seedCamera ?? initialCamera(selectedArea);
    const left = new maplibregl.Map({
      container: leftRef.current,
      style: baseStyle("osm"),
      center: camera.center,
      zoom: camera.zoom,
    });
    const right = new maplibregl.Map({
      container: rightRef.current,
      style: baseStyle("osm"),
      center: camera.center,
      zoom: camera.zoom,
    });
    leftMapRef.current = left;
    rightMapRef.current = right;
    const compare = new MaplibreCompare(left, right, wrapperRef.current, {
      orientation: "vertical",
      mousemove: false,
    });
    compareRef.current = compare;

    return () => {
      compare.remove();
      left.remove();
      right.remove();
      compareRef.current = null;
      leftMapRef.current = null;
      rightMapRef.current = null;
    };
    // Only init once — changing layers is handled by separate effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconcile each side's imagery + opacity whenever any of those change.
  // applyImageryLayer's fast-path keeps the same source in place when only
  // opacity changes, so slider drags are smooth (no tile re-fetch).
  useEffect(() => {
    const m = leftMapRef.current;
    if (!m) return;
    const layer = imageryLayers.find((l) => l.id === leftLayerId) ?? null;
    const apply = () => applyImageryLayer(m, layer, leftOpacity);
    if (m.isStyleLoaded()) apply();
    else m.on("load", apply);
  }, [leftLayerId, imageryLayers, leftOpacity]);

  useEffect(() => {
    const m = rightMapRef.current;
    if (!m) return;
    const layer = imageryLayers.find((l) => l.id === rightLayerId) ?? null;
    const apply = () => applyImageryLayer(m, layer, rightOpacity);
    if (m.isStyleLoaded()) apply();
    else m.on("load", apply);
  }, [rightLayerId, imageryLayers, rightOpacity]);

  // Map layer_id → legend_hint for any demo side currently in `olmoDemos`.
  // Used to paint a color-key panel on whichever side of the split the
  // demo is mounted. Users previously saw two colored rasters with no
  // idea what the colors meant — now they see "Mangrove softmax score
  // (uncalibrated)" + the gradient that matches the pixels on screen.
  const legendByLayerId = new Map<string, OlmoEarthLegendHint>();
  for (const pair of olmoDemos) {
    if (pair.a.legend_hint) legendByLayerId.set(pair.a.id, pair.a.legend_hint);
    if (pair.b.legend_hint) legendByLayerId.set(pair.b.id, pair.b.legend_hint);
  }
  const leftLegend = leftLayerId ? legendByLayerId.get(leftLayerId) ?? null : null;
  const rightLegend = rightLayerId ? legendByLayerId.get(rightLayerId) ?? null : null;

  return (
    <div className="w-full h-full relative">
      <div ref={wrapperRef} className="w-full h-full relative">
        {/* Inline `position: absolute; inset: 0` because MapLibre's own
            `.maplibregl-map { position: relative }` rule overrides tailwind
            class-based positioning once it attaches, collapsing height to 0. */}
        <div
          ref={leftRef}
          style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
        />
        <div
          ref={rightRef}
          style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
        />
      </div>

      {/* A/B pickers + Exit — pinned top-left, all in one flex-wrap row.
          Earlier bug: the dropdowns ballooned to ~242 px each to fit long
          option names (e.g. "MODIS Terra · 2019-12-15 (pre-fire)"),
          making a 620 px row overflow the 435 px map area (main = viewport
          - 480 px sidebar) by 209 px. Exit compare ended up offscreen and
          unclickable. Top-right was also bad because the floating LLM
          panel occupies that corner. Fix: cap dropdown widths with
          `max-w-[180px] truncate`, let the row `flex-wrap`, and cap the
          container at `max-w-[calc(100%-160px)]` so Exit can't be pushed
          past the map's right edge. `flex-shrink-0` on Exit guarantees it
          stays at its natural width even if A/B wrap to a second line. */}
      {/* z-50 (above DraggablePanel's z-40) so the LLM floating panel
          can't obscure the compare controls. The panel is still draggable,
          but these controls remain always clickable on top. */}
      <div className="absolute top-6 left-6 max-w-[calc(100%-24px)] flex flex-wrap gap-2 z-50 items-center bg-gradient-panel backdrop-blur-sm px-3 py-2 rounded-xl shadow-md border border-geo-border">
        {/* Exit placed FIRST (leftmost) so the LLM floating panel —
            which users often park in the top-right — can never overlap
            it. The row is pinned at left-6 of the map area, anchored
            left, so Exit's position is stable regardless of A/B dropdown
            width. */}
        <button
          onClick={onExit}
          data-testid="split-map-exit"
          title="Exit compare — return to the single map view"
          className="flex-shrink-0 px-3 py-1.5 rounded-md text-[11px] font-semibold border border-geo-border bg-geo-surface text-geo-text hover:border-red-500 hover:text-red-600 hover:bg-red-50 cursor-pointer transition-colors inline-flex items-center gap-1"
        >
          <span className="text-[13px] leading-none">×</span>
          <span>Exit</span>
        </button>
        <label className="flex items-center gap-2 text-xs text-geo-text min-w-0">
          <span className="uppercase tracking-wider text-geo-muted flex-shrink-0">A</span>
          <select
            value={leftLayerId ?? ""}
            onChange={(e) => setLeftLayerId(e.target.value || null)}
            title={imageryLayers.find((l) => l.id === leftLayerId)?.label ?? "basemap only"}
            className="px-2 py-1 bg-geo-surface border border-geo-border rounded text-xs max-w-[180px] truncate cursor-pointer"
          >
            <option value="">(basemap only)</option>
            {imageryLayers.map((l) => (
              <option key={l.id} value={l.id}>{l.label ?? l.id}</option>
            ))}
          </select>
          {/* Opacity slider A — 0 → fully see basemap, 1 → fully opaque
              layer. Hidden when no layer is picked (nothing to fade). */}
          {leftLayerId && (
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={leftOpacity}
              onChange={(e) => setLeftOpacity(Number(e.target.value))}
              title={`A opacity: ${Math.round(leftOpacity * 100)}%`}
              className="w-16 accent-geo-accent cursor-pointer flex-shrink-0"
              data-testid="split-map-opacity-a"
            />
          )}
        </label>
        <label className="flex items-center gap-2 text-xs text-geo-text min-w-0">
          <span className="uppercase tracking-wider text-geo-muted flex-shrink-0">B</span>
          <select
            value={rightLayerId ?? ""}
            onChange={(e) => setRightLayerId(e.target.value || null)}
            title={imageryLayers.find((l) => l.id === rightLayerId)?.label ?? "basemap only"}
            className="px-2 py-1 bg-geo-surface border border-geo-border rounded text-xs max-w-[180px] truncate cursor-pointer"
          >
            <option value="">(basemap only)</option>
            {imageryLayers.map((l) => (
              <option key={l.id} value={l.id}>{l.label ?? l.id}</option>
            ))}
          </select>
          {rightLayerId && (
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={rightOpacity}
              onChange={(e) => setRightOpacity(Number(e.target.value))}
              title={`B opacity: ${Math.round(rightOpacity * 100)}%`}
              className="w-16 accent-geo-accent cursor-pointer flex-shrink-0"
              data-testid="split-map-opacity-b"
            />
          )}
        </label>
      </div>

      {/* Per-side mini-legends stacked in the top-right gutter. Earlier
          placement (A at bottom-left, B at bottom-right) collided with
          the demo-pairs card at bottom-left — the A legend was invisible
          behind it. Top-right is the one empty quadrant (A/B controls
          live in top-left, demo-pairs card in bottom-left, opacity slider
          at bottom-right would overlap too). Stacking A above B keeps the
          natural reading order (left raster = A = first legend) even
          though spatial correspondence with the divider is lost — the
          internal A/B badges carry that association. pointer-events-none
          on the wrapper + pointer-events-auto inside each card means the
          legend doesn't steal map pan/zoom interactions. */}
      {(leftLegend || rightLegend) && (
        <div
          className="absolute top-20 right-6 pointer-events-none z-10 flex flex-col gap-2 items-end"
          data-testid="split-legend-stack"
        >
          {leftLegend && (
            <div data-testid={`split-legend-A-${leftLegend.colormap}`}>
              <MiniLegend
                side="A"
                legend={leftLegend}
                stub={leftLayerId ? stubByLayerId[leftLayerId] ?? null : null}
              />
            </div>
          )}
          {rightLegend && (
            <div data-testid={`split-legend-B-${rightLegend.colormap}`}>
              <MiniLegend
                side="B"
                legend={rightLegend}
                stub={rightLayerId ? stubByLayerId[rightLayerId] ?? null : null}
              />
            </div>
          )}
        </div>
      )}

      {/* OlmoEarth-sourced compare demos — fetched from the backend
          `/api/olmoearth/demo-pairs` registry. Each pair runs a real FT
          model (Mangrove / AWF / EcosystemTypeMapping) on Sentinel-2 L2A
          composites at two different dates/regions. Single-click loads A
          + B (and kicks off lazy inference via prebakeOlmoEarthDemo so
          tiles start rendering server-side); double-click drops both.
          Mirrors SampleData's load/drop UX so users already trained on
          the sample-data pattern find this familiar. */}
      {onLoadExamplePair && olmoDemos.length > 0 && (
        <div className="absolute bottom-6 left-6 bg-gradient-panel border border-geo-border backdrop-blur text-geo-text px-4 py-3 rounded-lg text-[11px] z-10 max-w-sm shadow-md space-y-2">
          <div>
            <div className="font-semibold text-geo-accent mb-0.5">OlmoEarth demo pairs</div>
            <div className="text-geo-muted leading-snug">
              <span className="font-semibold text-geo-text">Click</span> to load A + B ·
              <span className="font-semibold text-geo-text"> double-click</span> to drop.
              Tiles are generated on-demand by the cached FT heads
              (Mangrove / AWF / Ecosystem). First click takes ~30 s while
              Sentinel-2 is fetched + the model runs; subsequent loads are
              instant from the on-disk cache.
            </div>
          </div>
          <div className="space-y-1.5">
            {olmoDemos.map((pair) => {
              // Build ImageryLayer-shaped objects from the backend's
              // demo sides so the add/drop plumbing doesn't change.
              const aLayer: ImageryLayer = {
                id: pair.a.id,
                label: pair.a.label,
                tileUrl: pair.a.tile_url,
              };
              const bLayer: ImageryLayer = {
                id: pair.b.id,
                label: pair.b.label,
                tileUrl: pair.b.tile_url,
              };
              const loaded =
                imageryLayers.some((l) => l.id === aLayer.id) &&
                imageryLayers.some((l) => l.id === bLayer.id);
              const loading = loadingPairs[pair.id];
              const handleLoad = async () => {
                if (loaded || loading) return;
                // Flip on the loading indicator BEFORE network work begins
                // so the button re-renders immediately on click — otherwise
                // the user stares at an unchanged button for ~30 s and
                // concludes the app glitched. Removed only when both sides
                // flip to status==ready (or the polling budget runs out).
                setLoadingPairs((prev) => ({
                  ...prev,
                  [pair.id]: { a: true, b: true, startedAt: Date.now() },
                }));
                // CRITICAL: await prebake BEFORE adding the imagery layers.
                // Otherwise MapLibre attaches the raster source and fires
                // tile requests in the ~50 ms before the backend has
                // registered the job entries via start_inference, and
                // those first tile requests hit an unknown job_id. We
                // now serve transparent+no-store PNGs on unknown ids
                // (see olmoearth.py) but awaiting here avoids the round-
                // trip cost entirely. Prebake returns in <100 ms because
                // the server uses asyncio.create_task.
                try {
                  await prebakeOlmoEarthDemo(pair.id);
                } catch (e) {
                  console.warn("OlmoEarth demo prebake failed:", e);
                }
                onLoadExamplePair(aLayer, bLayer);
                setLeftLayerId(aLayer.id);
                setRightLayerId(bLayer.id);
                setLeftOpacity(1);
                setRightOpacity(1);
                const bounds: maplibregl.LngLatBoundsLike = [
                  [pair.fit_bbox.west, pair.fit_bbox.south],
                  [pair.fit_bbox.east, pair.fit_bbox.north],
                ];
                leftMapRef.current?.fitBounds(bounds, { padding: 40, duration: 600 });
                rightMapRef.current?.fitBounds(bounds, { padding: 40, duration: 600 });
                // Poll for job completion on both sides and force the
                // compare-imagery source to re-query tiles once each job
                // flips to status==ready. Without this, the initial
                // transparent placeholders would stay on screen until the
                // user panned/zoomed (the backend now sends no-store on
                // pending tiles, but the already-attached source doesn't
                // retry on its own unless we poke it).
                const pollReady = async (
                  jobId: string,
                  side: "left" | "right",
                  loadKey: "a" | "b",
                ) => {
                  for (let i = 0; i < 60; i++) { // up to 5 minutes
                    await new Promise((r) => setTimeout(r, 5000));
                    try {
                      // Use GET (not HEAD — FastAPI's GET route doesn't
                      // auto-respond to HEAD, returning 405). The tile
                      // coords (10/0/0) are nominal — the response shape
                      // (cache-control header) is what we care about; the
                      // 334-byte transparent PNG body is discarded.
                      const r = await fetch(`/api/olmoearth/infer-tile/${jobId}/10/0/0.png`);
                      const cc = r.headers.get("cache-control") ?? "";
                      // `public, max-age=3600` means status==ready. While
                      // the job is still running, the backend sends
                      // `no-store` (see cache-header switch).
                      if (cc.includes("max-age")) {
                        const map = side === "left" ? leftMapRef.current : rightMapRef.current;
                        const src = map?.getSource("compare-imagery") as
                          | { reload?: () => void }
                          | undefined;
                        src?.reload?.();
                        // The backend tags the tile response with the job
                        // kind. "stub" means real inference failed (PC
                        // STAC hiccup, parallel-prebake race, transient
                        // network glitch); the raster the user is looking
                        // at is a preview gradient + watermark, NOT real
                        // FT output. Record that so the MiniLegend can
                        // shout "preview stub — click to retry" instead
                        // of silently showing a fake result.
                        const kind = r.headers.get("x-inference-kind") ?? "unknown";
                        const reason = r.headers.get("x-inference-stub-reason") ?? "";
                        const layerId = `demo-${jobId}`;
                        if (kind === "stub") {
                          setStubByLayerId((prev) => ({
                            ...prev,
                            [layerId]: { reason: reason || "real inference failed" },
                          }));
                        } else {
                          // Clear any prior stub marker (user may have
                          // retried after a previous failure).
                          setStubByLayerId((prev) => {
                            if (!(layerId in prev)) return prev;
                            const { [layerId]: _cleared, ...rest } = prev;
                            return rest;
                          });
                        }
                        // The first side to flip ready has forced its FT
                        // head through load_encoder, so by the time the
                        // second side finishes the repo is in warm cache.
                        // Re-fetch the snapshot so sibling demo buttons
                        // (different pairs using a different head) still
                        // reflect accurate state, and so re-clicking the
                        // SAME pair after a drop correctly shows "warm".
                        refreshLoadedModels();
                        setLoadingPairs((prev) => {
                          const cur = prev[pair.id];
                          if (!cur) return prev;
                          const next = { ...cur, [loadKey]: false };
                          // Both sides done → remove the loading entry
                          // entirely so the button snaps to the "loaded"
                          // state and the elapsed-seconds ticker stops.
                          if (!next.a && !next.b) {
                            const { [pair.id]: _done, ...rest } = prev;
                            return rest;
                          }
                          return { ...prev, [pair.id]: next };
                        });
                        return;
                      }
                    } catch { /* keep polling */ }
                  }
                  // Ran out of polling budget — clear the loading flag
                  // so the user isn't stuck on an infinite spinner. The
                  // tiles will still appear once the backend finishes
                  // (MapLibre auto-refreshes when the user pans/zooms).
                  setLoadingPairs((prev) => {
                    const cur = prev[pair.id];
                    if (!cur) return prev;
                    const next = { ...cur, [loadKey]: false };
                    if (!next.a && !next.b) {
                      const { [pair.id]: _done, ...rest } = prev;
                      return rest;
                    }
                    return { ...prev, [pair.id]: next };
                  });
                };
                pollReady(pair.a.job_id, "left", "a");
                pollReady(pair.b.job_id, "right", "b");
              };
              const handleDrop = () => {
                if (!loaded || !onRemoveImageryLayer) return;
                onRemoveImageryLayer(aLayer.id);
                onRemoveImageryLayer(bLayer.id);
                if (leftLayerId === aLayer.id || leftLayerId === bLayer.id) setLeftLayerId(null);
                if (rightLayerId === aLayer.id || rightLayerId === bLayer.id) setRightLayerId(null);
              };
              const elapsedS = loading
                ? Math.floor((Date.now() - loading.startedAt) / 1000)
                : 0;
              // Surface the current stage via the number of pending sides.
              // Two sides pending → still in early S2 + weights-load phase.
              // One side pending → the faster side finished, the slow one
              // is still running forward pass / rastering.
              const loadingStage = loading
                ? loading.a && loading.b
                  ? "fetching Sentinel-2 + loading FT head"
                  : "finishing last side"
                : null;
              // Both sides of a pair use the same FT head repo_id, so one
              // membership check tells us the warm path is available. If
              // the head hasn't been loaded this backend lifetime the click
              // will pay the ~30 s cold cost; otherwise it's ~3 s.
              const pairHead = pair.a.spec.model_repo_id;
              const isWarm = loadedModels.has(pairHead);
              const readyEta = isWarm ? "~3 s" : "~30 s";
              const readyBadge: { text: string; className: string; title: string } = isWarm
                ? {
                    text: `warm · ${readyEta}`,
                    className: "bg-geo-success/15 text-geo-success border-geo-success/40",
                    title: `FT head already in memory. Click cost is S2 fetch + forward pass only (~3 s).`,
                  }
                : {
                    text: `cold · ${readyEta}`,
                    className: "bg-geo-muted/15 text-geo-muted border-geo-border",
                    title: `FT head not yet loaded this backend lifetime. First click will load weights from disk (~30 s); subsequent clicks on this pair are warm.`,
                  };
              return (
                <button
                  key={pair.id}
                  type="button"
                  onClick={handleLoad}
                  onDoubleClick={handleDrop}
                  disabled={!!loading}
                  title={
                    loading
                      ? `Loading — ${loadingStage} (${elapsedS}s elapsed). First click per backend lifetime takes ~30 s.`
                      : loaded
                      ? "Loaded — double-click to drop both A and B from the map"
                      : "Click to load A and B onto the split maps (triggers inference)"
                  }
                  className={`w-full text-left px-3 py-2 rounded-lg transition-all border ${
                    loading
                      ? "border-geo-accent/60 bg-geo-accent/10 cursor-wait"
                      : loaded
                      ? "border-geo-success/50 bg-geo-success/5 hover:bg-red-50 hover:border-red-300 cursor-pointer"
                      : "border-geo-border bg-geo-surface hover:border-geo-accent hover:bg-geo-bg hover:-translate-y-px hover:shadow-sm active:translate-y-0 active:shadow-none cursor-pointer"
                  }`}
                  data-testid={`compare-example-${pair.id}`}
                >
                  <div className="flex items-start gap-2">
                    {loading ? (
                      // Amber spinner reads as "actively working" in the
                      // same visual idiom as the Cesium init-error banner
                      // — users already associate this palette with async
                      // work in the app.
                      <svg
                        className="mt-0.5 w-3 h-3 flex-shrink-0 animate-spin text-geo-accent"
                        viewBox="0 0 24 24"
                        fill="none"
                        aria-hidden="true"
                      >
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" />
                        <path d="M22 12a10 10 0 0 1-10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      </svg>
                    ) : (
                      <span
                        className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${
                          loaded ? "bg-geo-success" : "bg-geo-border"
                        }`}
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="font-semibold text-geo-text text-[12px] truncate">
                          {pair.title}
                        </span>
                        {!loading && !loaded && (
                          <span
                            className={`text-[9px] px-1.5 py-px rounded-full border flex-shrink-0 font-medium ${readyBadge.className}`}
                            title={readyBadge.title}
                          >
                            {readyBadge.text}
                          </span>
                        )}
                      </div>
                      {loading ? (
                        <div className="text-geo-accent text-[10px] leading-snug font-medium">
                          Loading… {loadingStage} · {elapsedS}s
                        </div>
                      ) : (
                        <div className="text-geo-muted text-[10px] leading-snug">{pair.blurb}</div>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="text-[9px] text-geo-muted leading-snug pt-1 border-t border-geo-border">
            Source: OlmoEarth FT heads + Sentinel-2 L2A (Microsoft Planetary Computer STAC).
          </div>
        </div>
      )}
      {onLoadExamplePair && olmoDemos.length === 0 && (
        <div className="absolute bottom-6 left-6 bg-gradient-panel border border-geo-border backdrop-blur text-geo-text px-4 py-3 rounded-lg text-[11px] z-10 max-w-sm shadow-md">
          <div className="font-semibold text-geo-muted mb-0.5">Loading OlmoEarth demos…</div>
          <div className="text-geo-muted leading-snug">
            If this persists, check that the backend is running — demos come from
            <span className="font-mono"> /api/olmoearth/demo-pairs</span>.
          </div>
        </div>
      )}
    </div>
  );
}
