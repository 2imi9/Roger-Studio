import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import type { BBox, AnalysisResult, DatasetInfo, EnvDataResult, ViewMode } from "./types";
import type { DataSourceConfig } from "./components/DataSourcePicker";
import { DEFAULT_CONFIG } from "./components/DataSourcePicker";
import { MapView, type ImageryLayer } from "./components/MapView";
import { Sidebar } from "./components/Sidebar";
import { SplitMap } from "./components/SplitMap";
import { InferenceLegendPanel } from "./components/InferenceLegendPanel";
import { colorForTag, type LabelFeature, type LabelMode } from "./components/LabelPanel";
import { analyze, getEnvData, getOlmoEarthCacheStatus, type OlmoEarthRepoStatus } from "./api/client";
import { DATASET_COVERAGE } from "./constants/olmoEarthCoverage";
import { safeSetItem } from "./util/sessionStorage";

// sessionStorage namespace for the label set — same per-tab pattern as the
// chat panes use (geoenv.llm.*, geoenv.cloud.*, geoenv.claude.*). Closes
// when the tab does; cross-session persistence is deliberately deferred.
const SS_LABEL_PROJECT = "geoenv.label.project";
const SS_LABEL_FEATURES = "geoenv.label.features";
const SS_LABEL_CUSTOM_TAGS = "geoenv.label.customTags";
const SS_LABEL_MODE = "geoenv.label.mode";

function readSession<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function newId(): string {
  return `lbl_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

// Kenyan coast near Mombasa — chosen so the demo lights up the full MCP
// chain: bbox intersects BOTH OlmoEarth project regions (Mangrove tropical
// belt + AWF southern Kenya), so query_olmoearth returns real project hits
// instead of the empty state SF used to show.
const DEFAULT_BBOX: BBox = { west: 39.6, south: -4.2, east: 40.0, north: -3.9 };

// (ErrorBoundary class was here — only consumer was the dropped 3D Globe
// view. Re-introduce if any future view needs render-time error containment.)

function polygonBounds(poly: GeoJSON.Polygon): BBox {
  const ring = poly.coordinates[0] ?? [];
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return {
    west: Math.min(...lons),
    south: Math.min(...lats),
    east: Math.max(...lons),
    north: Math.max(...lats),
  };
}

export function App() {
  const [viewMode, setViewMode] = useState<ViewMode>("map");
  const [selectedArea, setSelectedArea] = useState<BBox | null>(null);
  // Set alongside selectedArea when the user draws an actual polygon (terra-draw).
  // When null, PolygonStats / analyze fall back to a 4-vertex rectangle from bbox.
  const [selectedGeometry, setSelectedGeometry] = useState<GeoJSON.Polygon | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetInfo | null>(null);
  const [envData, setEnvData] = useState<EnvDataResult | null>(null);
  const [dataSourceConfig, setDataSourceConfig] = useState<DataSourceConfig>(DEFAULT_CONFIG);
  const [imageryLayers, setImageryLayers] = useState<ImageryLayer[]>([]);
  // Compare-mode swaps the 2D MapView for a side-by-side SplitMap so the user
  // can pit two inference layers (or a prediction vs a basemap) against each
  // other. Off by default — only the "Compare" affordance turns it on.
  const [compareMode, setCompareMode] = useState<boolean>(false);
  // Latest camera state reported by MapView via onCameraChange. Passed
  // as a seed to SplitMap so entering compare preserves the user's
  // zoom + pan instead of resetting to zoom=8 at the bbox centroid.
  // Stored in a ref (not state) because we don't need React to re-
  // render on every moveend — we just need a stable pointer to read at
  // compare-mode entry. A state would force a re-render of everything
  // downstream of App on every map pan.
  const mapCameraRef = useRef<{ center: [number, number]; zoom: number } | null>(null);
  const [olmoCache, setOlmoCache] = useState<Record<string, OlmoEarthRepoStatus>>({});
  // visibleDataLayers still feeds the overlayGeojson memo so coverage
  // polygons would render if ever populated again — the toggler that
  // populated it (handleToggleDataLayer) was removed 2026-04-21 with the
  // Labeling sub-view. State kept as an empty Set so no render paths break.
  const [visibleDataLayers] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Labeling MVP state — persisted to sessionStorage so a tab refresh keeps
  // the in-progress label set. Closes when the tab closes (no backend save
  // in MVP — researchers download GeoJSON to keep work durable).
  const [projectName, setProjectName] = useState<string>(() =>
    readSession<string>(SS_LABEL_PROJECT, "Untitled label set"),
  );
  const [labeledFeatures, setLabeledFeatures] = useState<LabelFeature[]>(() =>
    readSession<LabelFeature[]>(SS_LABEL_FEATURES, []),
  );
  const [customTags, setCustomTags] = useState<string[]>(() =>
    readSession<string[]>(SS_LABEL_CUSTOM_TAGS, []),
  );
  const [labelMode, setLabelMode] = useState<LabelMode>(() =>
    readSession<LabelMode>(SS_LABEL_MODE, { active: false, type: "polygon", tag: "Forest" }),
  );

  useEffect(() => { safeSetItem(SS_LABEL_PROJECT, JSON.stringify(projectName)); }, [projectName]);
  // Labeled features can accumulate 100+ polygons → the big sessionStorage
  // writer on this page. Flagged for eviction so it's first on the chopping
  // block if a chat pane writes run up against the 5 MB quota.
  useEffect(() => {
    safeSetItem(SS_LABEL_FEATURES, JSON.stringify(labeledFeatures), { trackForEviction: true });
  }, [labeledFeatures]);
  useEffect(() => { safeSetItem(SS_LABEL_CUSTOM_TAGS, JSON.stringify(customTags)); }, [customTags]);
  useEffect(() => { safeSetItem(SS_LABEL_MODE, JSON.stringify(labelMode)); }, [labelMode]);

  const handleLabelDrawn = useCallback(
    (geometry: GeoJSON.Point | GeoJSON.Polygon | GeoJSON.LineString, type: "point" | "polygon" | "line") => {
      const tag = labelModeRef.current.tag || "(untagged)";
      const customs = customTagsRef.current;
      const color = colorForTag(tag, customs);
      const feat: LabelFeature = {
        type: "Feature",
        id: newId(),
        properties: {
          tag,
          label_type: type,
          color,
          class_name: tag,
          metadata: {},
          created_at: new Date().toISOString(),
          source: "manual",
        },
        geometry,
      };
      setLabeledFeatures((prev) => [...prev, feat]);
    },
    [],
  );

  const handleDeleteLabel = useCallback((id: string) => {
    setLabeledFeatures((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const handleClearLabels = useCallback(() => {
    setLabeledFeatures([]);
  }, []);

  const handleAddCustomTag = useCallback((tag: string) => {
    setCustomTags((prev) => (prev.includes(tag) ? prev : [...prev, tag]));
  }, []);

  // Removing a custom tag does NOT delete labeled features that use it —
  // their `properties.tag` stays put and they keep rendering with the same
  // hash-fallback color (colorForTag is stable for any string). The user
  // deletes those individually from the feature list if they want them gone.
  // Confirmation prompt with affected-feature count lives in LabelPanel.
  const handleRemoveCustomTag = useCallback((tag: string) => {
    setCustomTags((prev) => prev.filter((t) => t !== tag));
  }, []);

  // Refs for callbacks the once-attached terra-draw handler needs to read at
  // draw-finish time (avoids re-creating handleLabelDrawn on every render).
  const labelModeRef = useRef(labelMode);
  useEffect(() => { labelModeRef.current = labelMode; }, [labelMode]);
  const customTagsRef = useRef(customTags);
  useEffect(() => { customTagsRef.current = customTags; }, [customTags]);

  const handleAreaSelect = useCallback((bbox: BBox) => {
    setSelectedArea(bbox);
    setSelectedGeometry(null);
    setError(null);
  }, []);

  const handleGeometrySelect = useCallback((polygon: GeoJSON.Polygon) => {
    setSelectedGeometry(polygon);
    setSelectedArea(polygonBounds(polygon));
    setError(null);
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedGeometry(null);
    setSelectedArea(null);
    setAnalysisResult(null);
    setQueryPixel(null);
    setPickQueryActive(false);
  }, []);

  // Embedding-tools query pixel: lifted to App so MapView can install a
  // one-shot click capture handler AND OlmoEarthImport (in either Sidebar
  // or MapView popover) can show / clear / use the picked location. The
  // picked pixel becomes the cosine-similarity query when the user runs
  // the Similarity tool; with no pick set, the backend falls back to AOI
  // center (existing behaviour).
  const [queryPixel, setQueryPixel] = useState<{ lon: number; lat: number } | null>(null);
  const [pickQueryActive, setPickQueryActive] = useState(false);
  const handleStartPickQuery = useCallback(() => setPickQueryActive(true), []);
  const handleQueryPixelPicked = useCallback(
    (lon: number, lat: number) => {
      setQueryPixel({ lon, lat });
      setPickQueryActive(false);
    },
    [],
  );
  const handleClearQueryPixel = useCallback(() => {
    setQueryPixel(null);
    setPickQueryActive(false);
  }, []);

  const handleAddImageryLayer = useCallback((layer: ImageryLayer) => {
    setImageryLayers((prev) => {
      // Replace any existing layer with the same id; otherwise append
      const i = prev.findIndex((l) => l.id === layer.id);
      if (i >= 0) {
        const next = prev.slice();
        next[i] = layer;
        return next;
      }
      return [...prev, layer];
    });
  }, []);

  const handleRemoveImageryLayer = useCallback((id: string) => {
    setImageryLayers((prev) => prev.filter((l) => l.id !== id));
  }, []);

  // Merge the current label FeatureCollection into imageryLayers as a
  // single vector layer. Uses a deterministic id ("labels-<project>") so:
  //   1. The Added Layer "On map" list stays clean (no duplicate rows
  //      per click — re-clicking updates the existing layer's features).
  //   2. `mergedToMapLayers` (below) can check membership by id without
  //      scanning features.
  //   3. Removing via Added Layer's × flips this membership off, so the
  //      LabelPanel button re-enables automatically.
  const labelsLayerId = useMemo(() => {
    const safe = (projectName || "labels").trim().replace(/[^a-z0-9-_]+/gi, "_") || "labels";
    return `labels-${safe.toLowerCase()}`;
  }, [projectName]);
  const handleMergeLabelsToMapLayers = useCallback(() => {
    if (labeledFeatures.length === 0) return;
    handleAddImageryLayer({
      id: labelsLayerId,
      label: `Labels: ${projectName || "Untitled"} (${labeledFeatures.length})`,
      featureCollection: {
        type: "FeatureCollection",
        features: labeledFeatures as unknown as GeoJSON.Feature[],
      },
    });
  }, [labelsLayerId, labeledFeatures, projectName, handleAddImageryLayer]);
  const labelsMergedToMapLayers = useMemo(
    () => imageryLayers.some((l) => l.id === labelsLayerId),
    [imageryLayers, labelsLayerId],
  );

  // One-click A/B example loader for the SplitMap compare view. Mirrors the
  // OlmoEarth Studio "Nigeria Mangrove 2018 vs 2024" preset affordance —
  // adds both tile layers to imageryLayers so the A/B dropdowns can
  // reference them, and SplitMap sets its left/right picks on its side.
  //
  // All preset-sourced layer ids share the ``example-`` prefix so we can
  // cleanly strip them on exit without touching user-loaded rasters
  // (STAC scenes / uploaded GeoTIFFs / OlmoEarth inference tiles).
  const handleLoadExamplePair = useCallback(
    (a: ImageryLayer, b: ImageryLayer) => {
      setImageryLayers((prev) => {
        // Dedupe by id — replace if already present, else append.
        const next = prev.filter((l) => l.id !== a.id && l.id !== b.id);
        return [...next, a, b];
      });
    },
    [],
  );

  // Exit-compare handler: drop all preset-sourced layers and return to
  // single-map view. Two prefixes qualify as "preset":
  //   ``example-*``  — the old NASA-GIBS static presets (now removed, but
  //                    keep filtering for backward compat if any linger)
  //   ``demo-*``     — OlmoEarth FT demo-pair layers (Mangrove / AWF /
  //                    Ecosystem) sourced from /api/olmoearth/demo-pairs
  // User-loaded imagery (STAC scenes, uploaded GeoTIFFs, OlmoEarth user
  // inference) uses different id patterns and is preserved so the user
  // doesn't lose real work. Matches the spec: "when we exit compare we
  // just drop the [demo] map layers" (real datasets stay).
  const handleExitCompare = useCallback(() => {
    setImageryLayers((prev) =>
      prev.filter((l) => !l.id.startsWith("example-") && !l.id.startsWith("demo-")),
    );
    setCompareMode(false);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedArea) return;
    setLoading(true);
    setError(null);
    try {
      const [result, env] = await Promise.all([
        analyze({ area: selectedArea }),
        getEnvData(selectedArea).catch(() => null),
      ]);
      setAnalysisResult(result);
      setEnvData(env);
      // Stay on current view — don't force switch to map
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [selectedArea]);

  const handleDemo = useCallback(async () => {
    setSelectedArea(DEFAULT_BBOX);
    setLoading(true);
    setError(null);
    try {
      const [result, env] = await Promise.all([
        analyze({ area: DEFAULT_BBOX }),
        getEnvData(DEFAULT_BBOX).catch(() => null),
      ]);
      setAnalysisResult(result);
      setEnvData(env);
      // Stay on current view
    } catch (e) {
      setError(e instanceof Error ? e.message : "Demo failed");
    } finally {
      setLoading(false);
    }
  }, []);

  // Bumped after every new upload — MapView watches it and fitBounds on
  // change. We can't just react to selectedArea changes because the user-draw
  // path also sets selectedArea (and the rect they drew is already in view —
  // re-fitting would feel like the map is fighting them). A bumped counter
  // disambiguates: only handleUpload (or sample load) increments it.
  const [flyToTrigger, setFlyToTrigger] = useState<{ bbox: BBox; nonce: number } | null>(null);

  // OlmoEarthImport's "Use demo AOI" shortcut — sets the AOI to a model-
  // appropriate ~3 km bbox AND pans the map there, so the user immediately
  // sees what region the demo will run over. Distinct from handleUpload
  // because there's no dataset to register; just an AOI + map fly.
  const handleDemoAreaSelect = useCallback((bbox: BBox) => {
    setSelectedArea(bbox);
    setSelectedGeometry(null);
    setError(null);
    setFlyToTrigger({ bbox, nonce: Date.now() });
  }, []);

  const handleUpload = useCallback((ds: DatasetInfo) => {
    setDatasets((prev) => [...prev, ds]);
    const bbox = ds.raster?.bbox || ds.vector?.bbox || ds.point_cloud?.bbox || ds.multidim?.bbox;
    if (bbox) {
      setSelectedArea(bbox);
      setSelectedGeometry(null);
      // Tell MapView to actually pan + zoom to the new data. Without this,
      // sample / upload loads look like nothing happened (data lands far
      // off-screen on the default US-wide view).
      setFlyToTrigger({ bbox, nonce: Date.now() });
    }
  }, []);

  const handleDeleteDataset = useCallback((filename: string) => {
    setDatasets((prev) => prev.filter((d) => d.filename !== filename));
    // Also deselect if this was the selected dataset
    setSelectedDataset((prev) => (prev?.filename === filename ? null : prev));
  }, []);

  const handleAutoLabelResult = useCallback(
    (geojson: GeoJSON.FeatureCollection, _meta: Record<string, unknown>) => {
      // Add auto-labeled features as an overlay on the map
      setDatasets((prev) => {
        // Local timezone date+time for a readable, collision-free filename.
        // Filename doubles as dataset key (see handleDeleteDataset), so keep
        // seconds-level uniqueness — two same-day runs mustn't share a key.
        const d = new Date();
        const stamp = `${d.toLocaleDateString("sv-SE")}-${d
          .toLocaleTimeString("sv-SE", { hour12: false })
          .replace(/:/g, "")}`;
        const overlay: DatasetInfo = {
          filename: `auto-label-${stamp}.geojson`,
          format: "geojson",
          size_bytes: JSON.stringify(geojson).length,
          raster: null,
          vector: {
            geometry_type: "Polygon",
            feature_count: geojson.features.length,
            crs: { epsg: 4326, wkt: null },
            bbox: null,
            columns: ["class_name", "confidence", "needs_review"],
            sample_properties: null,
          },
          point_cloud: null,
          multidim: null,
          preview_geojson: geojson,
        };
        return [...prev, overlay];
      });
    },
    []
  );

  const handleSelectDataset = useCallback((ds: DatasetInfo) => {
    setSelectedDataset((prev) => (prev?.filename === ds.filename ? null : ds));
    const bbox = ds.raster?.bbox || ds.vector?.bbox || ds.point_cloud?.bbox || ds.multidim?.bbox;
    if (bbox) {
      setSelectedArea(bbox);
      setSelectedGeometry(null);
    }
  }, []);

  // (``activeBbox`` derivation removed with the 3D Globe — no remaining
  // consumer uses the "selectedArea ?? DEFAULT_BBOX" fallback shape.)

  // Poll OlmoEarth cache status so we can surface cached datasets as toggleable
  // Cache status feeds OlmoEarthImport's "cached / loading / error"
  // pill. Earlier 2 s poll was overkill and spammed the backend logs
  // with hundreds of GETs/minute even while no download was in flight;
  // bumped to 30 s. A user who kicks off a repo download via the
  // Import tab will see the state flip within 30 s, which matches the
  // human cadence of "did it finish yet?" checking. The OlmoEarth tab's
  // own polling (when opened) can still poll faster if that view needs
  // tighter feedback.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getOlmoEarthCacheStatus();
        if (!cancelled) setOlmoCache(s.repos);
      } catch {
        /* transient — ignore */
      }
    };
    tick();
    const handle = window.setInterval(tick, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, []);

  // handleToggleDataLayer toggled OlmoEarth coverage polygons on the map.
  // The UI that surfaced those toggles (Sidebar's "OlmoEarth Data Layers ·
  // Live" section) was removed 2026-04-21. visibleDataLayers stays empty by
  // default, so the overlayGeojson assembly below produces no coverage
  // polygons; `olmoCache` is still polled because OlmoEarthPanel reads it.

  // Collect all GeoJSON previews:
  //   1. Uploaded datasets (user data + auto-label outputs)
  //   2. OlmoEarth dataset coverage polygons — only for repos that are BOTH
  //      cached locally AND toggled on. The coverage is visual context ("this
  //      dataset covers the tropical belt"), not the actual samples.
  const overlayGeojson = useMemo(() => {
    const out: (GeoJSON.Feature | GeoJSON.FeatureCollection)[] = datasets
      .filter((d) => d.preview_geojson)
      .map((d) => d.preview_geojson!);
    for (const repoId of visibleDataLayers) {
      const coverage = DATASET_COVERAGE[repoId];
      const cached = olmoCache[repoId]?.status === "cached";
      if (!coverage || !cached) continue;
      out.push({
        type: "Feature",
        properties: {
          name: coverage.name,
          class_name: coverage.name,
          color: coverage.color,
          repo_id: repoId,
          source: "olmoearth-coverage",
        },
        geometry: coverage.geometry,
      });
    }
    // Manual labels go through the same render pipe — color comes from
    // properties.color (set at draw time per tag), class_name fuels the
    // existing hover popup. Point/Line/Polygon all already have layers in
    // MapView's overlay-fill / overlay-line / overlay-circle stack.
    if (labeledFeatures.length > 0) {
      out.push({
        type: "FeatureCollection",
        features: labeledFeatures as unknown as GeoJSON.Feature[],
      });
    }
    return out;
  }, [datasets, visibleDataLayers, olmoCache, labeledFeatures]);

  return (
    <div className="flex h-screen font-sans">
      <Sidebar
        viewMode={viewMode}
        onViewChange={setViewMode}
        selectedArea={selectedArea}
        selectedGeometry={selectedGeometry}
        onSelectDemoArea={handleDemoAreaSelect}
        onClearSelection={handleClearSelection}
        queryPixel={queryPixel}
        pickQueryActive={pickQueryActive}
        onStartPickQuery={handleStartPickQuery}
        onClearQueryPixel={handleClearQueryPixel}
        imageryLayers={imageryLayers}
        onAddImageryLayer={handleAddImageryLayer}
        onRemoveImageryLayer={handleRemoveImageryLayer}
        analysisResult={analysisResult}
        envData={envData}
        datasets={datasets}
        selectedDataset={selectedDataset}
        dataSourceConfig={dataSourceConfig}
        onDataSourceChange={setDataSourceConfig}
        loading={loading}
        error={error}
        onAnalyze={handleAnalyze}
        onDemo={handleDemo}
        onUpload={handleUpload}
        onDeleteDataset={handleDeleteDataset}
        onSelectDataset={handleSelectDataset}
        onAutoLabelResult={handleAutoLabelResult}
        projectName={projectName}
        onProjectNameChange={setProjectName}
        labelMode={labelMode}
        onLabelModeChange={setLabelMode}
        customTags={customTags}
        onAddCustomTag={handleAddCustomTag}
        onRemoveCustomTag={handleRemoveCustomTag}
        labeledFeatures={labeledFeatures}
        onDeleteLabeledFeature={handleDeleteLabel}
        onClearLabeledFeatures={handleClearLabels}
        onMergeLabelsToMapLayers={handleMergeLabelsToMapLayers}
        labelsMergedToMapLayers={labelsMergedToMapLayers}
        olmoCache={olmoCache}
        compareMode={compareMode}
      />
      <main className="flex-1 relative">
        {compareMode ? (
          <SplitMap
            selectedArea={selectedArea}
            imageryLayers={imageryLayers}
            // Seed the split camera from MapView's last reported state
            // so the user's zoom is preserved across the compare toggle.
            initialCamera={mapCameraRef.current}
            onLoadExamplePair={handleLoadExamplePair}
            onRemoveImageryLayer={handleRemoveImageryLayer}
            onExit={handleExitCompare}
          />
        ) : (
          <>
            <MapView
              onAreaSelect={handleAreaSelect}
              onGeometrySelect={handleGeometrySelect}
              onSelectDemoArea={handleDemoAreaSelect}
              queryPixel={queryPixel}
              pickQueryActive={pickQueryActive}
              onQueryPixelPicked={handleQueryPixelPicked}
              onStartPickQuery={handleStartPickQuery}
              onClearQueryPixel={handleClearQueryPixel}
              selectedArea={selectedArea}
              selectedGeometry={selectedGeometry}
              overlayGeojson={overlayGeojson}
              imageryLayers={imageryLayers}
              onRemoveImageryLayer={handleRemoveImageryLayer}
              onAddImageryLayer={handleAddImageryLayer}
              datasets={datasets}
              olmoCache={olmoCache}
              labelMode={labelMode}
              onLabelDrawn={handleLabelDrawn}
              flyToTrigger={flyToTrigger}
              // TIPSv2 polygons need real imagery underneath to be readable
              // (an OSM cartographic basemap shows roads + labels but no
              // landcover signal). Snap to satellite when the user lands on
              // the TIPSv2 tab; they can still flip back via the basemap
              // picker if they want.
              preferredBasemap={viewMode === "tipsv2" ? "satellite" : undefined}
              // Capture camera state into a ref (not useState) so React
              // doesn't re-render every component downstream on every
              // map pan. Read at compare-mode entry to seed SplitMap.
              onCameraChange={(cam) => { mapCameraRef.current = cam; }}
              leadingBasemapControl={
                /* ⇌ Compare — OlmoEarth Studio–style side-by-side toggle.
                   Rendered INSIDE MapView's basemap row (immediately LEFT of
                   OSM) so it shares the same top-right flex container as the
                   basemap pills. Sharing that row keeps it visible without
                   stacking awkwardly into MapLibre's native zoom/compass
                   column. Always visible on every MapView tab regardless of
                   how many imagery layers are loaded; compare entry with
                   zero layers is still useful for OSM / Satellite / Dark
                   basemap A/B. */
                <button
                  type="button"
                  onClick={() => setCompareMode(true)}
                  title={
                    imageryLayers.length === 0
                      ? "Open side-by-side compare. Add imagery layers (STAC, GeoTIFF, or OlmoEarth inference) to compare datasets; otherwise you can still compare basemaps."
                      : imageryLayers.length < 2
                        ? `1 imagery layer loaded — add another for a full A/B compare (still opens with your layer on one side and basemap on the other)`
                        : `Open side-by-side compare with ${imageryLayers.length} imagery layers`
                  }
                  className="px-4 py-2.5 rounded-lg text-[13px] font-semibold shadow-md border border-geo-border bg-gradient-panel text-geo-text hover:border-geo-accent hover:text-geo-accent cursor-pointer transition-all inline-flex items-center gap-1.5"
                  data-testid="compare-mode-toggle"
                >
                  {/* Split the glyph + label into separate spans so
                      flex `gap-1.5` above gives real breathing room between
                      the exchange icon and the word. Earlier "⇌ Compare"
                      in one string rendered with no kerning — the glyph
                      hugged the "C" and looked like a typo. */}
                  <span aria-hidden="true">⇌</span>
                  <span>Compare</span>
                  {imageryLayers.length > 0 && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-geo-accent/15 text-geo-accent font-bold leading-none">
                      {imageryLayers.length}
                    </span>
                  )}
                </button>
              }
            />
            {/* Floating inference legend popovers (one per active layer) —
                anchored top-right inside <main>'s relative box, sitting on
                top of MapView. Shipped in place of the prior Sidebar inline
                legend so the user sees the legend WHERE THE PIXELS ARE. */}
            <InferenceLegendPanel
              imageryLayers={imageryLayers}
              onRemove={handleRemoveImageryLayer}
            />
          </>
        )}
      </main>
    </div>
  );
}
