import { useEffect, useMemo, useState } from "react";
import type { BBox, AnalysisResult, EnvDataResult, DatasetInfo, ViewMode } from "../types";
import type { DataSourceConfig } from "./DataSourcePicker";
import { DataSourcePicker } from "./DataSourcePicker";
import { AutoLabel } from "./AutoLabel";
import { DataUpload } from "./DataUpload";
import { DatasetDetail } from "./DatasetDetail";
import { SampleData } from "./SampleData";
import { SampleRasters } from "./SampleRasters";
import { ProjectMenu } from "./ProjectMenu";
import { Panel, SectionTitle } from "./ui/Panel";
import { Stat } from "./ui/Stat";
import { GemmaChat } from "./GemmaChat";
import { CloudHub } from "./CloudHub";
import { LLMExamples } from "./LLMExamples";
import { DraggablePanel } from "./DraggablePanel";
import { LabelPanel, type LabelFeature, type LabelMode } from "./LabelPanel";
import { RogerLogo } from "./ui/RogerLogo";
import { MapTab, AnalysisTab, OlmoEarthTab, LLMTab, TIPSv2Tab } from "./icons";
import { TIPSv2Panel } from "./TIPSv2Panel";
import { PolygonStats } from "./PolygonStats";
import { StacImagery } from "./StacImagery";
import { OlmoEarthPanel } from "./OlmoEarthPanel";
import { OlmoEarthImport } from "./OlmoEarthImport";
// InferenceLegendPanel is now rendered in App.tsx as a floating overlay on
// top of MapView (closer to the actual pixels) — Sidebar no longer hosts it.
import type { ImageryLayer } from "./MapView";
import { RasterResultsAccordion } from "./MapView";
import { OlmoEarthDemoPairsList } from "./OlmoEarthDemoPairsList";
import type { OlmoEarthRepoStatus } from "../api/client";

interface SidebarProps {
  viewMode: ViewMode;
  onViewChange: (mode: ViewMode) => void;
  selectedArea: BBox | null;
  selectedGeometry?: GeoJSON.Polygon | null;
  onClearSelection?: () => void;
  /** Set the AOI to a small per-model demo bbox + pan the map there.
   * Wired to App.handleDemoAreaSelect; threaded through to OlmoEarthImport
   * so the "Use demo AOI" button in the inference panel works. */
  onSelectDemoArea?: (bbox: BBox) => void;
  /** Embedding similarity query pixel state — App-lifted so the same
   * picked location is shared between the Sidebar OlmoEarthImport
   * (full panel) and the MapView popover OlmoEarthImport (compact). */
  queryPixel?: { lon: number; lat: number } | null;
  pickQueryActive?: boolean;
  onStartPickQuery?: () => void;
  onClearQueryPixel?: () => void;
  imageryLayers?: ImageryLayer[];
  onAddImageryLayer?: (l: ImageryLayer) => void;
  onRemoveImageryLayer?: (id: string) => void;
  analysisResult: AnalysisResult | null;
  envData: EnvDataResult | null;
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  dataSourceConfig: DataSourceConfig;
  onDataSourceChange: (config: DataSourceConfig) => void;
  loading: boolean;
  error: string | null;
  onAnalyze: () => void;
  onDemo?: () => void;
  onUpload: (ds: DatasetInfo) => void;
  onDeleteDataset: (filename: string) => void;
  onSelectDataset: (ds: DatasetInfo) => void;
  onAutoLabelResult: (geojson: GeoJSON.FeatureCollection, meta: Record<string, unknown>) => void;
  // Labeling MVP — wired only on the Map tab. State lives in App.
  projectName: string;
  onProjectNameChange: (v: string) => void;
  labelMode: LabelMode;
  onLabelModeChange: (m: LabelMode) => void;
  customTags: string[];
  onAddCustomTag: (tag: string) => void;
  onRemoveCustomTag: (tag: string) => void;
  labeledFeatures: LabelFeature[];
  onDeleteLabeledFeature: (id: string) => void;
  onClearLabeledFeatures: () => void;
  onMergeLabelsToMapLayers?: () => void;
  labelsMergedToMapLayers?: boolean;
  /** Feeds the live cache-status pill inside the shared OlmoEarthImport
   * form (both Import Data sub-view + Added Layer popover use it). */
  olmoCache?: Record<string, OlmoEarthRepoStatus>;
  /** True when SplitMap (Compare view) is active. SplitMap renders its own
   * OlmoEarth demo-pairs picker overlaid on the map, so the sidebar should
   * hide its duplicate copy to avoid the "same panel in two places" UX
   * smell when the user enters Compare. */
  compareMode?: boolean;
}

const LAND_COVER_COLORS: Record<string, string> = {
  Forest: "#228b22",
  Cropland: "#f0e68c",
  Urban: "#808080",
  Water: "#1e90ff",
  Grassland: "#90ee90",
  Barren: "#d2b48c",
  Wetland: "#5f9ea0",
  "Snow/Ice": "#f0f8ff",
};

export function Sidebar({
  viewMode,
  onViewChange,
  selectedArea,
  selectedGeometry,
  onClearSelection,
  onSelectDemoArea,
  queryPixel,
  pickQueryActive,
  onStartPickQuery,
  onClearQueryPixel,
  imageryLayers,
  onAddImageryLayer,
  onRemoveImageryLayer,
  analysisResult,
  envData,
  datasets,
  selectedDataset,
  dataSourceConfig,
  onDataSourceChange,
  loading,
  error,
  onAnalyze,
  onDemo,
  onUpload,
  onDeleteDataset,
  onSelectDataset,
  onAutoLabelResult,
  projectName,
  onProjectNameChange,
  labelMode,
  onLabelModeChange,
  customTags,
  onAddCustomTag,
  onRemoveCustomTag,
  labeledFeatures,
  onDeleteLabeledFeature,
  onClearLabeledFeatures,
  onMergeLabelsToMapLayers,
  labelsMergedToMapLayers,
  olmoCache,
  compareMode,
}: SidebarProps) {
  const loadedNames = useMemo(() => new Set(datasets.map((d) => d.filename)), [datasets]);
  // Sub-view inside the LLM tab — flipped from the hover submenu. Persists
  // via sessionStorage. Four options:
  //   "local"    = vLLM/Ollama Gemma 4 (GemmaChat)
  //   "cloud"    = CloudHub (NIM / Claude / Gemini — provider picker inside)
  //   "settings" = vLLM/Ollama config panel (cloud provider config — API key
  //                + model picker — lives inline inside each provider pane
  //                within CloudHub, since each needs only those two knobs
  //                and no docker/start lifecycle to manage)
  const [llmSubView, setLlmSubView] = useState<"local" | "cloud" | "examples" | "settings">(() => {
    try {
      const raw = sessionStorage.getItem("geoenv.llm.subView");
      // Backwards-compat: legacy "chat" → "local"; "claude" → "cloud" (now
      // nested under the CloudHub provider picker).
      if (raw === "settings") return "settings";
      if (raw === "cloud" || raw === "claude") return "cloud";
      if (raw === "examples") return "examples";
      return "local";
    } catch {
      return "local";
    }
  });
  const pickLlmSubView = (v: "local" | "cloud" | "examples" | "settings") => {
    setLlmSubView(v);
    try { sessionStorage.setItem("geoenv.llm.subView", v); } catch { /* noop */ }
    if (viewMode !== "gemma") onViewChange("gemma");
  };
  // Pop-out the LLM pane into a draggable floating window (Google Earth
  // Studio pattern). The sidebar's 480 px width is too narrow for long
  // chat + artifact + reasoning blocks, so a detached panel gives the
  // user as much horizontal room as they want. Position + size persist
  // inside DraggablePanel via its own sessionStorage key.
  const [llmFloating, setLlmFloating] = useState<boolean>(() => {
    try { return sessionStorage.getItem("geoenv.llm.floating") === "1"; }
    catch { return false; }
  });
  const toggleLlmFloating = () => {
    setLlmFloating((v) => {
      const next = !v;
      try { sessionStorage.setItem("geoenv.llm.floating", next ? "1" : "0"); } catch { /* noop */ }
      return next;
    });
  };
  // Portal target for the floating chat body. React populates this via a
  // ref callback when the DraggablePanel mounts its inner <div>. Chat
  // components receive this as a prop and portal their messages+composer
  // into it — config + API key + model picker stay docked in the sidebar.
  // null when not floating → chat body renders inline in the sidebar.
  const [floatingChatTarget, setFloatingChatTarget] = useState<HTMLElement | null>(null);
  // Last sub-view that was an actual chat (local / cloud). Used so the
  // floating popout stays on the user's last chat when they flip to
  // Examples or Settings — those are static panels with no chat body, so
  // routing the popout to the chat they were just on keeps the float from
  // emptying out every time they check a reference.
  const [lastChatSubView, setLastChatSubView] = useState<"local" | "cloud">(() => {
    try {
      return sessionStorage.getItem("geoenv.llm.lastChat") === "cloud" ? "cloud" : "local";
    } catch {
      return "local";
    }
  });
  useEffect(() => {
    if (llmSubView === "local" || llmSubView === "cloud") {
      setLastChatSubView(llmSubView);
      try { sessionStorage.setItem("geoenv.llm.lastChat", llmSubView); } catch { /* noop */ }
    }
  }, [llmSubView]);
  // GemmaChat takes a 2-state subView ("chat" | "settings"); when the user
  // is on Local Chat we render its chat view, when on Settings we let it
  // render its full vLLM/Ollama config panel.
  const gemmaSubView: "chat" | "settings" = llmSubView === "settings" ? "settings" : "chat";

  // Sub-view inside the Map tab — same hover-submenu pattern as LLM. Four
  // sub-views, each surfaces one workflow:
  //   "sample"     = SampleData (vector label presets) — DEFAULT. Teaches the
  //                  labeling flow: preset polygons/points land in the
  //                  dataset list as something the user could annotate or
  //                  export as training data.
  //   "rasters"    = SampleRasters (raster imagery presets). Teaches the
  //                  imagery-layer flow: preset GeoTIFFs are fetched + cached
  //                  + added as map layers in one click, appearing alongside
  //                  OlmoEarth inference outputs in the Added Layer popover.
  //   "labeling"   = LabelPanel (manual annotation) + AutoLabel (TIPSv2 /
  //                  SamGeo / spectral) + StacImagery (reference composites
  //                  for the labeler). The old "OlmoEarth Data Layers ·
  //                  Live · mangrove tropical belt" panel was removed from
  //                  this subview 2026-04-21 — the rest of the labeling UI
  //                  stays intact.
  //   "import"     = DataUpload + DatasetDetail (bring-your-own data)
  type MapSubView = "labeling" | "import" | "sample" | "rasters";
  const [mapSubView, setMapSubView] = useState<MapSubView>(() => {
    try {
      const raw = sessionStorage.getItem("geoenv.map.subView");
      if (raw === "labeling") return "labeling";
      if (raw === "import") return "import";
      if (raw === "rasters") return "rasters";
      return "sample";
    } catch {
      return "sample";
    }
  });
  const pickMapSubView = (v: MapSubView) => {
    setMapSubView(v);
    try { sessionStorage.setItem("geoenv.map.subView", v); } catch { /* noop */ }
    if (viewMode !== "map") onViewChange("map");
  };

  // OlmoEarth sub-tab driver. The OlmoEarthPanel owns the actual `sub`
  // state internally (encoder | ft | dataset, hydrated from sessionStorage
  // key `roger.olmo.subtab.v2`). The Sidebar's hover dropdown picks the
  // sub-tab from outside the panel — to avoid lifting state up through App,
  // we write to the SAME sessionStorage key here AND dispatch a custom
  // event the panel listens for. Sidebar tracks its own copy purely for
  // highlighting the active item in the dropdown.
  type OlmoSub = "encoder" | "ft" | "dataset";
  const OLMO_SUB_LABEL: Record<OlmoSub, string> = {
    encoder: "Encoder",
    ft: "Finetune head",
    dataset: "Dataset",
  };
  const [olmoSub, setOlmoSub] = useState<OlmoSub>(() => {
    try {
      const raw = sessionStorage.getItem("roger.olmo.subtab.v2");
      if (raw === "encoder" || raw === "ft" || raw === "dataset") return raw;
    } catch { /* noop */ }
    return "encoder";
  });
  const pickOlmoSub = (v: OlmoSub) => {
    setOlmoSub(v);
    try { sessionStorage.setItem("roger.olmo.subtab.v2", v); } catch { /* noop */ }
    // Live-update OlmoEarthPanel without lifting state to App.
    window.dispatchEvent(new CustomEvent("roger:olmo-subtab-change", { detail: v }));
    if (viewMode !== "olmoearth") onViewChange("olmoearth");
  };

  // Whole-sidebar collapse — slides the aside down to ~16 px (just the
  // toggle chip is visible) so the map gets the full window. Persisted to
  // sessionStorage. Smooth width transition via CSS.
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    try { return sessionStorage.getItem("roger.sidebarCollapsed") === "1"; }
    catch { return false; }
  });
  const toggleSidebar = () => {
    setSidebarCollapsed((v) => {
      const next = !v;
      try { sessionStorage.setItem("roger.sidebarCollapsed", next ? "1" : "0"); }
      catch { /* noop */ }
      return next;
    });
  };

  // Header collapse — hides the big Roger Studio wordmark + 200 px logo
  // when the user wants more vertical room for panels. Persisted per-tab
  // via sessionStorage so refreshes preserve the collapsed state. A thin
  // toggle bar sits between the header and the tab strip and is always
  // visible (whether collapsed or expanded) so the action is always
  // reachable.
  const [headerCollapsed, setHeaderCollapsed] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem("roger.headerCollapsed") === "1";
    } catch {
      return false;
    }
  });
  const toggleHeader = () => {
    setHeaderCollapsed((v) => {
      const next = !v;
      try { sessionStorage.setItem("roger.headerCollapsed", next ? "1" : "0"); } catch { /* noop */ }
      return next;
    });
  };

  return (
    <aside
      className={`relative bg-gradient-sidebar text-geo-text flex flex-col border-r border-geo-border h-full overflow-visible ${
        sidebarCollapsed ? "w-[16px]" : "w-[480px]"
      }`}
    >
      {/* Collapse / expand chip — sits half-over the right border so it
          stays grabbable in both states. ◀ collapses, ▶ expands. */}
      <button
        onClick={toggleSidebar}
        title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        aria-expanded={!sidebarCollapsed}
        className="absolute top-1/2 -translate-y-1/2 left-full z-40 w-7 h-20 rounded-r-md bg-geo-surface border border-geo-border flex items-center justify-center text-geo-muted hover:text-geo-text hover:bg-geo-elevated transition-colors cursor-pointer shadow-sm font-mono text-[12px]"
      >
        {sidebarCollapsed ? "▶" : "◀"}
      </button>
      {/* All sidebar content lives inside this wrapper so a single
          collapsed-state class can hide everything at once. Display
          (not visibility) so the hidden content doesn't take up any
          layout / scroll space. */}
      <div className={`flex-1 flex flex-col min-h-0 overflow-hidden ${sidebarCollapsed ? "hidden" : ""}`}>
      {/* Header — pinned top. Compact variant: every vertical dimension
          halved from the earlier hero lockup — padding pt-10→pt-5 /
          pb-8→pb-4, wordmark 48→24 px, logo 200→100 px, tag margin 4→2.
          Horizontal padding stays at px-10 since the sidebar width is
          fixed. Collapsible via the toggle bar below. */}
      {!headerCollapsed && (
        <div className="relative flex-shrink-0 px-10 pt-5 pb-4 border-b border-geo-border bg-gradient-header">
          {/* Subtle topographic decoration (clipped by this overflow-hidden wrapper) */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div
              className="absolute inset-0 opacity-60"
              style={{
                background:
                  "radial-gradient(ellipse 300px 120px at 90% -20%, rgba(74,123,167,0.15) 0%, transparent 60%), radial-gradient(ellipse 200px 80px at 0% 120%, rgba(201,138,60,0.12) 0%, transparent 60%)",
              }}
            />
          </div>
          <div className="relative" data-testid="roger-studio-brand">
            <div className="flex items-center gap-4">
              {/* Interactive mark — cursor-tracked brackets + click-spin
                  earth ring. Now on the LEFT of the wordmark. */}
              <span className="shrink-0">
                <RogerLogo size={100} />
              </span>
              {/* Title + pixel-art subtitle stack. items-center on the
                  inner flex centers the subtitle UNDER the title (the
                  outer flex still centers the whole column against the logo). */}
              <div className="flex flex-col items-center">
                {/* Style 06 wordmark — Fraunces 900, accent-blue "Studio"
                    (italic removed). */}
                <h1 className="roger-wordmark text-[48px] whitespace-nowrap">
                  Roger <span className="studio">Studio</span>
                </h1>
                {/* Pixel-art subtitle (Silkscreen) with deep-forest → forest
                    → sage gradient, painted via background-clip:text. Centered
                    under the title via items-center on the parent column. */}
                <p className="roger-subtitle">Earth Observation Copilot</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header toggle bar — always visible. Sits directly under the header
          (or replaces it when collapsed). Hover-tinted background so it
          reads as a clickable affordance without competing with the tabs
          below. Mono caret + label keeps the tone technical. */}
      <button
        onClick={toggleHeader}
        className={`flex-shrink-0 flex items-center justify-center gap-2 px-4 ${
          headerCollapsed ? "py-2" : "py-1.5"
        } text-[10px] font-mono uppercase tracking-[0.18em] text-geo-muted bg-geo-surface/60 hover:bg-geo-elevated hover:text-geo-text border-b border-geo-border cursor-pointer transition-colors`}
        title={headerCollapsed ? "Show Roger Studio header" : "Collapse Roger Studio header"}
        aria-label={headerCollapsed ? "Expand header" : "Collapse header"}
        aria-expanded={!headerCollapsed}
      >
        <span className="text-[12px] leading-none">{headerCollapsed ? "▾" : "▴"}</span>
        <span>{headerCollapsed ? "Roger Studio · expand" : "collapse header"}</span>
      </button>

      {/* Project menu + Introduction Doc. Persistent-session container
          borrowed from OE Studio's /api/v1/projects resource shape. Sits
          above the tab strip so it's always reachable regardless of
          which tab is active. Centered horizontally so the two controls
          read as a matched pair, not as a left-anchored header. */}
      <div className="flex-shrink-0 flex items-center justify-center gap-2 px-4 py-1.5 bg-geo-surface border-b border-geo-border">
        <ProjectMenu />
      </div>

      {/* View Tabs — pinned top. Map / OlmoEarth / LLM tabs have hover
          submenus. Layout: each tab is its NATURAL width (no flex-1) so the
          text widths drive the size, and justify-evenly distributes equal
          space between every tab + at both edges. This gives uniform
          label-to-label spacing — the equal-slot version (flex-1) made
          spacing look uneven because variable-width labels centered in
          equal slots produce variable inter-label gaps. */}
      <div className="flex-shrink-0 flex border-b border-geo-border px-4 bg-geo-surface justify-evenly">
        {(["map", "analysis", "olmoearth", "tipsv2", "gemma"] as ViewMode[]).map((mode) => {
          if (mode === "map") {
            return (
              <div key={mode} className="relative group" data-testid="tab-map">
                <button
                  onClick={() => onViewChange(mode)}
                  className={`px-2 py-4 text-[12px] font-medium border-b-2 cursor-pointer transition-all inline-flex items-center justify-center gap-1 whitespace-nowrap ${
                    viewMode === mode
                      ? "text-geo-accent border-geo-accent"
                      : "text-geo-muted border-transparent hover:text-geo-text hover:border-geo-border hover:bg-geo-bg/60"
                  }`}
                >
                  <MapTab state={viewMode === mode ? "active" : "default"} className="w-4 h-4" />
                  Map
                </button>
                {/* Fixed 150 px width on all three dropdown panels so they
                    feel uniform; left-0 anchors to parent's left edge. */}
                <div className="absolute left-0 top-full z-30 hidden group-hover:block pt-0 w-[150px]">
                  <div className="bg-geo-bg border border-geo-border border-t-0 rounded-b-lg shadow-lg overflow-hidden">
                    {(["sample", "rasters", "labeling", "import"] as const).map((sv) => (
                      <button
                        key={sv}
                        onClick={() => pickMapSubView(sv)}
                        className={`w-full text-left px-3 py-2 text-[12px] font-medium cursor-pointer transition-colors ${
                          viewMode === "map" && mapSubView === sv
                            ? "bg-geo-accent/10 text-geo-accent"
                            : "text-geo-text hover:bg-geo-bg/60"
                        }`}
                      >
                        {sv === "sample"
                          ? "Sample Label"
                          : sv === "rasters"
                            ? "Sample Rasters"
                            : sv === "labeling"
                              ? "Labeling"
                              : "Import Data"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          }
          if (mode === "gemma") {
            return (
              <div key={mode} className="relative group" data-testid="tab-llm">
                <button
                  onClick={() => onViewChange(mode)}
                  className={`px-2 py-4 text-[12px] font-medium border-b-2 cursor-pointer transition-all inline-flex items-center justify-center gap-1 whitespace-nowrap ${
                    viewMode === mode
                      ? "text-geo-accent border-geo-accent"
                      : "text-geo-muted border-transparent hover:text-geo-text hover:border-geo-border hover:bg-geo-bg/60"
                  }`}
                >
                  <LLMTab state={viewMode === mode ? "active" : "default"} className="w-4 h-4" />
                  LLM
                </button>
                {/* LLM is the rightmost tab — anchor dropdown to parent's
                    RIGHT edge (right-0) so the 150 px panel expands LEFT
                    and doesn't overflow the sidebar. */}
                <div className="absolute right-0 top-full z-30 hidden group-hover:block pt-0 w-[150px]">
                  <div className="bg-geo-bg border border-geo-border border-t-0 rounded-b-lg shadow-lg overflow-hidden">
                    {(["local", "cloud", "examples", "settings"] as const).map((sv) => (
                      <button
                        key={sv}
                        onClick={() => pickLlmSubView(sv)}
                        className={`w-full text-left px-3 py-2 text-[12px] font-medium cursor-pointer transition-colors ${
                          viewMode === "gemma" && llmSubView === sv
                            ? "bg-geo-accent/10 text-geo-accent"
                            : "text-geo-text hover:bg-geo-bg/60"
                        }`}
                      >
                        {sv === "local"
                          ? "Local"
                          : sv === "cloud"
                            ? "Cloud"
                            : sv === "examples"
                              ? "Examples"
                              : "Settings"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          }
          if (mode === "olmoearth") {
            return (
              <div key={mode} className="relative group" data-testid="tab-olmoearth">
                <button
                  onClick={() => onViewChange(mode)}
                  className={`px-2 py-4 text-[12px] font-medium border-b-2 cursor-pointer transition-all inline-flex items-center justify-center gap-1 whitespace-nowrap ${
                    viewMode === mode
                      ? "text-geo-accent border-geo-accent"
                      : "text-geo-muted border-transparent hover:text-geo-text hover:border-geo-border hover:bg-geo-bg/60"
                  }`}
                >
                  <OlmoEarthTab state={viewMode === mode ? "active" : "default"} className="w-4 h-4" />
                  OlmoEarth
                </button>
                {/* OlmoEarth is mid-right — left-0 anchored is fine since
                    150 px from its left fits within sidebar width. */}
                <div className="absolute left-0 top-full z-30 hidden group-hover:block pt-0 w-[150px]">
                  <div className="bg-geo-bg border border-geo-border border-t-0 rounded-b-lg shadow-lg overflow-hidden">
                    {(["encoder", "ft", "dataset"] as const).map((sv) => (
                      <button
                        key={sv}
                        onClick={() => pickOlmoSub(sv)}
                        className={`w-full text-left px-3 py-2 text-[12px] font-medium cursor-pointer transition-colors ${
                          viewMode === "olmoearth" && olmoSub === sv
                            ? "bg-geo-accent/10 text-geo-accent"
                            : "text-geo-text hover:bg-geo-bg/60"
                        }`}
                      >
                        {OLMO_SUB_LABEL[sv]}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          }
          if (mode === "tipsv2") {
            // TIPSv2 — sits between OlmoEarth and LLM. No submenu (single
            // workflow surface — pick raster, prompts, run). Branded as
            // the model name to mirror OlmoEarth's tab.
            return (
              <button
                key={mode}
                onClick={() => onViewChange(mode)}
                data-testid="tab-tipsv2"
                className={`px-2 py-4 text-[12px] font-medium border-b-2 cursor-pointer transition-all inline-flex items-center justify-center gap-1 whitespace-nowrap ${
                  viewMode === mode
                    ? "text-geo-accent border-geo-accent"
                    : "text-geo-muted border-transparent hover:text-geo-text hover:border-geo-border hover:bg-geo-bg/60"
                }`}
              >
                <TIPSv2Tab state={viewMode === mode ? "active" : "default"} className="w-4 h-4" />
                TIPSv2
              </button>
            );
          }
          // Plain Analysis tab — no submenu yet, awaiting future features.
          // (3D Globe was removed; if more plain-tab modes return, generalise.)
          return (
            <button
              key={mode}
              onClick={() => onViewChange(mode)}
              data-testid={`tab-${mode}`}
              className={`px-2 py-4 text-[12px] font-medium capitalize border-b-2 cursor-pointer transition-all inline-flex items-center justify-center gap-1 whitespace-nowrap ${
                viewMode === mode
                  ? "text-geo-accent border-geo-accent"
                  : "text-geo-muted border-transparent hover:text-geo-text hover:border-geo-border hover:bg-geo-bg/60 active:bg-geo-border/40"
              }`}
            >
              <AnalysisTab state={viewMode === mode ? "active" : "default"} className="w-4 h-4" />
              {mode}
            </button>
          );
        })}
      </div>

      {/* LLM panes stay mounted across tab switches so state (health polls,
          messages, draft input, HF/NIM tokens) isn't lost when the user
          flips to Map / Analysis / 3D and back. We hide each with display:none
          when its sub-view is inactive. Local Chat (GemmaChat) and Cloud Chat
          (CloudChat) hold independent conversations + independent settings —
          switching between them never wipes the other side's history.

          When llmFloating is true, the three panes remount inside a
          DraggablePanel overlay (Google Earth Studio pattern) so the user
          has more horizontal room than the 480 px sidebar. Sidebar slots
          collapse to a "docked elsewhere" placeholder in that case. */}
      {(() => {
        const visible = viewMode === "gemma";
        // Pop-out / dock affordance for the LLM tab. Rendered both in
        // sidebar (when docked, subtle chip above panes) and inside the
        // DraggablePanel's header area (via onClose). Users toggle mode
        // via the pop-out button in the Sidebar header row.
        const popOutChip = visible && !llmFloating && (
          <div className="px-10 pt-3 flex-shrink-0 flex justify-end">
            <button
              type="button"
              onClick={toggleLlmFloating}
              className="text-[10px] font-mono uppercase tracking-wider px-2 py-1 rounded bg-geo-bg hover:bg-geo-accent hover:text-white border border-geo-border hover:border-geo-accent transition-colors cursor-pointer"
              title="Pop the LLM panel out as a draggable floating window"
              data-testid="llm-popout-toggle"
            >
              ⤢ pop out
            </button>
          </div>
        );
        // LLMExamples is a static reference list — no chat body to portal.
        // When floating, the Examples subview keeps rendering in the sidebar
        // (config+setup details stay there) and the floating panel just
        // shows a note that Examples aren't detachable.
        //
        // GemmaChat + CloudHub each accept chatBodyPortalTarget. When
        // llmFloating is on AND the provider's subview is selected, the chat
        // body (messages + composer) portals into floatingChatTarget. The
        // config header (API key, model picker, status) stays inline in the
        // sidebar so users can still tweak keys/models without toggling
        // the float state.
        //
        // IMPORTANT: the portal routing is based on llmSubView alone, NOT
        // on `visible` — so when the user clicks Analysis / Map / 3D / Olmo
        // the floating chat window stays alive and interactive. The sidebar
        // LLM config slot is simply hidden while the user is on another tab
        // (toggled via display:none on the outer container below).
        //
        // Sidebar display gating is driven by llmSubView (which pane shows
        // in the 480 px aside), but the FLOATING popout follows a different
        // routing: when the user is on Examples or Settings — both static
        // panels with no chat body to portal — the popout keeps showing
        // whichever chat they were last on. Without this, flipping to
        // Settings would empty the popout window.
        const popoutChatSubView: "local" | "cloud" =
          llmSubView === "local" ? "local" :
          llmSubView === "cloud" ? "cloud" :
          lastChatSubView;
        // Sidebar display: Settings form lives inside GemmaChat (subView=
        // "settings"), so the GemmaChat sidebar slot is visible for both
        // "local" and "settings". CloudHub is visible only for "cloud".
        const sidebarShowLocal = visible && (llmSubView === "local" || llmSubView === "settings");
        const sidebarShowCloud = visible && llmSubView === "cloud";
        const sidebarShowExamples = visible && llmSubView === "examples";
        const paneBlock = (
          <>
            <div
              className="flex-1 min-h-0 flex flex-col"
              style={{ display: sidebarShowLocal ? "flex" : "none" }}
            >
              <GemmaChat
                selectedArea={selectedArea}
                datasets={datasets}
                autoLabelSummary={null}
                subView={gemmaSubView}
                onSubViewChange={(v) => pickLlmSubView(v === "settings" ? "settings" : "local")}
                chatBodyPortalTarget={llmFloating && popoutChatSubView === "local" ? floatingChatTarget : null}
              />
            </div>
            <div
              className="flex-1 min-h-0 flex flex-col"
              style={{ display: sidebarShowCloud ? "flex" : "none" }}
            >
              <CloudHub
                selectedArea={selectedArea}
                datasets={datasets}
                autoLabelSummary={null}
                chatBodyPortalTarget={llmFloating && popoutChatSubView === "cloud" ? floatingChatTarget : null}
              />
            </div>
            <div
              className="flex-1 min-h-0 flex flex-col"
              style={{ display: sidebarShowExamples ? "flex" : "none" }}
            >
              <LLMExamples />
            </div>
          </>
        );

        return (
          <>
            {popOutChip}
            {/* paneBlock is ALWAYS mounted so chat components stay alive
                across tab switches — that's what lets the floating chat
                panel keep working when the user flips to Analysis / Map /
                3D / Olmo. The outer wrapper just toggles display:none to
                hide the sidebar slot when the LLM tab is inactive. */}
            <div
              className="px-10 py-6 flex-1 min-h-0 flex flex-col"
              style={{ display: visible ? "flex" : "none" }}
              data-testid="llm-sidebar-slot"
            >
              {paneBlock}
            </div>
            {/* Floating overlay — renders via portal to document.body so it
                escapes the sidebar's stacking context and survives:
                  1. Sidebar collapse (aside width → 16 px)
                  2. Tab switches (viewMode !== "gemma")
                The inner ref'd div is the portal target the active
                provider's chat body renders into. Close via × in the panel
                header (onClose → toggleLlmFloating docks the chat back). */}
            {llmFloating && (
              <DraggablePanel
                title={`LLM · ${llmSubView}`}
                onClose={toggleLlmFloating}
                storageKey="geoenv.llm.floatingGeom"
                defaultX={200}
                defaultY={120}
                defaultWidth={560}
                defaultHeight={720}
                minWidth={360}
                minHeight={440}
              >
                <div
                  ref={setFloatingChatTarget}
                  className="h-full min-h-0 flex flex-col"
                  data-testid="llm-floating-chat-target"
                />
              </DraggablePanel>
            )}
          </>
        );
      })()}

      {/* Non-LLM tabs */}
      {viewMode !== "gemma" && (
      <div className="px-10 py-10 flex-1 min-h-0 overflow-y-auto space-y-10">
        {/* Selected Area — shown on all tabs */}
        {selectedArea ? (
          <>
            <Panel>
              <SectionTitle>Selected Area</SectionTitle>
              <div className="font-mono text-sm leading-loose">
                <div>W: {selectedArea.west.toFixed(4)}</div>
                <div>S: {selectedArea.south.toFixed(4)}</div>
                <div>E: {selectedArea.east.toFixed(4)}</div>
                <div>N: {selectedArea.north.toFixed(4)}</div>
              </div>
            </Panel>
            <PolygonStats
              selectedArea={selectedArea}
              selectedGeometry={selectedGeometry ?? null}
              onClear={onClearSelection}
            />
          </>
        ) : (
          <Panel className="text-center">
            <p className="mb-5 text-sm text-geo-muted leading-relaxed">
              {viewMode === "map"
                ? "Draw an area on the map to begin"
                : viewMode === "analysis"
                ? "No area selected yet — go to Map tab to draw one"
                : "No area selected yet — go to Map tab to draw one"}
            </p>
            {onDemo && (
              <button
                onClick={onDemo}
                className="px-5 py-2.5 bg-geo-bg text-geo-text border border-geo-border rounded-lg text-sm font-medium cursor-pointer hover:border-geo-accent hover:text-geo-accent transition-colors"
              >
                Try Demo (Kenyan Coast)
              </button>
            )}
          </Panel>
        )}

        {/* Error — always shown if present */}
        {error && (
          <div className="bg-red-50 text-red-700 border border-red-200 rounded-lg p-3 text-[13px]">
            {error}
          </div>
        )}

        {/* ============ MAP TAB — split into Labeling | Import Data | Sample Data ============ */}
        {viewMode === "map" && mapSubView === "labeling" && (
          <>
            {/* Imagery (STAC composites) — reference layer for the labeler */}
            {selectedArea && onAddImageryLayer && onRemoveImageryLayer && (
              <StacImagery
                selectedArea={selectedArea}
                imageryLayers={imageryLayers ?? []}
                onAdd={onAddImageryLayer}
                onRemove={onRemoveImageryLayer}
              />
            )}

            {/* NOTE: the "OlmoEarth Data Layers · Live · olmoearth_projects_
                mangrove · Mangrove (tropical belt)" coverage-toggle panel was
                REMOVED 2026-04-21 per product direction. The rest of the
                Labeling subview (manual annotation + auto-label runner + STAC
                imagery reference) stays because it's the productive labeling
                surface. If we ever re-introduce dataset coverage overlays,
                put them in the Added Layer popover over MapView instead of
                a sidebar section — that matches the new layer-management
                convention. */}

            {/* Build Labels — manual annotation MVP */}
            <LabelPanel
              projectName={projectName}
              onProjectNameChange={onProjectNameChange}
              labelMode={labelMode}
              onLabelModeChange={onLabelModeChange}
              customTags={customTags}
              onAddCustomTag={onAddCustomTag}
              onRemoveCustomTag={onRemoveCustomTag}
              features={labeledFeatures}
              onDeleteFeature={onDeleteLabeledFeature}
              onClearAll={onClearLabeledFeatures}
              onMergeToMapLayers={onMergeLabelsToMapLayers}
              mergedToMapLayers={labelsMergedToMapLayers}
            />

            {/* Auto-Label — TIPSv2/SamGeo/spectral on uploaded datasets.
                Lives under Labeling since it produces labels too. Only shown
                when there's a dataset to run it on. */}
            {datasets.length > 0 && (
              <div>
                <AutoLabel datasets={datasets} onResult={onAutoLabelResult} />
              </div>
            )}
          </>
        )}

        {/* ============ MAP → IMPORT DATA ============ */}
        {viewMode === "map" && mapSubView === "import" && (
          <>
            <div>
              <DataUpload
                datasets={datasets}
                onUpload={onUpload}
                onDelete={onDeleteDataset}
                onSelect={onSelectDataset}
              />
            </div>
            {selectedDataset && (
              <div>
                <DatasetDetail
                  dataset={selectedDataset}
                  onClose={() => onSelectDataset(selectedDataset)}
                  onAddImageryLayer={onAddImageryLayer}
                />
              </div>
            )}
            {/* Import OlmoEarth / OlmoEarth-FT data — shared form with the
                Added Layer popover. Pick an FT head or base encoder, run
                inference on the current selection, and the result lands
                as a map layer. Sidebar version gets the wrapped panel
                chrome (not compact). */}
            <OlmoEarthImport
              olmoCache={olmoCache}
              selectedArea={selectedArea}
              onAddImageryLayer={onAddImageryLayer}
              onSelectArea={onSelectDemoArea}
              queryPixel={queryPixel}
              pickQueryActive={pickQueryActive}
              onStartPickQuery={onStartPickQuery}
              onClearQueryPixel={onClearQueryPixel}
            />
          </>
        )}

        {/* ============ MAP → SAMPLE LABEL (vector presets) ============ */}
        {viewMode === "map" && mapSubView === "sample" && (
          <div>
            <SampleData
              onLoad={onUpload}
              onDelete={onDeleteDataset}
              loadedNames={loadedNames}
            />
          </div>
        )}

        {/* ============ MAP → SAMPLE RASTERS (imagery presets) ============ */}
        {viewMode === "map" && mapSubView === "rasters" && onAddImageryLayer && onRemoveImageryLayer && (
          <div>
            <SampleRasters
              onLoad={onUpload}
              onDelete={onDeleteDataset}
              onAddImageryLayer={onAddImageryLayer}
              onRemoveImageryLayer={onRemoveImageryLayer}
              imageryLayers={imageryLayers ?? []}
              datasets={datasets}
            />
          </div>
        )}

        {/* ============ ANALYSIS TAB — computed results ============ */}
        {viewMode === "analysis" && (
          <>
            {/* Data source config — drives /analyze, not imagery rendering */}
            <DataSourcePicker config={dataSourceConfig} onChange={onDataSourceChange} />

            {/* Pre-flight size warning. /analyze reads the WorldCover COG
                windowed to the bbox over HTTPS — wall-clock scales roughly
                linearly with bbox area. Empirically anything beyond
                ~50,000 km² (~Lake Erie scale) starts pushing the 90 s
                frontend timeout; the entire Great Lakes region (~245k km²)
                routinely hits it. Surface a warning here so the user
                knows to expect a slow run or to redraw a tighter bbox. */}
            {selectedArea && (() => {
              const lonSpan = Math.abs(selectedArea.east - selectedArea.west);
              const latSpan = Math.abs(selectedArea.north - selectedArea.south);
              const midLatRad = ((selectedArea.north + selectedArea.south) / 2) * (Math.PI / 180);
              // 1° lat ≈ 111 km; 1° lon ≈ 111 km × cos(lat). Good enough
              // for an order-of-magnitude check; the underlying Open-Meteo
              // grid + WorldCover COG read time is what we're proxying.
              const areaKm2 = lonSpan * latSpan * 111 * 111 * Math.cos(midLatRad);
              if (areaKm2 < 50_000) return null;
              const slow = areaKm2 >= 200_000;
              return (
                <div
                  className={`flex-shrink-0 mb-3 px-3 py-2 rounded-lg border text-[11px] leading-snug ${
                    slow
                      ? "bg-red-50 border-red-200 text-red-900"
                      : "bg-amber-50 border-amber-200 text-amber-900"
                  }`}
                  data-testid="analyze-size-warning"
                >
                  <div className="font-semibold mb-0.5">
                    {slow ? "Bbox is very large — likely to time out" : "Bbox is large — Analyze may run slow"}
                  </div>
                  <div>
                    Selected area is{" "}
                    <span className="font-mono">~{Math.round(areaKm2).toLocaleString()} km²</span>.
                    {" /analyze"} reads the WorldCover COG over HTTPS for the full bbox; wall-clock scales with area.
                    {slow
                      ? " The 90 s request budget is likely to expire before completion."
                      : " Consider a tighter bbox if it doesn't return within ~60 s."}
                  </div>
                </div>
              );
            })()}

            {/* Analyze button */}
            <button
              onClick={onAnalyze}
              disabled={!selectedArea || loading}
              className={`w-full py-3.5 rounded-xl text-sm font-semibold tracking-wide transition-all ${
                !selectedArea || loading
                  ? "bg-geo-elevated text-geo-muted cursor-not-allowed"
                  : "bg-gradient-primary text-white cursor-pointer hover:shadow-lg hover:-translate-y-0.5 shadow-md"
              }`}
            >
              {loading ? "Analyzing..." : analysisResult ? "Re-analyze Area" : "Analyze Area"}
            </button>

            {/* Empty state */}
            {!analysisResult && !loading && selectedArea && (
              <Panel className="text-center">
                <p className="text-sm text-geo-muted">
                  Click <span className="font-semibold text-geo-text">Analyze Area</span> to compute land cover, suitability scores, and current weather for this region based on the OlmoEarth base model.
                </p>
              </Panel>
            )}

            {/* Raster results — one pill per OlmoEarth inference layer
                currently on the map. Click a pill to see the LLM agent's
                plain-language explanation of what the colors represent,
                plus scene metadata + full class list. The MAP itself only
                carries color swatches / gradient bars; the text details
                live here in Analysis. Hidden when no inference layers
                are active. */}
            {/* Sample Data — one-click OlmoEarth demo pairs. Surfaces
                the curated compare presets (Mangrove Niger Delta, AWF
                Tsavo, Ecosystem California) as load-to-map buttons with
                warm/cold pills + elapsed-seconds feedback. Mirrors the
                SplitMap preset chrome but drops both sides onto the
                regular map so users can inspect a single-scene output
                without entering compare mode. Hidden while Compare mode
                is active — SplitMap renders its own copy of this picker
                overlaid on the map, so showing both at once was visually
                redundant (same 3 cards, same click action). */}
            {!compareMode && (
              <div>
                <div className="mb-3">
                  <SectionTitle>Sample Data — OlmoEarth demo pairs</SectionTitle>
                </div>
                <OlmoEarthDemoPairsList
                  onAddImageryLayer={onAddImageryLayer}
                  onRemoveImageryLayer={onRemoveImageryLayer}
                  imageryLayers={imageryLayers}
                />
              </div>
            )}

            {imageryLayers && imageryLayers.length > 0 && (
              <div>
                <div className="mb-3">
                  <SectionTitle>Raster Results on Map</SectionTitle>
                </div>
                <RasterResultsAccordion
                  imageryLayers={imageryLayers}
                  onRemoveImageryLayer={onRemoveImageryLayer}
                />
              </div>
            )}

            {/* Analysis Results — mark as demo since backend uses lat heuristics */}
            {analysisResult && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <SectionTitle>Land Cover Classification</SectionTitle>
                  <span
                    className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-warn-soft text-geo-warn"
                    title="This analysis uses latitude-based heuristics, not real satellite data. Upload a GeoTIFF and run Auto-Label for real classification."
                  >
                    Demo
                  </span>
                </div>
                <div className="space-y-4">
                  {analysisResult.land_cover.map((lc) => (
                    <div key={lc.id} className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-sm shrink-0"
                        style={{ background: LAND_COVER_COLORS[lc.name] || lc.color }}
                      />
                      <span className="flex-1 text-sm">{lc.name}</span>
                      <span className="font-mono text-sm text-geo-muted">
                        {lc.percentage.toFixed(1)}%
                      </span>
                      <div className="w-[80px] h-2 bg-geo-elevated rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${lc.percentage}%`,
                            background: LAND_COVER_COLORS[lc.name] || lc.color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-5 pt-4 border-t border-geo-border font-mono text-xs text-geo-muted">
                  Total area: {analysisResult.area_km2.toFixed(1)} km²
                </div>
                <p className="mt-3 text-xs text-geo-muted leading-relaxed">
                  Upload a GeoTIFF and run <span className="font-semibold text-geo-text">Auto-Label</span> on the Map tab for real satellite-based classification.
                </p>
              </div>
            )}

            {/* Suitability Scores */}
            {/* Suitability Scores removed: they were latitude-based heuristics,
                not real remote-sensing signals, so they misled more than they
                helped. Land Cover above is still flagged DEMO for the same
                reason; real land-cover comes from running Auto-Label on
                uploaded imagery. */}

            {/* OlmoEarth Coverage block moved to its own dedicated tab —
                the Analysis tab stays focused on area-scoped computed
                results (land cover heuristic, weather, elevation). */}

            {/* Environmental Data */}
            {envData && (envData.temperature != null || envData.wind || envData.solar_irradiance != null) && (
              <div>
                <SectionTitle>Current Weather</SectionTitle>
                <div className="grid grid-cols-2 gap-4">
                  {envData.temperature != null && (
                    <Panel border={false} className="text-center">
                      <Stat
                        label="Temp"
                        value={envData.temperature.toFixed(1)}
                        unit="°C"
                        color={
                          envData.temperature > 25
                            ? "#f59e0b"
                            : envData.temperature < 5
                            ? "#3b82f6"
                            : undefined
                        }
                        size="lg"
                      />
                    </Panel>
                  )}
                  {envData.wind && (
                    <Panel border={false} className="text-center">
                      <Stat
                        label={`Wind km/h (${envData.wind.direction.toFixed(0)}deg)`}
                        value={envData.wind.speed.toFixed(1)}
                        color={envData.wind.speed > 20 ? "#ef4444" : undefined}
                        size="lg"
                      />
                    </Panel>
                  )}
                  {envData.solar_irradiance != null && (
                    <Panel border={false} className="text-center">
                      <Stat
                        label="Solar W/m2"
                        value={envData.solar_irradiance.toFixed(0)}
                        color="#f59e0b"
                        size="lg"
                      />
                    </Panel>
                  )}
                  {envData.humidity != null && (
                    <Panel border={false} className="text-center">
                      <Stat
                        label="Humidity"
                        value={`${envData.humidity.toFixed(0)}%`}
                        color="#3b82f6"
                        size="lg"
                      />
                    </Panel>
                  )}
                </div>
              </div>
            )}

            {/* Auto-Label results summary on results tabs */}
            {datasets.some((d) => d.filename.startsWith("auto-label-")) && (
              <div>
                <SectionTitle>Labeled Overlays</SectionTitle>
                <div className="space-y-2">
                  {datasets
                    .filter((d) => d.filename.startsWith("auto-label-"))
                    .map((d) => (
                      <div
                        key={d.filename}
                        className="flex items-center justify-between text-sm p-3 bg-geo-surface border border-geo-border rounded-lg"
                      >
                        <span className="text-geo-text font-mono truncate">
                          {d.filename.replace("auto-label-", "").replace(".geojson", "")}
                        </span>
                        <span className="text-xs text-geo-muted font-mono">
                          {d.vector?.feature_count ?? 0} features
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ============ OLMOEARTH TAB — live HF catalog ============ */}
        {viewMode === "olmoearth" && (
          <OlmoEarthPanel
            selectedArea={selectedArea}
            onAddImageryLayer={onAddImageryLayer}
          />
        )}

        {/* ============ TIPSv2 TAB — zero-shot semantic labeling ============
            Standalone from OlmoEarth (the encoder is locked to 12-band S2 so
            fusion isn't feasible on uploaded high-res rasters). Researchers
            overlay this tab's polygon output with the OlmoEarth raster
            output on the map and ask the LLM tab to interpret each one. */}
        {viewMode === "tipsv2" && (
          <TIPSv2Panel datasets={datasets} onResult={onAutoLabelResult} />
        )}
      </div>
      )}

      {/* Footer — pinned bottom (shared across all tabs) */}
      <div className="flex-shrink-0 px-10 py-6 border-t border-geo-border bg-geo-surface text-[11px] text-geo-dim tracking-widest">
        OlmoEarth &middot; TIPSv2 &middot; MapLibre
      </div>
      </div>
    </aside>
  );
}
