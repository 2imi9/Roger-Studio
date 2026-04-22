import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { TerraDraw, TerraDrawPolygonMode, TerraDrawPointMode, TerraDrawLineStringMode } from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import type { BBox, BasemapStyle, DatasetInfo } from "../types";
import type { OlmoEarthRepoStatus } from "../api/client";
// SAMPLES / loadSampleDataset imports dropped 2026-04-21 with the
// Samples tab removal — preset geojson datasets are label references,
// not raster layers, and now live only under Map → Sample Label.
import { DATASET_COVERAGE } from "../constants/olmoEarthCoverage";
import { OlmoEarthImport } from "./OlmoEarthImport";
import { FitToBounds } from "./icons";
// explainRaster import removed — RasterExplanation no longer auto-fires
// the LLM explainer. Endpoint still exists (MCP tool surface) for
// future re-enable.

// Curated FT-head allowlist for the Added Layer "OlmoEarth" tab. We were
// previously listing EVERY cached OlmoEarth repo including base encoders
// (Nano/Tiny/Base/Large) and ad-hoc project datasets, which turned the tab
// into a noisy dump. Per product direction the OlmoEarth tab shows only
// the five FT heads that have a registered colormap + user-facing task
// (see ``backend/app/services/olmoearth_inference._COLORMAPS``). If a new
// FT head ships, add it here AND to _COLORMAPS / _COLORMAP_LEGEND so the
// frontend picks up the gradient hint automatically.
// ``supported`` mirrors the flag inside OlmoEarthImport.tsx's FT_HEADS
// registry — keep the two lists in sync when a loader update lands.
// Shown as a "beta" pill on unsupported rows so users know loading cache
// for these is allowed but running inference will stub until the loader
// catches up.
const OLMOEARTH_FT_HEADS: { repoId: string; task: string; colorKey: string; supported: boolean }[] = [
  { repoId: "allenai/OlmoEarth-v1-FT-LFMC-Base", task: "Live fuel moisture", colorKey: "flammability", supported: true },
  { repoId: "allenai/OlmoEarth-v1-FT-Mangrove-Base", task: "Mangrove extent", colorKey: "mangrove", supported: true },
  { repoId: "allenai/OlmoEarth-v1-FT-AWF-Base", task: "Southern-Kenya land use", colorKey: "landuse", supported: true },
  { repoId: "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base", task: "Forest loss driver", colorKey: "forestloss", supported: false },
  { repoId: "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base", task: "Ecosystem type", colorKey: "ecosystem", supported: true },
];

// Public draw modes the LabelPanel can request — corresponds to terra-draw's
// own mode names (point / linestring / polygon). Keeping the prop type here
// rather than importing from LabelPanel avoids a cyclic dep.
export type LabelDrawType = "point" | "polygon" | "line";

export interface LabelModeProp {
  active: boolean;
  type: LabelDrawType;
  tag: string;
}

const BASEMAPS: Record<BasemapStyle, { url: string; label: string }> = {
  osm: { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", label: "OSM" },
  satellite: { url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", label: "Satellite" },
  dark: { url: "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png", label: "Dark" },
};

export interface ImageryLayer {
  id: string;
  /** XYZ tile template for RASTER layers (STAC composite, dataset tile
   * server, OlmoEarth inference). Optional so the same list can also
   * carry vector layers — see ``featureCollection`` below. Exactly one
   * of ``tileUrl`` / ``featureCollection`` should be set per layer. */
  tileUrl?: string;
  /** VECTOR payload for label sets a user pushed to the map via
   * "Add to map layers" in LabelPanel. Rendered as a geojson source
   * with fill / line / circle layers keyed off the feature geometry
   * type. Lets the Added Layer panel treat both rasters and vector
   * labels uniformly (one list, one toggle/remove flow). */
  featureCollection?: GeoJSON.FeatureCollection;
  label?: string;
  opacity?: number;
  // Populated when the layer came from /api/olmoearth/infer. The legend
  // panel keys off this to show per-class swatches / units / scene info.
  inferenceMetadata?: import("../api/client").OlmoEarthInferenceResult;
}

interface MapViewProps {
  onAreaSelect: (bbox: BBox) => void;
  onGeometrySelect?: (polygon: GeoJSON.Polygon) => void;
  selectedArea: BBox | null;
  selectedGeometry?: GeoJSON.Polygon | null;
  overlayGeojson?: (GeoJSON.Feature | GeoJSON.FeatureCollection)[];
  imageryLayers?: ImageryLayer[];
  /** Remove an imagery layer by id. Wired to the Added Layer panel's
   * per-row × button so users can clear inference outputs / imported
   * rasters from the map without leaving the map view. App owns the
   * actual state; MapView just forwards the id. */
  onRemoveImageryLayer?: (id: string) => void;
  /** Add an existing imagery layer back onto the map (used from the
   * Added Layer "Imported" tab's "+ add to map" button). */
  onAddImageryLayer?: (layer: ImageryLayer) => void;
  /** User uploads (incl. auto-label outputs) so the Added Layer "Imported"
   * tab can list them with an "add to map" affordance. */
  datasets?: DatasetInfo[];
  /** HuggingFace repo cache snapshot — drives the Added Layer "OlmoEarth"
   * tab's list of cached datasets + the "Import" tab's status display. */
  olmoCache?: Record<string, OlmoEarthRepoStatus>;
  /** Emitted on every ``moveend`` so callers can observe the current
   * camera state. Used by App to carry MapView's zoom+center into
   * SplitMap when the user enters compare mode — otherwise the user
   * zooms in on MapView, clicks Compare, and the split view resets to
   * ``zoom=8`` at the bbox centroid (the audit caught this as issue #21).
   * Coalesced to moveend (not every animation frame) so React isn't
   * re-rendering on every mouse drag. */
  onCameraChange?: (cam: { center: [number, number]; zoom: number }) => void;
  /** Extra controls to render inline LEFT of the basemap (OSM/Satellite/
   * Dark) switcher — typically the ⇌ Compare toggle. App owns the click
   * handler + mode state; MapView just positions the node so it sits in
   * the same row as the basemap picker instead of stacking awkwardly
   * against MapLibre's native zoom/compass column. */
  leadingBasemapControl?: React.ReactNode;
  // Labeling MVP — when labelMode.active is true, terra-draw stays in the
  // requested mode (point/polygon/line) and each finished draw is forwarded
  // to onLabelDrawn instead of onGeometrySelect. The selection-singleton
  // behavior (single-bbox feeding Analysis/OlmoEarth) is preserved when
  // labelMode is inactive.
  labelMode?: LabelModeProp;
  onLabelDrawn?: (
    geometry: GeoJSON.Point | GeoJSON.Polygon | GeoJSON.LineString,
    type: LabelDrawType,
  ) => void;
  // App bumps `nonce` whenever a new dataset (sample / upload) lands and the
  // map should pan + zoom to its bbox. We can't react to selectedArea alone
  // because user-draw also sets it (and the rect they drew is already in
  // view — re-fitting would feel like the map fights them). The nonce
  // disambiguates: only data-load paths increment it.
  flyToTrigger?: { bbox: BBox; nonce: number } | null;
}

type DrawMode = "none" | "rectangle" | "polygon";

export function MapView({
  onAreaSelect,
  onGeometrySelect,
  selectedArea,
  selectedGeometry,
  overlayGeojson,
  imageryLayers,
  onRemoveImageryLayer,
  onAddImageryLayer,
  datasets,
  olmoCache,
  labelMode,
  onLabelDrawn,
  flyToTrigger,
  leadingBasemapControl,
  onCameraChange,
}: MapViewProps) {
  // Keep the latest ``onCameraChange`` callback in a ref so the moveend
  // listener (attached once at mount) calls the current closure instead
  // of a stale reference. Classic React-hook-in-imperative-binding
  // pattern.
  const onCameraChangeRef = useRef(onCameraChange);
  useEffect(() => { onCameraChangeRef.current = onCameraChange; }, [onCameraChange]);
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const terraDrawRef = useRef<TerraDraw | null>(null);
  const [drawMode, setDrawMode] = useState<DrawMode>("none");
  const [basemap, setBasemap] = useState<BasemapStyle>("osm");
  // Per-layer visibility toggles driven by the Added Layer panel. We hide
  // by id-set rather than mutating the upstream imageryLayers prop so App
  // retains authoritative state — toggling a layer off on the map shouldn't
  // drop it from the sidebar's Added Layer list. Removal (× button) DOES
  // call onRemoveImageryLayer so it drops from App too. Empty by default:
  // newly-added layers are visible until the user toggles them.
  const [hiddenLayerIds, setHiddenLayerIds] = useState<Set<string>>(new Set());
  const toggleLayerHidden = (id: string) => {
    setHiddenLayerIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const drawStart = useRef<{ lng: number; lat: number } | null>(null);
  const drawing = drawMode === "rectangle";  // legacy name used by the rect handler below

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "&copy; OpenStreetMap contributors",
          },
        },
        layers: [
          {
            id: "osm-tiles",
            type: "raster",
            source: "osm",
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [-98.5, 39.8],
      zoom: 4,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    mapRef.current = map;

    // Report camera state to parent on move-end. Coalesced to
    // ``moveend`` (not ``move``) so React state updates happen once
    // per pan/zoom gesture, not every frame. Used by App to carry the
    // zoom level into SplitMap when the user enters compare mode —
    // otherwise clicking Compare would reset the view to ``zoom=8``
    // at the bbox centroid and the user would lose their detail view.
    map.on("moveend", () => {
      const cb = onCameraChangeRef.current;
      if (!cb) return;
      const c = map.getCenter();
      cb({ center: [c.lng, c.lat], zoom: map.getZoom() });
    });

    // Flip the "style parsed" flag on first load. Any reconcile call that
    // landed before load completes gets flushed now.
    // Terra-draw polygon mode — initialized once, started only when the user
    // picks the polygon tool so it doesn't intercept rectangle drags.
    map.once("load", () => {
      const draw = new TerraDraw({
        adapter: new TerraDrawMapLibreGLAdapter({ map, coordinatePrecision: 9 }),
        modes: [
          new TerraDrawPolygonMode({
            // Bright amber stroke + subtle fill so vertices and the closing
            // edge read clearly on any basemap (default styles are near-invisible
            // on OSM / satellite tiles).
            styles: {
              fillColor: "#f59e0b",
              fillOpacity: 0.18,
              outlineColor: "#f59e0b",
              outlineWidth: 3,
              closingPointColor: "#dc2626",
              closingPointOutlineColor: "#ffffff",
              closingPointOutlineWidth: 2,
              closingPointWidth: 6,
              snappingPointColor: "#2563eb",
              snappingPointOutlineColor: "#ffffff",
              snappingPointOutlineWidth: 2,
              snappingPointWidth: 5,
              coordinatePointColor: "#f59e0b",
              coordinatePointOutlineColor: "#ffffff",
              coordinatePointOutlineWidth: 2,
              coordinatePointWidth: 5,
            },
          }),
          // Point + LineString modes added for the labeling MVP. Same amber
          // palette as polygon for visual continuity while drawing; final
          // labeled features get their per-tag color via overlayGeojson.
          new TerraDrawPointMode({
            styles: { pointColor: "#f59e0b", pointOutlineColor: "#ffffff", pointOutlineWidth: 2, pointWidth: 6 },
          }),
          new TerraDrawLineStringMode({
            styles: { lineStringColor: "#f59e0b", lineStringWidth: 3 },
          }),
        ],
      });
      draw.on("finish", (id, ctx) => {
        if (ctx.action !== "draw") return;
        const snap = draw.getSnapshot();
        const feat = snap.find((f) => f.id === id);
        if (!feat) return;
        const label = labelModeRef.current;

        // Label-mode path: append to labeled set, keep drawing additively.
        // We DON'T clear/static the draw — re-set the same mode so the user
        // can keep clicking out features without reopening the panel.
        if (label?.active && onLabelDrawnRef.current) {
          const tdMode = label.type === "line" ? "linestring" : label.type;
          if (
            (label.type === "polygon" && feat.geometry.type !== "Polygon") ||
            (label.type === "point" && feat.geometry.type !== "Point") ||
            (label.type === "line" && feat.geometry.type !== "LineString")
          ) {
            return; // type mismatch — ignore
          }
          const geom = feat.geometry as GeoJSON.Point | GeoJSON.Polygon | GeoJSON.LineString;
          // Wipe just the in-progress draw layer so the next click starts
          // clean, then re-arm the same mode. The persisted labeled feature
          // is rendered separately via overlayGeojson, so this clear doesn't
          // remove it from the map.
          draw.clear();
          requestAnimationFrame(() =>
            requestAnimationFrame(() => {
              onLabelDrawnRef.current?.(geom, label.type);
              if (draw.enabled) draw.setMode(tdMode);
            }),
          );
          return;
        }

        // Selection path (legacy): polygon-only, single-shot, returns to static.
        if (feat.geometry.type !== "Polygon") return;
        const geom = feat.geometry;
        draw.clear();
        draw.setMode("static");
        setDrawMode("none");
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            onGeometrySelectRef.current?.(geom);
          }),
        );
      });
      terraDrawRef.current = draw;
    });

    return () => {
      terraDrawRef.current?.stop();
      terraDrawRef.current = null;
      map.remove();
      if (mapRef.current === map) mapRef.current = null;
    };
  }, []);

  // Keep a ref to the latest onGeometrySelect so the once-attached terra-draw
  // handler always calls the current prop (no re-init needed on prop change).
  const onGeometrySelectRef = useRef(onGeometrySelect);
  useEffect(() => {
    onGeometrySelectRef.current = onGeometrySelect;
  }, [onGeometrySelect]);
  // Same pattern for the labeling callbacks — once-attached finish handler
  // reads from refs so prop changes don't require a draw re-init.
  const onLabelDrawnRef = useRef(onLabelDrawn);
  useEffect(() => { onLabelDrawnRef.current = onLabelDrawn; }, [onLabelDrawn]);
  const labelModeRef = useRef(labelMode);
  useEffect(() => { labelModeRef.current = labelMode; }, [labelMode]);

  // Sync labelMode → terra-draw mode. Active label mode owns the draw and
  // overrides the legacy drawMode state. When labelMode flips off, this
  // effect's deps change and the legacy effect below re-runs to restore
  // whatever the legacy drawMode state requests (polygon or static).
  useEffect(() => {
    const draw = terraDrawRef.current;
    if (!draw || !labelMode?.active) return;
    const tdMode = labelMode.type === "line" ? "linestring" : labelMode.type;
    if (!draw.enabled) draw.start();
    draw.setMode(tdMode);
  }, [labelMode?.active, labelMode?.type]);

  // Start / stop terra-draw in sync with the drawMode state.
  useEffect(() => {
    const draw = terraDrawRef.current;
    if (!draw) return;
    if (labelMode?.active) return; // label mode owns the draw — bail
    if (drawMode === "polygon") {
      if (!draw.enabled) draw.start();
      draw.setMode("polygon");
    } else {
      if (draw.enabled) draw.setMode("static");
    }
  }, [drawMode, labelMode?.active]);

  // Draw rectangle selection box.
  //
  // Previous bug (audit item #13): if the user pressed Escape mid-drag —
  // or dragged off the map and released the mouse outside the element —
  // the mousedown-registered handlers never fired their cleanup, so
  // ``drawStart.current`` stayed dangling AND the mousemove/mouseup
  // handlers stayed attached to the map. Worse: a delayed mouseup on
  // the page still committed a giant bbox via ``onAreaSelect``.
  //
  // Fix: lift the mouseup/mousemove refs + a shared ``cancelDraw()``
  // helper up to effect scope so an Escape keydown listener can reset
  // state AND detach the handlers. Also cancels ``drawMode`` itself so
  // the Draw Rectangle button flips back to its idle state visibly.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Store the currently-attached handlers so the Escape path can detach
    // them without having them in lexical scope.
    let activeMove: ((e: maplibregl.MapMouseEvent) => void) | null = null;
    let activeUp: ((e: maplibregl.MapMouseEvent) => void) | null = null;

    const cancelDraw = (commitBbox: BBox | null) => {
      if (activeMove) map.off("mousemove", activeMove);
      if (activeUp) map.off("mouseup", activeUp);
      activeMove = null;
      activeUp = null;
      map.dragPan.enable();
      drawStart.current = null;
      setDrawMode("none");
      if (commitBbox) {
        onAreaSelect(commitBbox);
      } else {
        // Aborted mid-drag — wipe the ghost selection rectangle so the
        // user doesn't see a leftover box from the cancelled drag.
        clearSelectionShape(map);
      }
    };

    const onMouseDown = (e: maplibregl.MapMouseEvent) => {
      if (!drawing) return;
      e.preventDefault();
      drawStart.current = { lng: e.lngLat.lng, lat: e.lngLat.lat };

      const onMouseMove = (ev: maplibregl.MapMouseEvent) => {
        if (!drawStart.current) return;
        const bbox: BBox = {
          west: Math.min(drawStart.current.lng, ev.lngLat.lng),
          south: Math.min(drawStart.current.lat, ev.lngLat.lat),
          east: Math.max(drawStart.current.lng, ev.lngLat.lng),
          north: Math.max(drawStart.current.lat, ev.lngLat.lat),
        };
        updateSelectionShape(map, bboxToPolygon(bbox));
      };

      const onMouseUp = (ev: maplibregl.MapMouseEvent) => {
        if (!drawStart.current) return;
        const bbox: BBox = {
          west: Math.min(drawStart.current.lng, ev.lngLat.lng),
          south: Math.min(drawStart.current.lat, ev.lngLat.lat),
          east: Math.max(drawStart.current.lng, ev.lngLat.lng),
          north: Math.max(drawStart.current.lat, ev.lngLat.lat),
        };
        cancelDraw(bbox);
      };

      activeMove = onMouseMove;
      activeUp = onMouseUp;
      map.dragPan.disable();
      map.on("mousemove", onMouseMove);
      map.on("mouseup", onMouseUp);
    };

    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key !== "Escape") return;
      // Two distinct cancel cases:
      //  1. Mid-drag: drawStart is set → cancel the drag + ignore the bbox.
      //  2. Idle in draw mode (no drag yet): exit draw mode so the button
      //     flips back from "Cancel" to "Draw Rectangle" without requiring
      //     a click. Matches typical drawing-tool UX conventions.
      if (drawing) {
        cancelDraw(null);
      }
    };

    map.on("mousedown", onMouseDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      map.off("mousedown", onMouseDown);
      window.removeEventListener("keydown", onKeyDown);
      // On effect teardown (drawing flipped off for any reason, component
      // unmount, etc.) make sure we're not leaving stale mousemove/mouseup
      // handlers attached.
      if (activeMove) map.off("mousemove", activeMove);
      if (activeUp) map.off("mouseup", activeUp);
      map.dragPan.enable();
    };
  }, [drawing, onAreaSelect]);

  // Render uploaded dataset GeoJSON overlays
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      // Remove previous overlay layers
      if (map.getLayer("overlay-fill")) map.removeLayer("overlay-fill");
      if (map.getLayer("overlay-line")) map.removeLayer("overlay-line");
      if (map.getLayer("overlay-circle")) map.removeLayer("overlay-circle");
      if (map.getSource("overlay")) map.removeSource("overlay");

      if (!overlayGeojson || overlayGeojson.length === 0) return;

      // Merge all features into one FeatureCollection
      const features: GeoJSON.Feature[] = [];
      for (const g of overlayGeojson) {
        if (g.type === "FeatureCollection") {
          features.push(...g.features);
        } else {
          features.push(g);
        }
      }
      const fc: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };

      map.addSource("overlay", { type: "geojson", data: fc });

      // Use per-feature color if available (auto-label results), fallback to orange
      const colorExpr: maplibregl.ExpressionSpecification = [
        "coalesce",
        ["get", "color"],       // auto-label raster results
        ["get", "auto_color"],  // auto-label vector results
        "#f59e0b",              // fallback for regular uploads
      ];

      map.addLayer({
        id: "overlay-fill",
        type: "fill",
        source: "overlay",
        paint: { "fill-color": colorExpr, "fill-opacity": 0.35 },
      });
      map.addLayer({
        id: "overlay-line",
        type: "line",
        source: "overlay",
        paint: { "line-color": colorExpr, "line-width": 1.5 },
      });
      map.addLayer({
        id: "overlay-circle",
        type: "circle",
        source: "overlay",
        paint: {
          "circle-color": colorExpr,
          "circle-radius": 6,
          "circle-stroke-color": "#fff",
          "circle-stroke-width": 1.5,
        },
      });

      // Add hover popup showing class name + confidence
      const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });
      map.on("mousemove", "overlay-fill", (e) => {
        if (!e.features || e.features.length === 0) return;
        const props = e.features[0].properties || {};
        const name = props.class_name || props.auto_class || props.name || "";
        const conf = props.confidence != null ? ` (${(props.confidence * 100).toFixed(0)}%)` : "";
        if (name) {
          popup.setLngLat(e.lngLat).setHTML(`<strong>${name}</strong>${conf}`).addTo(map);
        }
      });
      map.on("mouseleave", "overlay-fill", () => popup.remove());
      map.on("mousemove", "overlay-circle", (e) => {
        if (!e.features || e.features.length === 0) return;
        const props = e.features[0].properties || {};
        const name = props.class_name || props.auto_class || props.name || props.karst_type || "";
        const conf = props.confidence != null ? ` (${(props.confidence * 100).toFixed(0)}%)` : "";
        if (name) {
          popup.setLngLat(e.lngLat).setHTML(`<strong>${name}</strong>${conf}`).addTo(map);
        }
      });
      map.on("mouseleave", "overlay-circle", () => popup.remove());
    };
    if (map.isStyleLoaded()) apply();
    else map.on("load", apply);
  }, [overlayGeojson]);

  // Switch basemap tiles. Skip if the current map is already serving the
  // requested basemap — a redundant setStyle() during mount holds
  // `isStyleLoaded()` false long enough that downstream effects (imagery,
  // selection) can't successfully add sources/layers, and StrictMode's double
  // mount makes this easy to trip. Comparing source tile URLs is a reliable,
  // ref-free signal.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const targetUrl = BASEMAPS[basemap].url;
    const apply = () => {
      const source = map.getSource("osm") as maplibregl.RasterTileSource | undefined;
      const currentUrl = (source as unknown as { tiles?: string[] })?.tiles?.[0];
      if (currentUrl === targetUrl) return;
      if (source) {
        // Can't change tiles on existing source — swap style
        const center = map.getCenter();
        const zoom = map.getZoom();
        map.setStyle({
          version: 8,
          sources: {
            osm: {
              type: "raster",
              tiles: [BASEMAPS[basemap].url],
              tileSize: 256,
              attribution: basemap === "satellite" ? "Esri" : basemap === "dark" ? "CartoDB" : "OSM",
            },
          },
          layers: [{ id: "osm-tiles", type: "raster", source: "osm", minzoom: 0, maxzoom: 19 }],
        });
        // Restore position after style change
        map.once("styledata", () => {
          map.setCenter(center);
          map.setZoom(zoom);
          // Re-add selection after style swap — prefer the real polygon, fall
          // back to a rectangle built from bbox.
          const shape = selectedGeometry
            ? selectedGeometry
            : selectedArea
            ? bboxToPolygon(selectedArea)
            : null;
          if (shape) updateSelectionShape(map, shape);
        });
      }
    };
    if (map.isStyleLoaded()) apply();
    else map.on("load", apply);
  }, [basemap]);

  // Reconcile XYZ imagery layers (STAC composites, cloud-free mosaics, etc.)
  // with the map. Each layer has a stable id, so we can add new ones, drop
  // removed ones, and update opacity in place without full rebuilds. Imagery
  // slots in ABOVE the basemap but BELOW the selection highlight — that way
  // you can still see your polygon's outline over a satellite mosaic.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      // Filter out layers the user has toggled hidden via the Added Layer
      // panel. They stay in the upstream imageryLayers state (App still
      // knows about them) so the panel can render them as "hidden" and
      // let the user toggle back on — we just don't attach them here.
      const want = (imageryLayers ?? []).filter((l) => !hiddenLayerIds.has(l.id));
      // A vector-payload layer produces up to 3 style layers (fill / line
      // / circle) keyed off the same source, so the "wanted" set tracks
      // every style-layer id we might own, not just one per input.
      const wantIds = new Set<string>();
      for (const l of want) {
        if (l.featureCollection) {
          wantIds.add(`imagery-${l.id}-fill`);
          wantIds.add(`imagery-${l.id}-line`);
          wantIds.add(`imagery-${l.id}-circle`);
        } else {
          wantIds.add(`imagery-${l.id}`);
        }
      }
      // Drop any stale layers/sources from previous renders
      const style = map.getStyle();
      for (const layer of style?.layers ?? []) {
        if (layer.id.startsWith("imagery-") && !wantIds.has(layer.id)) {
          if (map.getLayer(layer.id)) map.removeLayer(layer.id);
          if (map.getSource(layer.id)) map.removeSource(layer.id);
        }
      }
      // Add or update each wanted layer
      for (const l of want) {
        const opacity = l.opacity ?? 1;
        if (l.featureCollection) {
          // VECTOR path — label set merged into the Added Layer list via
          // LabelPanel's "Add to map layers" button. Render the source
          // once and attach a fill layer (polygons + geometry fill for
          // MultiPolygon), a line layer (LineString + polygon outlines),
          // and a circle layer (Point). MapLibre auto-filters each layer
          // against the geometry type via the filter expression so one
          // FeatureCollection with mixed geometries still renders
          // correctly in each style layer.
          const sourceId = `imagery-${l.id}`;
          const existingSrc = map.getSource(sourceId) as
            | maplibregl.GeoJSONSource
            | undefined;
          if (existingSrc && typeof existingSrc.setData === "function") {
            existingSrc.setData(l.featureCollection);
          } else if (!existingSrc) {
            map.addSource(sourceId, {
              type: "geojson",
              data: l.featureCollection,
            });
          }
          // Use the feature's own `properties.color` when present (labels
          // set it via LabelPanel's colorForTag) so users see the color
          // they picked; fall back to a friendly neutral when missing.
          const fallback = "#3a6690";
          if (!map.getLayer(`imagery-${l.id}-fill`)) {
            map.addLayer({
              id: `imagery-${l.id}-fill`,
              type: "fill",
              source: sourceId,
              filter: ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
              paint: {
                "fill-color": ["coalesce", ["get", "color"], fallback],
                "fill-opacity": opacity * 0.35,
              },
            });
          } else {
            map.setPaintProperty(`imagery-${l.id}-fill`, "fill-opacity", opacity * 0.35);
          }
          if (!map.getLayer(`imagery-${l.id}-line`)) {
            map.addLayer({
              id: `imagery-${l.id}-line`,
              type: "line",
              source: sourceId,
              filter: ["in", ["geometry-type"], ["literal", ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]]],
              paint: {
                "line-color": ["coalesce", ["get", "color"], fallback],
                "line-width": 2,
                "line-opacity": opacity,
              },
            });
          } else {
            map.setPaintProperty(`imagery-${l.id}-line`, "line-opacity", opacity);
          }
          if (!map.getLayer(`imagery-${l.id}-circle`)) {
            map.addLayer({
              id: `imagery-${l.id}-circle`,
              type: "circle",
              source: sourceId,
              filter: ["in", ["geometry-type"], ["literal", ["Point", "MultiPoint"]]],
              paint: {
                "circle-color": ["coalesce", ["get", "color"], fallback],
                "circle-radius": 5,
                "circle-stroke-color": "#ffffff",
                "circle-stroke-width": 1.5,
                "circle-opacity": opacity,
              },
            });
          } else {
            map.setPaintProperty(`imagery-${l.id}-circle`, "circle-opacity", opacity);
          }
          continue;
        }
        // RASTER path — STAC / dataset-tiles / OlmoEarth inference.
        if (!l.tileUrl) continue;
        const id = `imagery-${l.id}`;
        if (map.getLayer(id)) {
          map.setPaintProperty(id, "raster-opacity", opacity);
          continue;
        }
        map.addSource(id, {
          type: "raster",
          tiles: [l.tileUrl],
          tileSize: 256,
        });
        map.addLayer({
          id,
          type: "raster",
          source: id,
          paint: { "raster-opacity": opacity },
        });
      }
    };
    if (map.isStyleLoaded()) apply();
    else map.on("load", apply);
  }, [imageryLayers, hiddenLayerIds]);

  // Pan + zoom to a newly-loaded dataset's bbox. Triggered from App on
  // sample-card click or file upload. `nonce` is a monotonic counter so
  // re-uploading the same dataset still re-flies. We pad bounds by ~2% so
  // the dataset doesn't kiss the map edges.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !flyToTrigger) return;
    const { bbox } = flyToTrigger;
    // We compute the target camera once via cameraForBounds, then jump —
    // deliberately skipping fitBounds' built-in animation. fitBounds with
    // `duration > 0` routes through easeTo which depends on rAF for frame
    // stepping, and any environment that throttles rAF (hidden tab,
    // headless browser, some preview iframes) lets the animation queue
    // but never advance → the camera silently stays put. jumpTo is
    // synchronous and always works. A slight UX cost (no smooth pan) for
    // a reliability win; worth it for the sample-load path.
    try {
      const cam = map.cameraForBounds(
        [[bbox.west, bbox.south], [bbox.east, bbox.north]],
        { padding: 60, maxZoom: 14 },
      );
      if (cam) map.jumpTo(cam);
    } catch {
      /* zero-area bbox or invalid coords — ignore */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flyToTrigger?.nonce]);

  // Show the current selection — either the drawn polygon or a rectangle
  // derived from bbox. Single `selection` source so only one highlight ever
  // renders on the map.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const shape: GeoJSON.Polygon | null = selectedGeometry
      ? selectedGeometry
      : selectedArea
      ? bboxToPolygon(selectedArea)
      : null;

    const apply = () => {
      if (!shape) {
        clearSelectionShape(map);
        return;
      }
      updateSelectionShape(map, shape);
    };
    if (map.isStyleLoaded()) apply();
    else map.on("load", apply);
  }, [selectedArea, selectedGeometry]);

  // Build legend from overlay features
  const legend = useMemo(() => {
    if (!overlayGeojson || overlayGeojson.length === 0) return [];
    const classMap = new Map<string, string>();
    for (const g of overlayGeojson) {
      const feats = g.type === "FeatureCollection" ? g.features : [g];
      for (const f of feats) {
        const p = f.properties || {};
        const name = p.class_name || p.auto_class || p.karst_type;
        const color = p.color || p.auto_color;
        if (name && color && !classMap.has(name)) {
          classMap.set(name, color);
        }
      }
    }
    return Array.from(classMap.entries()).map(([name, color]) => ({ name, color }));
  }, [overlayGeojson]);

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="w-full h-full" />

      {/* Draw buttons. The bbox coordinate readout that used to live here
          was removed — same numbers already render in the sidebar's
          "Selected Area" block (Sidebar.tsx). Showing both was visual
          noise; the sidebar version is more legible (W/S/E/N labels,
          stacked) and stays visible regardless of map zoom.
          Recenter button (right of the draw buttons) fits the map to
          whatever's currently selected — selectedGeometry takes priority
          over selectedArea so a manually-drawn polygon recenters tightly
          rather than to its bounding rectangle. Hidden when nothing is
          selected. */}
      <div className="absolute top-6 left-6 flex gap-4 z-10 items-center" data-testid="map-draw-controls">
        <button
          onClick={() => setDrawMode(drawMode === "rectangle" ? "none" : "rectangle")}
          data-testid="draw-rectangle-button"
          className={`px-6 py-3 text-white border-none rounded-xl cursor-pointer font-semibold text-sm shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg ${
            drawMode === "rectangle" ? "bg-gradient-to-br from-red-500 to-red-700" : "bg-gradient-primary"
          }`}
        >
          {drawMode === "rectangle" ? "Cancel" : "Draw Rectangle"}
        </button>
        <button
          onClick={() => setDrawMode(drawMode === "polygon" ? "none" : "polygon")}
          data-testid="draw-polygon-button"
          className={`px-6 py-3 text-white border-none rounded-xl cursor-pointer font-semibold text-sm shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg ${
            drawMode === "polygon" ? "bg-gradient-to-br from-red-500 to-red-700" : "bg-gradient-primary"
          }`}
        >
          {drawMode === "polygon" ? "Cancel" : "Draw Polygon"}
        </button>
        {(selectedArea || selectedGeometry) && (
          <button
            onClick={() => {
              const map = mapRef.current;
              if (!map) return;
              // Prefer the actual polygon geometry if present (tight fit
              // around the drawn shape). Fall back to the bbox rectangle.
              let bounds: [[number, number], [number, number]] | null = null;
              if (selectedGeometry?.coordinates?.[0]?.length) {
                let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
                for (const ring of selectedGeometry.coordinates) {
                  for (const [lon, lat] of ring) {
                    if (lon < minLon) minLon = lon;
                    if (lon > maxLon) maxLon = lon;
                    if (lat < minLat) minLat = lat;
                    if (lat > maxLat) maxLat = lat;
                  }
                }
                if (Number.isFinite(minLon)) bounds = [[minLon, minLat], [maxLon, maxLat]];
              }
              if (!bounds && selectedArea) {
                bounds = [[selectedArea.west, selectedArea.south], [selectedArea.east, selectedArea.north]];
              }
              if (!bounds) return;
              try {
                // Same jumpTo-via-cameraForBounds pattern as the flyToTrigger
                // path — reliable in preview iframes / hidden tabs where the
                // animated easeTo silently stalls on rAF throttling.
                const cam = map.cameraForBounds(bounds, { padding: 80, maxZoom: 14 });
                if (cam) map.jumpTo(cam);
              } catch { /* zero-area or invalid coords — ignore */ }
            }}
            data-testid="recenter-to-selection"
            title="Recenter map on the selected area"
            className="p-3 bg-gradient-panel backdrop-blur-sm border border-geo-border rounded-xl text-geo-text shadow-md cursor-pointer transition-all hover:-translate-y-0.5 hover:shadow-lg hover:bg-geo-elevated"
          >
            <FitToBounds className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Layers control — replaces the 3-pill OSM/Satellite/Dark row with
          a single consolidated panel that splits cleanly into:
            • Base Layer  (pick one: OSM / Satellite / Dark)
            • Added Layer (list: inference outputs + imported rasters,
              each with a visibility toggle + remove button)
          The pill-row was acceptable when there were only 3 basemaps, but
          with inference outputs + imported rasters stacking on top of it,
          users had no way to see what was on the map or toggle them
          off without going back to the sidebar. This overlay is the
          single source of truth for "what's visible on the map".
          Optional ``leadingBasemapControl`` (Compare toggle) stays at the
          far right so it's adjacent to the pickers without crowding them. */}
      <div className="absolute top-6 right-24 flex items-start gap-3 z-10">
        <MapLayersControl
          basemap={basemap}
          onBasemapChange={setBasemap}
          imageryLayers={imageryLayers ?? []}
          hiddenLayerIds={hiddenLayerIds}
          onToggleHidden={toggleLayerHidden}
          onRemoveImageryLayer={onRemoveImageryLayer}
          onAddImageryLayer={onAddImageryLayer}
          datasets={datasets ?? []}
          olmoCache={olmoCache ?? {}}
          selectedArea={selectedArea}
        />
        {leadingBasemapControl}
      </div>

      {/* Minimal color-only legend strip. Split 2026-04-21 per product
          direction: the MAP shows only the raster color representation
          (swatches / gradient bar so users can decode colors visually),
          and the TEXT DETAILS (LLM explanation, class list, scene info)
          live in the Analysis tab's RasterResultsAccordion. Separating
          the two stops the legend from crowding the map AND keeps the
          color key always visible at a glance. */}
      <MapRasterColorStrip imageryLayers={imageryLayers ?? []} />

      {/* Drawing overlay hints */}
      {drawMode === "rectangle" && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/70 backdrop-blur text-white px-6 py-3 rounded-lg text-sm pointer-events-none z-10">
          Click and drag to select an area
        </div>
      )}
      {drawMode === "polygon" && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/70 backdrop-blur text-white px-6 py-3 rounded-lg text-sm pointer-events-none z-10">
          Click to add vertices · double-click to close the polygon
        </div>
      )}

      {/* Label legend */}
      {legend.length > 0 && (
        <div className="absolute bottom-6 left-6 bg-gradient-panel backdrop-blur-sm px-6 py-5 rounded-xl z-10 shadow-md max-w-[240px] border border-geo-border">
          <div className="text-xs font-semibold text-geo-muted mb-4 uppercase tracking-[0.15em]">
            Labels
          </div>
          <div className="space-y-3">
          {legend.map((item) => (
            <div
              key={item.name}
              className="flex items-center gap-3"
            >
              <div
                className="w-3.5 h-3.5 rounded shrink-0 border border-geo-border"
                style={{ background: item.color }}
              />
              <span className="text-sm text-geo-text">{item.name}</span>
            </div>
          ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Minimal color-only legend strip for the MAP. Shows one compact card
 * per inference raster with the task's color swatches (classification →
 * first few class colors; gradient task → mini gradient bar) and the
 * pretty model name. No click, no expand, no explanation text — that
 * lives in the Analysis tab now. Keeps the map uncluttered while still
 * letting users decode "which color means what" at a glance.
 *
 * Hidden entirely when no inference layers are active.
 */
function MapRasterColorStrip({
  imageryLayers,
}: {
  imageryLayers: ImageryLayer[];
}) {
  const inferLayers = imageryLayers.filter((l) => l.inferenceMetadata);
  if (inferLayers.length === 0) return null;

  return (
    <div
      className="absolute top-24 right-6 z-10 flex flex-col gap-1.5 max-w-[260px]"
      data-testid="map-raster-color-strip"
    >
      {inferLayers.map((layer) => {
        const meta = layer.inferenceMetadata!;
        const legend = meta.legend;
        const prettyName = meta.model_repo_id
          .replace(/^allenai\//, "")
          .replace(/^OlmoEarth-v1-FT-/, "")
          .replace(/^OlmoEarth-v1-/, "");

        // Classification / segmentation → first 6 class swatches +
        //   ellipsis if more exist.
        // Regression / embedding → mini gradient bar spanning the stops.
        // Stub → amber dot.
        let body: React.ReactNode = null;
        if (meta.kind === "stub") {
          body = (
            <span className="text-[9px] text-amber-700 font-semibold">
              preview stub — inference failed
            </span>
          );
        } else if (legend && "classes" in legend && legend.classes?.length) {
          // Filter to ONLY the classes actually present in the rendered
          // raster. Prevents a 110-class ecosystem legend from showing
          // 110 swatches when the user's AOI only paints 4 of them. Falls
          // back to the full catalog when the backend didn't send
          // `present_class_ids` (older response shape / classification
          // tasks without a per-pixel raster).
          const presentIds = meta.present_class_ids;
          const shown =
            presentIds && presentIds.length > 0
              ? legend.classes.filter((c) => presentIds.includes(c.index))
              : legend.classes;
          const head = shown.slice(0, 8);
          const extra = shown.length - head.length;
          body = (
            <div className="flex items-center gap-1 flex-wrap">
              {head.map((c) => (
                <span
                  key={c.index}
                  className="w-3 h-3 rounded-sm border border-geo-border/60 flex-shrink-0"
                  style={{ background: c.color }}
                  title={`${c.index} · ${c.name}`}
                />
              ))}
              {extra > 0 && (
                <span className="text-[9px] text-geo-muted" title={`+${extra} more classes on this tile`}>
                  +{extra}
                </span>
              )}
            </div>
          );
        } else if (legend && "stops" in legend && legend.stops?.length) {
          const gradient = `linear-gradient(to right, ${legend.stops
            .map(([color, pos]) => `${color} ${(pos * 100).toFixed(0)}%`)
            .join(", ")})`;
          body = (
            <div
              className="h-2 rounded-sm border border-geo-border/60"
              style={{ backgroundImage: gradient }}
              aria-label="colormap gradient"
            />
          );
        }

        return (
          <div
            key={layer.id}
            className="bg-gradient-panel border border-geo-border rounded-lg shadow-md px-2.5 py-1.5"
          >
            <div
              className="text-[10px] font-mono text-geo-text truncate mb-1"
              title={meta.model_repo_id}
            >
              {prettyName}
            </div>
            {body}
          </div>
        );
      })}
    </div>
  );
}


/** Click-to-inspect accordion for raster results — mounted in the
 * Analysis tab. One pill per inference raster; clicking a pill expands
 * it to show the LLM explanation + scene metadata + full class list.
 * Exported so Sidebar's Analysis view can render it without owning the
 * ImageryLayer type / legend rendering.
 *
 * Moved from the MapView overlay (2026-04-21) per product direction: the
 * MAP now carries only color-swatch representation, TEXT DETAILS live
 * in Analysis.
 */
export function RasterResultsAccordion({
  imageryLayers,
  onRemoveImageryLayer,
}: {
  imageryLayers: ImageryLayer[];
  onRemoveImageryLayer?: (id: string) => void;
}) {
  // Two buckets:
  //   - inference layers (with `inferenceMetadata`) — get the existing
  //     accordion-with-RasterExplanation treatment so the user can read the
  //     LLM agent's plain-language description of what each class color means.
  //   - non-inference raster layers (Sample Rasters, uploaded GeoTIFFs, STAC
  //     composites) — get a one-row summary with a viridis-gradient swatch
  //     so the user can see what's *actually on the map* and what color
  //     scheme is in play. Without this row Sample Rasters appeared in the
  //     Added Layer popover but never in the Analysis tab summary.
  // Vector layers (label sets pushed via "Add to map layers") are excluded
  // — they have their own representation in the LabelPanel section.
  const inferLayers = imageryLayers.filter((l) => l.inferenceMetadata);
  const otherRasterLayers = imageryLayers.filter(
    (l) => !l.inferenceMetadata && l.tileUrl,
  );
  const [openLayerId, setOpenLayerId] = useState<string | null>(null);
  const inferKey = inferLayers.map((l) => l.id).join("|");
  useEffect(() => {
    if (inferLayers.length === 0) {
      setOpenLayerId(null);
      return;
    }
    setOpenLayerId((cur) => {
      if (cur && inferLayers.some((l) => l.id === cur)) return cur;
      return inferLayers[inferLayers.length - 1].id;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inferKey]);

  if (inferLayers.length === 0 && otherRasterLayers.length === 0) return null;

  return (
    <div className="space-y-1.5" data-testid="raster-results-accordion">
      {inferLayers.map((layer) => {
        const meta = layer.inferenceMetadata!;
        const open = openLayerId === layer.id;
        const isStub = meta.kind === "stub";
        let pillColor = "#6b7280";
        const legend = meta.legend;
        if (isStub) {
          pillColor = "#f59e0b";
        } else if (legend && "classes" in legend && legend.classes?.[0]?.color) {
          pillColor = legend.classes[0].color;
        } else if (legend && "stops" in legend && legend.stops?.[0]?.[0]) {
          pillColor = legend.stops[0][0];
        }
        const prettyName = meta.model_repo_id
          .replace(/^allenai\//, "")
          .replace(/^OlmoEarth-v1-FT-/, "")
          .replace(/^OlmoEarth-v1-/, "");
        return (
          <div
            key={layer.id}
            className={`bg-gradient-panel border rounded-lg shadow-sm overflow-hidden transition-colors ${
              open ? "border-geo-accent" : "border-geo-border"
            }`}
            data-testid={`result-pill-${layer.id}`}
          >
            <button
              type="button"
              onClick={() => setOpenLayerId(open ? null : layer.id)}
              className={`w-full flex items-center gap-2 px-2.5 py-1.5 text-left cursor-pointer transition-colors ${
                open ? "bg-geo-accent/10" : "hover:bg-geo-bg/60"
              }`}
              title={open ? "Collapse" : "Click for the agent's explanation"}
            >
              <span
                className="w-3 h-3 rounded-sm border border-geo-border/60 flex-shrink-0"
                style={{ background: pillColor }}
                aria-hidden="true"
              />
              <span className="flex-1 min-w-0">
                <span className="block font-mono text-[11px] text-geo-text truncate">
                  {prettyName}
                </span>
              </span>
              <span
                className={`text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded flex-shrink-0 ${
                  isStub
                    ? "bg-amber-200 text-amber-800"
                    : "bg-geo-accent/10 text-geo-accent"
                }`}
              >
                {isStub ? "stub" : meta.task_type ?? meta.kind}
              </span>
              <span
                className={`text-geo-muted text-[10px] transition-transform ${
                  open ? "rotate-180" : ""
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {open && (
              <div className="border-t border-geo-border p-2.5 max-h-[50vh] overflow-y-auto">
                <RasterExplanation
                  layer={layer}
                  onRemove={onRemoveImageryLayer}
                />
              </div>
            )}
          </div>
        );
      })}
      {/* Non-inference rasters — Sample Rasters, uploaded GeoTIFFs, STAC
          composites. Single row each: viridis-gradient swatch (matches
          raster_tiles.py's _DEFAULT_STOPS for 1-band layers) + the layer's
          label + remove button. No accordion since there's no per-class
          legend to expand into; the colormap IS the legend. */}
      {otherRasterLayers.length > 0 && (
        <div className="mt-2 pt-2 border-t border-geo-border space-y-1">
          <div className="text-[10px] font-mono uppercase tracking-wider text-geo-muted px-1 mb-1">
            Other rasters on map
          </div>
          {otherRasterLayers.map((layer) => {
            // Detect render path from the tile URL: /api/datasets/...
            // means render_geotiff_tile (viridis colormap or RGB composite),
            // anything else (STAC composites, etc.) is a tinted XYZ tile.
            const isDatasetTile = !!layer.tileUrl?.includes("/api/datasets/");
            const isStacComposite = !!layer.tileUrl?.includes("/stac/");
            const subtitle = isDatasetTile
              ? "viridis colormap (1-band) or RGB composite (≥3-band)"
              : isStacComposite
                ? "Sentinel-2 true-color composite"
                : "XYZ raster tiles";
            return (
              <div
                key={layer.id}
                className="flex items-center gap-2 px-2 py-1.5 bg-geo-surface border border-geo-border rounded text-[11px]"
                data-testid={`other-raster-${layer.id}`}
              >
                {/* Viridis gradient swatch — visual cue that the colors on
                    the map come from a perceptually-uniform colormap, not
                    the bbox highlight. Matches raster_tiles.py stops. */}
                <span
                  className="w-6 h-3 rounded-sm border border-geo-border/60 flex-shrink-0"
                  style={{
                    background: isDatasetTile
                      ? "linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)"
                      : "linear-gradient(to right, #1e3a5f, #5b8bb5, #c0d4e6)",
                  }}
                  title="Colormap (low → high)"
                  aria-hidden="true"
                />
                <span className="flex-1 min-w-0">
                  <span className="block font-mono text-geo-text truncate">
                    {layer.label || layer.id}
                  </span>
                  <span className="block text-[9px] text-geo-dim truncate">
                    {subtitle}
                  </span>
                </span>
                {onRemoveImageryLayer && (
                  <button
                    type="button"
                    onClick={() => onRemoveImageryLayer(layer.id)}
                    className="text-geo-dim hover:text-red-700 cursor-pointer text-[14px] leading-none px-1"
                    title="Remove from map"
                  >
                    ×
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


/** LLM agent explanation of what a single inference raster represents.
 *
 * Fires ``/api/explain-raster`` on first render (then caches the reply
 * for the lifetime of this layer's pill expansion). Passes the raster's
 * model / task / scene / top-class metadata so the LLM can produce an
 * AOI-aware summary instead of a generic blurb.
 *
 * Provider chain on the server is NIM → Claude → Gemma → deterministic
 * fallback; the returned ``source`` drives the small attribution pill
 * at the bottom of the card so users can tell a free-tier NIM answer
 * apart from a templated fallback. The full class list is available
 * behind a ``<details>`` disclosure for users who really do want the
 * raw catalog.
 */
function RasterExplanation({
  layer,
  onRemove,
}: {
  layer: ImageryLayer;
  onRemove?: (id: string) => void;
}) {
  const meta = layer.inferenceMetadata!;
  const isStub = meta.kind === "stub";

  // LLM-explanation block removed 2026-04-21. Users preferred the raw
  // classes-on-tile list + scene info over the templated-summary
  // fallback text that polluted the panel when no NIM key was
  // configured. The explain-raster endpoint still exists (MCP tool
  // surface, future re-enable) but the UI no longer auto-fires it.

  return (
    <div className="space-y-2 text-[11px]">
      {/* Compact scene metadata strip — date + cloud so users can tell
          which Sentinel-2 scene the raster was computed on. */}
      <div className="text-[10px] text-geo-muted font-mono truncate" title={meta.scene_id ?? undefined}>
        {meta.scene_id ? `${meta.scene_id.slice(0, 24)}…` : "no scene id"}
        {meta.scene_datetime ? ` · ${meta.scene_datetime.split("T")[0]}` : ""}
        {typeof meta.scene_cloud_cover === "number"
          ? ` · ${meta.scene_cloud_cover.toFixed(1)}% cloud`
          : ""}
      </div>

      {isStub && meta.stub_reason && (
        <div className="px-2 py-1.5 rounded bg-amber-100 border border-amber-300 text-amber-900 text-[10px] leading-snug">
          <span className="font-bold">PREVIEW STUB · </span>
          {meta.stub_reason.slice(0, 160)}
          {meta.stub_reason.length > 160 ? "…" : ""}
        </div>
      )}

      {/* Download button — serves the raster as a georeferenced GeoTIFF
          straight from the job's in-memory class / scalar raster. Only
          shown when we have a job_id AND it's not a stub (stub rasters
          are synthetic gradients, not real model output). */}
      {meta.job_id && !isStub && (
        <a
          href={`/api/olmoearth/download/${meta.job_id}.tif`}
          download={`${meta.model_repo_id.split("/").pop()}-${meta.job_id}.tif`}
          className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-geo-accent hover:text-geo-accent-hover px-2 py-1 border border-geo-accent/40 rounded hover:bg-geo-accent/10 cursor-pointer transition-colors"
          title="Download the prediction raster as a georeferenced GeoTIFF (open in QGIS, ArcGIS, rasterio, etc.)"
        >
          ↓ download GeoTIFF
        </a>
      )}

      {/* Classes actually present on this tile — primary view. For a
          segmentation raster the user only cares about the 4-6 classes
          that paint their AOI, not the full 110-entry catalog. Hidden
          entirely for tasks without per-pixel class info. */}
      {meta.class_names && meta.present_class_ids && meta.present_class_ids.length > 0 && (
        <div className="border border-geo-border rounded">
          <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-geo-muted border-b border-geo-border">
            Classes on this tile ({meta.present_class_ids.length})
          </div>
          <ul className="px-2 py-1 max-h-40 overflow-y-auto text-[10px] space-y-0.5">
            {meta.present_class_ids.map((id) => {
              const name = meta.class_names?.[id] ?? `class_${id}`;
              // Match the color from the legend's classes array when we
              // can; otherwise leave the swatch blank.
              const legend = meta.legend;
              const color =
                legend && "classes" in legend
                  ? legend.classes.find((c) => c.index === id)?.color
                  : undefined;
              return (
                <li key={id} className="flex items-center gap-2">
                  {color && (
                    <span
                      className="w-2.5 h-2.5 rounded-sm border border-geo-border/60 flex-shrink-0"
                      style={{ background: color }}
                      aria-hidden="true"
                    />
                  )}
                  <span className="font-mono text-geo-muted w-6 flex-shrink-0">{id}</span>
                  <span className="truncate" title={name}>{name}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Full catalog — behind a collapsible disclosure for power users
          who want to see the classes that DIDN'T paint their AOI too. */}
      {meta.class_names && meta.class_names.length > 0 && (
        <details className="border border-geo-border rounded">
          <summary className="px-2 py-1 text-[10px] cursor-pointer text-geo-muted hover:text-geo-text hover:bg-geo-bg/60">
            View full class catalog ({meta.class_names.length})
          </summary>
          <ul className="px-2 py-1 max-h-40 overflow-y-auto text-[10px] space-y-0.5">
            {meta.class_names.map((name, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="font-mono text-geo-muted w-6 flex-shrink-0">{i}</span>
                <span className="truncate" title={name}>{name}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* Regression task — no classes; surface the predicted value, units,
          and the value-range gradient (legend.stops). Mirrors the
          structure of the segmentation block (boxed section + header) so
          every task type reads consistently in the panel. */}
      {meta.task_type === "regression" && (
        <div className="border border-geo-border rounded">
          <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-geo-muted border-b border-geo-border">
            Regression output{meta.units ? ` (${meta.units})` : ""}
          </div>
          <div className="px-2 py-1.5 text-[10px] space-y-1">
            {typeof meta.prediction_value === "number" && (
              <div className="flex items-baseline gap-2">
                <span className="text-geo-muted">Predicted value:</span>
                <span className="font-mono text-geo-text font-semibold">
                  {meta.prediction_value.toFixed(3)}
                  {meta.units ? ` ${meta.units}` : ""}
                </span>
              </div>
            )}
            {meta.legend && "stops" in meta.legend && meta.legend.stops?.length ? (
              <>
                <div
                  className="h-2 rounded border border-geo-border/60"
                  style={{
                    background: `linear-gradient(to right, ${meta.legend.stops
                      .map(([color]) => color)
                      .join(", ")})`,
                  }}
                  aria-hidden="true"
                />
                <div className="flex justify-between text-[9px] font-mono text-geo-muted">
                  <span>{meta.legend.stops[0][1]?.toFixed?.(2)}</span>
                  <span>{meta.legend.stops[meta.legend.stops.length - 1][1]?.toFixed?.(2)}</span>
                </div>
              </>
            ) : null}
          </div>
        </div>
      )}

      {/* Embedding task — encoder-only (no FT head). Surface the embedding
          dim + the viridis-style colormap used for the PCA-projected
          scalar raster, again in the same boxed-section style for parity. */}
      {meta.task_type === "embedding" && (
        <div className="border border-geo-border rounded">
          <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-geo-muted border-b border-geo-border">
            Encoder embedding
          </div>
          <div className="px-2 py-1.5 text-[10px] space-y-1">
            {typeof meta.embedding_dim === "number" && (
              <div className="flex items-baseline gap-2">
                <span className="text-geo-muted">Embedding dim:</span>
                <span className="font-mono text-geo-text">{meta.embedding_dim}</span>
              </div>
            )}
            <div className="text-geo-dim leading-snug">
              Per-patch embedding projected to a scalar via PCA (1st component),
              rescaled to [0, 1], then colormapped.
            </div>
            <div
              className="h-2 rounded border border-geo-border/60"
              style={{ background: "linear-gradient(to right, #0f172a, #6366f1, #f59e0b)" }}
              aria-hidden="true"
            />
            <div className="flex justify-between text-[9px] font-mono text-geo-muted">
              <span>low</span><span>high</span>
            </div>
          </div>
        </div>
      )}

      {onRemove && (
        <button
          type="button"
          onClick={() => onRemove(layer.id)}
          className="text-[10px] uppercase tracking-wider text-geo-muted hover:text-geo-danger cursor-pointer"
        >
          remove layer
        </button>
      )}
    </div>
  );
}


/** Consolidated layers panel — replaces the old 3-pill basemap row with:
 *   Base Layer  → one-of OSM / Satellite / Dark (derived from Map Layers)
 *   Added Layer → per-row toggle + remove, listing every current imagery
 *                 source (FT OlmoEarth inference outputs, imported rasters,
 *                 STAC composites from DatasetDetail, etc.)
 * A hover-menu on "Base Layer" and "Added Layer" keeps the on-screen
 * footprint small — the same pattern the Sidebar uses for Map / LLM tab
 * sub-views so the interaction feels native to the app.
 */
function MapLayersControl({
  basemap,
  onBasemapChange,
  imageryLayers,
  hiddenLayerIds,
  onToggleHidden,
  onRemoveImageryLayer,
  onAddImageryLayer,
  datasets,
  olmoCache,
  selectedArea,
}: {
  basemap: BasemapStyle;
  onBasemapChange: (b: BasemapStyle) => void;
  imageryLayers: ImageryLayer[];
  hiddenLayerIds: Set<string>;
  onToggleHidden: (id: string) => void;
  onRemoveImageryLayer?: (id: string) => void;
  onAddImageryLayer?: (layer: ImageryLayer) => void;
  datasets: DatasetInfo[];
  olmoCache: Record<string, OlmoEarthRepoStatus>;
  selectedArea: BBox | null;
}) {
  // Either menu open at a time — clicking the other button swaps focus.
  // Closed by default so the control's resting state is two compact pills.
  const [openPanel, setOpenPanel] = useState<"base" | "added" | null>(null);
  // Which tab is active inside the Added Layer popover. Added Layer is
  // the map's source-browser for LAYERS — raster data mounted on the
  // map. The old "Samples" tab was removed 2026-04-21 because the
  // preset datasets (SF Parks, PA Karst, Solar Sites, Knoxville NDVI)
  // are labeled reference GeoJSONs, not raster map layers. They now
  // live under Map tab → "Sample Label" which is the semantically
  // accurate home. Remaining tabs:
  //   • "on-map"    — current layers with toggle/remove
  //   • "olmoearth" — curated OlmoEarth FT heads + cache status
  //   • "import"    — OlmoEarth / OlmoEarth-FT HF-repo load form
  const [addedTab, setAddedTab] = useState<"on-map" | "olmoearth" | "import">(
    imageryLayers.length > 0 ? "on-map" : "olmoearth",
  );
  const addedCount = imageryLayers.length;
  const visibleCount = imageryLayers.filter((l) => !hiddenLayerIds.has(l.id)).length;

  // Pre-fill value for the Import tab when the user clicks a row's "import"
  // chip in the OlmoEarth tab. Kept at this level so the child
  // OlmoEarthImport component re-mounts with the right repo selection
  // when we flip tabs. The actual form state lives inside that component.
  const [importPrefill, setImportPrefill] = useState<string | undefined>();
  return (
    <div className="relative flex items-center gap-2">
      {/* Base Layer pill — stacked layout: tiny-caps label ABOVE the value,
          both left-aligned inside the button. Earlier we tried putting them
          inline (label left, value right on a single row) but the 11px
          uppercase caps sat on a different baseline than the 13px value
          font, so users read "BASE LAYER" and "Satellite" as two
          mis-aligned fragments. Stacking gives a clean two-line card: the
          label tells you WHAT you're picking, the value tells you WHICH
          option is active. Same pattern used for Added Layer. */}
      <button
        type="button"
        onClick={() => setOpenPanel((cur) => (cur === "base" ? null : "base"))}
        className={`px-4 py-2 rounded-lg cursor-pointer shadow-md border transition-all flex flex-col items-start leading-tight min-w-[100px] ${
          openPanel === "base"
            ? "bg-gradient-primary text-white border-transparent"
            : "bg-gradient-panel text-geo-text border-geo-border hover:border-geo-accent"
        }`}
        title="Pick a background basemap for the map"
      >
        <span className={`text-[10px] uppercase tracking-wider font-medium ${openPanel === "base" ? "opacity-80" : "text-geo-muted"}`}>
          Base Layer
        </span>
        <span className="text-[13px] font-semibold">{BASEMAPS[basemap].label}</span>
      </button>

      {/* Added Layer pill — ALWAYS visible now (previously gated on
          addedCount > 0). It's the single source-browser for the map:
          Samples + OlmoEarth cached datasets + user imports + a new
          "Import OlmoEarth/OlmoEarth-FT data" form. When nothing is
          attached yet, clicking it lands on the Samples tab so the user
          has a one-click demo path without going to the sidebar. */}
      <button
        type="button"
        onClick={() => setOpenPanel((cur) => (cur === "added" ? null : "added"))}
        className={`px-4 py-2 rounded-lg cursor-pointer shadow-md border transition-all flex flex-col items-start leading-tight min-w-[100px] ${
          openPanel === "added"
            ? "bg-gradient-primary text-white border-transparent"
            : "bg-gradient-panel text-geo-text border-geo-border hover:border-geo-accent"
        }`}
        title="Browse sample data, OlmoEarth cached datasets, imports, or import a new OlmoEarth/OlmoEarth-FT repo"
      >
        <span className={`text-[10px] uppercase tracking-wider font-medium ${openPanel === "added" ? "opacity-80" : "text-geo-muted"}`}>
          Added Layer
        </span>
        <span className="text-[13px] font-semibold">
          {addedCount === 0 ? (
            <span className="opacity-70 font-normal">none · browse</span>
          ) : (
            <>
              {visibleCount}
              {visibleCount !== addedCount && (
                <span className="opacity-60"> / {addedCount}</span>
              )}
              <span className="opacity-60 font-normal"> on map</span>
            </>
          )}
        </span>
      </button>

      {/* Popover — anchors to the control row's top-right so it doesn't
          drift over the MapLibre controls in the top-right corner. Click-
          outside handler would be overkill; the toggle itself closes it. */}
      {openPanel === "base" && (
        <div className="absolute top-full right-0 mt-2 w-[200px] bg-gradient-panel border border-geo-border rounded-lg shadow-lg overflow-hidden">
          {(Object.keys(BASEMAPS) as BasemapStyle[]).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => {
                onBasemapChange(key);
                setOpenPanel(null);
              }}
              className={`w-full text-left px-4 py-2.5 text-[12px] font-medium cursor-pointer transition-colors ${
                basemap === key
                  ? "bg-geo-accent/10 text-geo-accent"
                  : "text-geo-text hover:bg-geo-bg/60"
              }`}
            >
              {BASEMAPS[key].label}
            </button>
          ))}
        </div>
      )}

      {openPanel === "added" && (
        <div className="absolute top-full right-0 mt-2 w-[340px] bg-gradient-panel border border-geo-border rounded-lg shadow-lg overflow-hidden">
          {/* Tab bar — 4 equal-width pills mapping onto the four data
              origins: what's already on the map, preset samples, cached
              OlmoEarth repos, and a new OlmoEarth/FT import form. */}
          <div className="flex border-b border-geo-border bg-geo-bg/40">
            {(
              [
                ["on-map", `On map${addedCount > 0 ? ` (${addedCount})` : ""}`],
                ["olmoearth", "OlmoEarth"],
                ["import", "Import"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setAddedTab(key)}
                className={`flex-1 px-2 py-2 text-[11px] font-semibold cursor-pointer transition-colors ${
                  addedTab === key
                    ? "text-geo-accent border-b-2 border-geo-accent -mb-px bg-geo-surface"
                    : "text-geo-muted hover:text-geo-text hover:bg-geo-bg/60"
                }`}
                data-testid={`added-tab-${key}`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="max-h-[360px] overflow-y-auto">
            {/* === On-map tab ===================================== */}
            {addedTab === "on-map" && (
              addedCount === 0 ? (
                <div className="px-3 py-6 text-[11px] text-geo-muted text-center">
                  Nothing on the map yet. Switch to Samples, OlmoEarth, or Import to add a layer.
                </div>
              ) : (
                <div className="divide-y divide-geo-border">
                  {imageryLayers.map((l) => {
                    const hidden = hiddenLayerIds.has(l.id);
                    return (
                      <div
                        key={l.id}
                        className="flex items-center gap-2 px-3 py-2 text-[12px] hover:bg-geo-bg/40"
                      >
                        <button
                          type="button"
                          onClick={() => onToggleHidden(l.id)}
                          className={`w-7 h-4 rounded-full flex-shrink-0 transition-colors relative cursor-pointer ${
                            hidden ? "bg-geo-border" : "bg-geo-accent"
                          }`}
                          title={hidden ? "Show on map" : "Hide from map (still in list)"}
                          aria-label={`Toggle visibility for ${l.label ?? l.id}`}
                        >
                          <span
                            className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all ${
                              hidden ? "left-0.5" : "left-3.5"
                            }`}
                          />
                        </button>
                        <div className="flex-1 min-w-0">
                          <div
                            className={`truncate ${hidden ? "text-geo-muted line-through" : "text-geo-text"}`}
                            title={l.label ?? l.id}
                          >
                            {l.label ?? l.id}
                          </div>
                          {l.inferenceMetadata ? (
                            <div className="text-[9px] text-geo-muted truncate">
                              {l.inferenceMetadata.model_repo_id.split("/").pop()}
                              {l.inferenceMetadata.kind === "stub" && (
                                <span className="ml-1 text-amber-600 font-semibold">
                                  · preview stub
                                </span>
                              )}
                            </div>
                          ) : l.featureCollection ? (
                            <div className="text-[9px] text-geo-muted truncate">
                              vector · {l.featureCollection.features.length} feature
                              {l.featureCollection.features.length === 1 ? "" : "s"}
                            </div>
                          ) : null}
                        </div>
                        {onRemoveImageryLayer && (
                          <button
                            type="button"
                            onClick={() => onRemoveImageryLayer(l.id)}
                            className="text-geo-muted hover:text-geo-danger cursor-pointer text-lg leading-none px-1 flex-shrink-0"
                            title="Remove from map"
                            aria-label={`Remove ${l.label ?? l.id}`}
                          >
                            ×
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )
            )}

            {/* Samples tab removed 2026-04-21 — the preset datasets
                (SF Parks, PA Karst, Solar Sites, Knoxville NDVI) are
                labeled GeoJSONs, not raster layers, and live under
                Map tab → "Sample Label" now. */}

            {/* === OlmoEarth tab ================================== */}
            {/* Curated FT-head list. Each row shows cache status inline:
                cached (green, ready to run) vs not-cached (grey, with a
                shortcut to the Import tab). This matches the user-stated
                five heads — no more noisy base-encoder / project-dataset
                rows. If nothing in the allowlist is cached we still show
                the list so users can see what's available and one-click
                download. */}
            {addedTab === "olmoearth" && (
              <div className="divide-y divide-geo-border">
                {OLMOEARTH_FT_HEADS.map(({ repoId, task, colorKey, supported }) => {
                  const cov = DATASET_COVERAGE[repoId];
                  const info = olmoCache[repoId];
                  const cached = info?.status === "cached";
                  const loading = info?.status === "loading";
                  const errored = info?.status === "error";
                  const size = cached && info?.size_bytes
                    ? `${(info.size_bytes / 1_000_000).toFixed(0)} MB`
                    : "";
                  return (
                    <div
                      key={repoId}
                      className="flex items-start gap-2 px-3 py-2 text-[12px] hover:bg-geo-bg/40"
                    >
                      <span
                        className="w-3 h-3 rounded-sm mt-0.5 flex-shrink-0"
                        style={{ background: cov?.color ?? "#6b7280" }}
                        title={`colormap: ${colorKey}`}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-[11px] text-geo-text truncate flex items-center gap-1.5" title={repoId}>
                          {repoId.replace(/^allenai\//, "")}
                          {!supported && (
                            <span
                              className="text-[8px] uppercase tracking-wider px-1 py-0.5 rounded bg-amber-100 text-amber-800 font-semibold"
                              title="Decoder shape not supported by the FT-loader yet — running inference will fall back to a preview stub"
                            >
                              beta
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-geo-muted">
                          {task}
                          {size && ` · ${size}`}
                        </div>
                      </div>
                      {/* Cache-status pill — tells the user whether the
                          repo is ready to run inference or still needs a
                          download step. Clicking "not cached" jumps into
                          the Import tab with the repo id pre-filled so
                          it's a one-click path to load it. */}
                      {cached ? (
                        <span
                          className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-geo-success/15 text-geo-success flex-shrink-0"
                          title="Weights are on disk — ready to run from the OlmoEarth tab"
                        >
                          cached
                        </span>
                      ) : loading ? (
                        <span
                          className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-geo-accent/15 text-geo-accent flex-shrink-0"
                          title="Download in progress"
                        >
                          loading…
                        </span>
                      ) : errored ? (
                        <span
                          className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-geo-danger/15 text-geo-danger flex-shrink-0"
                          title={info?.error ?? "Previous download failed"}
                        >
                          error
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => {
                            setImportPrefill(repoId);
                            setAddedTab("import");
                          }}
                          className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-geo-border text-geo-muted hover:bg-geo-accent hover:text-white flex-shrink-0 cursor-pointer transition-colors"
                          title="Jump to Import tab with this repo id pre-filled"
                        >
                          import
                        </button>
                      )}
                    </div>
                  );
                })}
                <div className="px-3 py-2 text-[10px] text-geo-muted italic">
                  Tip: once an FT head is cached, run inference from the OlmoEarth tab to turn it into a map layer.
                </div>
              </div>
            )}

            {/* === Import tab ===================================== */}
            {/* Reuses the shared OlmoEarthImport form so the picker + run
                + add-to-map flow stays identical between the map overlay
                and the sidebar's Import Data sub-view. `compact` strips
                the outer panel chrome since the popover already has its
                own container. */}
            {addedTab === "import" && (
              <div className="p-3">
                {datasets.length > 0 && (
                  <div className="mb-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-geo-muted mb-1">
                      Your uploads ({datasets.length})
                    </div>
                    <div className="max-h-[80px] overflow-y-auto divide-y divide-geo-border rounded border border-geo-border">
                      {datasets.slice(0, 20).map((d) => (
                        <div key={d.filename} className="flex items-center gap-2 px-2 py-1.5 text-[11px]">
                          <span className="font-mono text-geo-text truncate flex-1" title={d.filename}>
                            {d.filename}
                          </span>
                          <span className="text-[9px] text-geo-muted flex-shrink-0">
                            {d.format}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <OlmoEarthImport
                  compact
                  initialRepoId={importPrefill}
                  olmoCache={olmoCache}
                  selectedArea={selectedArea}
                  onAddImageryLayer={onAddImageryLayer}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function bboxToPolygon(bbox: BBox): GeoJSON.Polygon {
  return {
    type: "Polygon",
    coordinates: [[
      [bbox.west, bbox.north],
      [bbox.east, bbox.north],
      [bbox.east, bbox.south],
      [bbox.west, bbox.south],
      [bbox.west, bbox.north],
    ]],
  };
}

function updateSelectionShape(map: maplibregl.Map, polygon: GeoJSON.Polygon) {
  const geojson: GeoJSON.Feature = {
    type: "Feature",
    properties: {},
    geometry: polygon,
  };
  // Always tear down + rebuild so the layers sit at the top of the stack. In
  // practice, terra-draw's draw-time layers get added after ours and stay on
  // top even after `draw.clear()`; updating in place left the selection
  // invisible beneath empty terra-draw layers. Rebuilding each update is cheap
  // for a single feature and keeps z-order deterministic.
  clearSelectionShape(map);
  map.addSource("selection", { type: "geojson", data: geojson });
  map.addLayer({
    id: "selection-fill",
    type: "fill",
    source: "selection",
    paint: { "fill-color": "#2563eb", "fill-opacity": 0.2 },
  });
  map.addLayer({
    id: "selection-line",
    type: "line",
    source: "selection",
    paint: {
      "line-color": "#2563eb",
      "line-width": 3,
    },
  });
  // Terra-draw's MapLibre adapter defers its own layer writes to the next
  // animation frame. If we add our selection now, terra-draw's setMode
  // transitions (fired right after a polygon `finish`) land AFTER us and
  // push us down the stack — the polygon appears invisible until the user
  // tab-switches and re-mounts. Schedule a rAF that re-asserts top-of-stack
  // once terra-draw's async writes settle.
  const bringToTop = () => {
    try {
      if (map.getLayer("selection-fill")) map.moveLayer("selection-fill");
      if (map.getLayer("selection-line")) map.moveLayer("selection-line");
    } catch {
      /* map may have been torn down mid-frame */
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(bringToTop));
}

function clearSelectionShape(map: maplibregl.Map) {
  if (map.getLayer("selection-fill")) map.removeLayer("selection-fill");
  if (map.getLayer("selection-line")) map.removeLayer("selection-line");
  if (map.getSource("selection")) map.removeSource("selection");
}
