import { useCallback, useEffect, useRef, useState } from "react";
import {
  Viewer,
  Cartesian3,
  Math as CesiumMath,
  Color,
  Rectangle,
  Cesium3DTileset,
  HeightReference,
  createWorldTerrainAsync,
  PolygonHierarchy,
  ArcType,
  Entity,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import type { BBox, AnalysisResult } from "../types";

interface CesiumViewProps {
  bbox: BBox;
  selectedGeometry?: GeoJSON.Polygon | null;
  analysisResult: AnalysisResult | null;
  overlayGeojson?: (GeoJSON.Feature | GeoJSON.FeatureCollection)[];
}

export function CesiumView({ bbox, selectedGeometry, analysisResult, overlayGeojson }: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [viewerReady, setViewerReady] = useState(false);

  // Init-time error surface. Previously terrain + 3D-tileset load
  // failures were silently swallowed (``.catch(() => {})``) — users
  // saw a blank globe with no explanation when the Ion token was
  // missing/invalid or the remote tileset was unreachable. Now we
  // capture into state and render a visible banner over the canvas so
  // the user can tell working-offline from broken-config.
  //
  // ``kind`` distinguishes the three independent failure modes:
  //   "viewer"   — ``new Viewer(...)`` itself threw (no WebGL, DOM issue)
  //   "terrain"  — ``createWorldTerrainAsync`` rejected (no token, API down)
  //   "tileset"  — ``Cesium3DTileset.fromIonAssetId`` rejected (asset gone,
  //                token lacks scope, offline)
  // Multiple can coexist; we keep the first one set so the banner stays
  // informative while subsequent failures only log.
  const [cesiumError, setCesiumError] = useState<
    { kind: "viewer" | "terrain" | "tileset"; msg: string } | null
  >(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let viewer: Viewer;
    try {
      viewer = new Viewer(containerRef.current, {
        animation: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: true,
        sceneModePicker: false,
        selectionIndicator: false,
        timeline: false,
        navigationHelpButton: false,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("Cesium Viewer init failed:", e);
      setCesiumError({ kind: "viewer", msg });
      return;
    }

    createWorldTerrainAsync({ requestWaterMask: true, requestVertexNormals: true })
      .then((tp) => { viewer.terrainProvider = tp; })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn("Cesium terrain load failed:", e);
        setCesiumError((cur) => cur ?? { kind: "terrain", msg });
      });

    Cesium3DTileset.fromIonAssetId(96188)
      .then((t) => { viewer.scene.primitives.add(t); })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn("Cesium Ion tileset load failed:", e);
        setCesiumError((cur) => cur ?? { kind: "tileset", msg });
      });

    viewer.scene.globe.enableLighting = true;
    // Keep depth test OFF so auto-label polygons render above terrain + buildings
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewerRef.current = viewer;
    setViewerReady(true);

    return () => { viewer.destroy(); viewerRef.current = null; setViewerReady(false); };
  }, []);

  // Fly to bbox + render analysis grid
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    viewer.entities.removeAll();

    // Selection boundary — prefer the actual drawn polygon when available,
    // fall back to a rectangle from bbox only when the user selected by drag.
    if (selectedGeometry) {
      const ring = selectedGeometry.coordinates[0] ?? [];
      if (ring.length >= 3) {
        const positions = Cartesian3.fromDegreesArray(
          ring.flatMap((pt) => [pt[0], pt[1]]),
        );
        viewer.entities.add({
          polygon: {
            hierarchy: new PolygonHierarchy(positions),
            material: Color.fromCssColorString("#2563eb").withAlpha(0.2),
            outline: true,
            outlineColor: Color.fromCssColorString("#2563eb"),
            outlineWidth: 2,
            heightReference: HeightReference.CLAMP_TO_GROUND,
          },
        });
        // Vertex dots for every ring point (skip the closing duplicate)
        const uniq = ring[0] === ring[ring.length - 1] ? ring.slice(0, -1) : ring;
        for (const [lng, lat] of uniq) {
          viewer.entities.add({
            position: Cartesian3.fromDegrees(lng, lat),
            point: {
              pixelSize: 7,
              color: Color.fromCssColorString("#2563eb"),
              outlineColor: Color.WHITE,
              outlineWidth: 2,
              heightReference: HeightReference.CLAMP_TO_GROUND,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
          });
        }
      }
    } else {
      viewer.entities.add({
        rectangle: {
          coordinates: Rectangle.fromDegrees(bbox.west, bbox.south, bbox.east, bbox.north),
          material: Color.fromCssColorString("#2563eb").withAlpha(0.1),
          outline: true,
          outlineColor: Color.fromCssColorString("#2563eb"),
          outlineWidth: 2,
          heightReference: HeightReference.CLAMP_TO_GROUND,
        },
      });

      for (const c of [
        { lng: bbox.west, lat: bbox.north, l: "NW" },
        { lng: bbox.east, lat: bbox.north, l: "NE" },
        { lng: bbox.west, lat: bbox.south, l: "SW" },
        { lng: bbox.east, lat: bbox.south, l: "SE" },
      ]) {
        viewer.entities.add({
          position: Cartesian3.fromDegrees(c.lng, c.lat),
          point: { pixelSize: 7, color: Color.fromCssColorString("#2563eb"), outlineColor: Color.WHITE, outlineWidth: 2, heightReference: HeightReference.CLAMP_TO_GROUND, disableDepthTestDistance: Number.POSITIVE_INFINITY },
          label: { text: c.l, font: "10px sans-serif", fillColor: Color.WHITE, outlineColor: Color.BLACK, outlineWidth: 1, style: 2, pixelOffset: { x: 10, y: -10 } as any, disableDepthTestDistance: Number.POSITIVE_INFINITY },
        });
      }
    }

    // Fly to area
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(
        (bbox.west + bbox.east) / 2,
        bbox.south - (bbox.north - bbox.south) * 0.3,
        Math.max((bbox.east - bbox.west) * 111000 * 1.5, 5000)
      ),
      orientation: { heading: CesiumMath.toRadians(0), pitch: CesiumMath.toRadians(-35), roll: 0 },
      duration: 2,
    });
  }, [bbox, selectedGeometry, analysisResult]);

  // Render GeoJSON overlay using Entity API directly (avoids Cesium rhumb line crash)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove previous overlay entities (tagged with _overlay)
    const toRemove = viewer.entities.values.filter((e: Entity) => (e as any)._overlay);
    for (const e of toRemove) viewer.entities.remove(e);

    if (!overlayGeojson || overlayGeojson.length === 0) return;

    // Collect all features
    const features: GeoJSON.Feature[] = [];
    for (const g of overlayGeojson) {
      if (g.type === "FeatureCollection") features.push(...g.features);
      else features.push(g);
    }

    const esc = (s: unknown) =>
      String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const buildDescription = (props: Record<string, any>): string => {
      const v = props.validation;
      const name = props.class_name || props.auto_class || props.name || "—";
      const conf = typeof props.confidence === "number" ? (props.confidence * 100).toFixed(1) + "%" : "—";
      let html = `<table style="width:100%;font-family:Inter,system-ui;font-size:13px;line-height:1.5;border-collapse:collapse">
        <tr><td style="color:#6b7280;padding:4px 8px">Class</td><td style="padding:4px 8px;font-weight:600">${esc(name)}</td></tr>
        <tr><td style="color:#6b7280;padding:4px 8px">Confidence</td><td style="padding:4px 8px;font-family:monospace">${esc(conf)}</td></tr>`;
      if (props.needs_review) {
        html += `<tr><td colspan="2" style="padding:4px 8px;color:#b45309;background:#fef3c7">⚠ flagged for review</td></tr>`;
      }
      if (v) {
        const actionColor: Record<string, string> = {
          accept: "#16a34a",
          reclassify: "#d97706",
          split: "#7c3aed",
          reject: "#dc2626",
        };
        const ac = actionColor[v.action] || "#6b7280";
        html += `</table>
          <div style="margin-top:12px;padding:10px 12px;background:#f4f3ee;border-radius:8px;border-left:3px solid ${ac}">
            <div style="font-size:11px;font-weight:600;color:${ac};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">Gemma 4 · ${esc(v.action)}</div>`;
        if (v.validated_class && v.validated_class !== v.original_class) {
          html += `<div style="font-size:13px;margin-bottom:6px"><b>${esc(v.original_class)}</b> → <b>${esc(v.validated_class)}</b></div>`;
        }
        if (v.reasoning) {
          html += `<div style="font-size:13px;color:#1f2937;margin-bottom:8px">${esc(v.reasoning)}</div>`;
        }
        if (Array.isArray(v.evidence_chain) && v.evidence_chain.length) {
          html += `<div style="font-size:11px;color:#6b7280">evidence: ${v.evidence_chain.map(esc).join(" · ")}</div>`;
        }
        html += `</div>`;
      } else {
        html += `</table>`;
      }
      return html;
    };

    for (const f of features) {
      try {
        const props = f.properties || {};
        const colorStr = props.color || props.auto_color || "#f59e0b";
        const c = Color.fromCssColorString(colorStr);
        const name = props.class_name || props.auto_class || props.name || "";
        const description = buildDescription(props);
        const gt = f.geometry?.type;

        if (gt === "Polygon") {
          const ring = f.geometry.coordinates[0];
          if (!ring || ring.length < 3) continue;
          const positions = Cartesian3.fromDegreesArray(ring.flat());
          const entity = viewer.entities.add({
            name,
            description,
            polygon: {
              hierarchy: new PolygonHierarchy(positions),
              material: c.withAlpha(0.55),
              outline: false,
              height: 200,
              arcType: ArcType.GEODESIC,
              classificationType: undefined,
            },
          });
          (entity as any)._overlay = true;

        } else if (gt === "MultiPolygon") {
          for (const poly of f.geometry.coordinates) {
            const ring = poly[0];
            if (!ring || ring.length < 3) continue;
            const positions = Cartesian3.fromDegreesArray(ring.flat());
            const entity = viewer.entities.add({
              name,
              description,
              polygon: {
                hierarchy: new PolygonHierarchy(positions),
                material: c.withAlpha(0.4),
                outline: false,
                height: 20,
                arcType: ArcType.GEODESIC,
              },
            });
            (entity as any)._overlay = true;
          }

        } else if (gt === "Point") {
          const [lng, lat] = f.geometry.coordinates;
          const entity = viewer.entities.add({
            name,
            description,
            position: Cartesian3.fromDegrees(lng, lat),
            point: {
              pixelSize: 8,
              color: c,
              outlineColor: Color.WHITE,
              outlineWidth: 1.5,
              heightReference: HeightReference.CLAMP_TO_GROUND,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
          });
          (entity as any)._overlay = true;

        } else if (gt === "LineString") {
          const coords = f.geometry.coordinates;
          const positions = Cartesian3.fromDegreesArray(coords.flat());
          const entity = viewer.entities.add({
            name,
            description,
            polyline: {
              positions,
              material: c,
              width: 3,
              arcType: ArcType.GEODESIC,
            },
          });
          (entity as any)._overlay = true;
        }
      } catch {
        // Skip invalid geometries silently
      }
    }
  }, [overlayGeojson, viewerReady]);

  // Navigation helpers
  const zoomIn = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.zoomIn(v.camera.positionCartographic.height * 0.3);
  }, []);
  const zoomOut = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.zoomOut(v.camera.positionCartographic.height * 0.3);
  }, []);
  const rotateLeft = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.rotateLeft(CesiumMath.toRadians(15));
  }, []);
  const rotateRight = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.rotateRight(CesiumMath.toRadians(15));
  }, []);
  const tiltUp = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.lookUp(CesiumMath.toRadians(5));
  }, []);
  const tiltDown = useCallback(() => {
    const v = viewerRef.current;
    if (v) v.camera.lookDown(CesiumMath.toRadians(5));
  }, []);
  const goHome = useCallback(() => {
    const v = viewerRef.current;
    if (v) {
      v.camera.flyTo({
        destination: Cartesian3.fromDegrees(
          (bbox.west + bbox.east) / 2,
          bbox.south - (bbox.north - bbox.south) * 0.3,
          Math.max((bbox.east - bbox.west) * 111000 * 1.5, 5000)
        ),
        orientation: { heading: CesiumMath.toRadians(0), pitch: CesiumMath.toRadians(-35), roll: 0 },
        duration: 1.5,
      });
    }
  }, [bbox]);

  const btnClass = "w-10 h-10 flex items-center justify-center bg-geo-surface hover:bg-geo-elevated text-geo-text border border-geo-border rounded-lg cursor-pointer text-base font-semibold shadow-sm transition-colors";

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="w-full h-full" />

      {/* Init-error banner. Rendered above the canvas (z-20, below the
          navigation toolbar at z-10 but above the globe) so the user
          sees WHY the 3D view is broken or degraded. Dismissible so a
          "terrain missing" notice doesn't block interaction with the
          fallback flat-earth view. */}
      {cesiumError && (
        <div
          className="absolute top-4 left-4 right-4 z-20 flex items-start gap-3 bg-amber-50 border border-amber-300 rounded-lg px-4 py-3 shadow-md max-w-2xl"
          role="alert"
          data-testid="cesium-error-banner"
        >
          <span className="text-lg leading-none flex-shrink-0" aria-hidden>⚠</span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-amber-900">
              {cesiumError.kind === "viewer"
                ? "3D globe failed to initialize"
                : cesiumError.kind === "terrain"
                  ? "World terrain unavailable — showing flat globe"
                  : "Photorealistic 3D tileset unavailable"}
            </div>
            <div className="mt-1 text-xs text-amber-800 leading-snug">
              {cesiumError.kind === "viewer" ? (
                <>
                  The Cesium viewer couldn&apos;t start. Most common cause:
                  WebGL is disabled, blocked by an extension, or the browser
                  doesn&apos;t support it. Details: <code className="font-mono">{cesiumError.msg}</code>.
                </>
              ) : (
                <>
                  Cesium Ion returned an error loading{" "}
                  {cesiumError.kind === "terrain"
                    ? "the world-terrain tileset"
                    : "the Google Photorealistic 3D Tiles asset"}
                  . Typically this means the bundled Ion token is expired or
                  rate-limited, or the machine is offline. The globe still
                  works — you&apos;re just seeing it without{" "}
                  {cesiumError.kind === "terrain" ? "elevation" : "buildings"}.
                  Details: <code className="font-mono">{cesiumError.msg}</code>.
                </>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={() => setCesiumError(null)}
            className="text-amber-800 hover:text-amber-900 text-lg leading-none flex-shrink-0"
            aria-label="Dismiss"
            title="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {/* Navigation toolbar — top right, below feature popups */}
      <div className="absolute top-24 right-6 flex flex-col gap-2 z-10 bg-gradient-panel backdrop-blur border border-geo-border rounded-xl p-3 shadow-md">
        <button onClick={goHome} className={btnClass} title="Reset view">H</button>
        <div className="h-2 border-t border-geo-border-subtle mx-1" />
        <button onClick={zoomIn} className={btnClass} title="Zoom in">+</button>
        <button onClick={zoomOut} className={btnClass} title="Zoom out">&minus;</button>
        <div className="h-2 border-t border-geo-border-subtle mx-1" />
        <button onClick={rotateLeft} className={btnClass} title="Rotate left">&larr;</button>
        <button onClick={rotateRight} className={btnClass} title="Rotate right">&rarr;</button>
        <div className="h-2 border-t border-geo-border-subtle mx-1" />
        <button onClick={tiltUp} className={btnClass} title="Tilt up">&uarr;</button>
        <button onClick={tiltDown} className={btnClass} title="Tilt down">&darr;</button>
      </div>

      {/* Help hint — bottom left, raised above Cesium attribution */}
      <div className="absolute bottom-16 left-6 bg-gradient-panel backdrop-blur text-geo-muted px-5 py-3 rounded-xl text-xs z-10 border border-geo-border shadow-md tracking-wide">
        <span className="font-semibold text-geo-text">Scroll</span> zoom
        <span className="mx-4 text-geo-dim">·</span>
        <span className="font-semibold text-geo-text">Drag</span> rotate
        <span className="mx-4 text-geo-dim">·</span>
        <span className="font-semibold text-geo-text">Ctrl+Drag</span> tilt
      </div>
    </div>
  );
}
