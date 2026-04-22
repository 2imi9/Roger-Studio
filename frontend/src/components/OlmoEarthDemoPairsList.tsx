/**
 * "Sample Data — OlmoEarth demo pairs" preset list for the Analysis tab.
 *
 * Mirrors the chrome of ``SplitMap.tsx``'s demo-pair cards (title, blurb,
 * warm/cold pill, spinner, tooltip copy) but does something different on
 * click: instead of adding both sides to the compare view, it drops A and
 * B as imagery layers on the regular map so the user can inspect a
 * single-scene inference output without entering compare mode first.
 *
 * Added 2026-04-21 after the product direction that the Analysis tab
 * should surface the curated demos as a one-click "try it now" surface
 * — replacing the older "Raster Results · Click to explain" header that
 * showed nothing until a user ran inference themselves.
 */
import { useEffect, useState } from "react";
import {
  getOlmoEarthDemoPairs,
  getLoadedOlmoEarthModels,
  prebakeOlmoEarthDemo,
  type OlmoEarthDemoPair,
} from "../api/client";
import type { ImageryLayer } from "./MapView";

interface Props {
  onAddImageryLayer?: (layer: ImageryLayer) => void;
  onRemoveImageryLayer?: (id: string) => void;
  imageryLayers?: ImageryLayer[];
}

type PairLoadState = { a: boolean; b: boolean; startedAt: number };

export function OlmoEarthDemoPairsList({
  onAddImageryLayer,
  onRemoveImageryLayer,
  imageryLayers,
}: Props) {
  const [pairs, setPairs] = useState<OlmoEarthDemoPair[]>([]);
  const [loadedHeads, setLoadedHeads] = useState<Set<string>>(new Set());
  const [loadingPairs, setLoadingPairs] = useState<Record<string, PairLoadState>>(
    {},
  );
  const [, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getOlmoEarthDemoPairs()
      .then((r) => {
        if (!cancelled) setPairs(r.pairs ?? []);
      })
      .catch((e) => console.warn("demo-pairs fetch failed:", e));
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshLoadedHeads = async () => {
    try {
      const r = await getLoadedOlmoEarthModels();
      setLoadedHeads(new Set(r.loaded));
    } catch (e) {
      console.warn("loaded-models fetch failed:", e);
    }
  };
  useEffect(() => {
    refreshLoadedHeads();
  }, []);

  // Keep elapsed-seconds counters ticking while any pair is loading.
  useEffect(() => {
    if (Object.keys(loadingPairs).length === 0) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [loadingPairs]);

  if (pairs.length === 0) return null;

  const handleLoad = async (pair: OlmoEarthDemoPair) => {
    if (!onAddImageryLayer) return;
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
    const alreadyLoaded =
      imageryLayers?.some((l) => l.id === aLayer.id) &&
      imageryLayers?.some((l) => l.id === bLayer.id);
    if (alreadyLoaded) return;

    setLoadingPairs((prev) => ({
      ...prev,
      [pair.id]: { a: true, b: true, startedAt: Date.now() },
    }));
    try {
      await prebakeOlmoEarthDemo(pair.id);
    } catch (e) {
      console.warn("prebake failed:", e);
    }
    // Attach immediately — tiles will fill in as inference completes on
    // the backend. Polling watches cache-control headers per side.
    onAddImageryLayer(aLayer);
    onAddImageryLayer(bLayer);

    const pollReady = async (jobId: string, side: "a" | "b") => {
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 5000));
        try {
          const r = await fetch(`/api/olmoearth/infer-tile/${jobId}/10/0/0.png`);
          const cc = r.headers.get("cache-control") ?? "";
          if (cc.includes("max-age")) {
            refreshLoadedHeads();
            setLoadingPairs((prev) => {
              const cur = prev[pair.id];
              if (!cur) return prev;
              const next = { ...cur, [side]: false };
              if (!next.a && !next.b) {
                const { [pair.id]: _done, ...rest } = prev;
                return rest;
              }
              return { ...prev, [pair.id]: next };
            });
            return;
          }
        } catch {
          /* keep polling */
        }
      }
      // Polling budget exhausted — clear regardless so the UI doesn't
      // show an infinite spinner.
      setLoadingPairs((prev) => {
        const cur = prev[pair.id];
        if (!cur) return prev;
        const next = { ...cur, [side]: false };
        if (!next.a && !next.b) {
          const { [pair.id]: _done, ...rest } = prev;
          return rest;
        }
        return { ...prev, [pair.id]: next };
      });
    };
    pollReady(pair.a.job_id, "a");
    pollReady(pair.b.job_id, "b");
  };

  const handleDrop = (pair: OlmoEarthDemoPair) => {
    if (!onRemoveImageryLayer) return;
    onRemoveImageryLayer(pair.a.id);
    onRemoveImageryLayer(pair.b.id);
  };

  return (
    <div className="space-y-2">
      <div className="text-[10px] text-geo-muted leading-snug">
        <span className="font-semibold text-geo-text">Click</span> to load A + B ·
        <span className="font-semibold text-geo-text"> double-click</span> to drop.
        Tiles are generated on-demand by the cached FT heads (Mangrove / AWF /
        Ecosystem). First click takes ~30 s while Sentinel-2 is fetched + the
        model runs; subsequent loads are instant from the on-disk cache.
      </div>
      <div className="space-y-2">
        {pairs.map((pair) => {
          const aOnMap = imageryLayers?.some((l) => l.id === pair.a.id);
          const bOnMap = imageryLayers?.some((l) => l.id === pair.b.id);
          const loaded = aOnMap && bOnMap;
          const loading = loadingPairs[pair.id];
          const elapsedS = loading
            ? Math.floor((Date.now() - loading.startedAt) / 1000)
            : 0;
          const headRepo = pair.a.spec.model_repo_id;
          const isWarm = loadedHeads.has(headRepo);
          const etaLabel = loading
            ? `loading · ${elapsedS}s`
            : isWarm
            ? "warm · ~3 s"
            : "cold · ~30 s";
          const etaClass = loading
            ? "bg-geo-accent/10 text-geo-accent border-geo-accent/40"
            : isWarm
            ? "bg-geo-success/15 text-geo-success border-geo-success/40"
            : "bg-geo-muted/15 text-geo-muted border-geo-border";
          return (
            <button
              key={pair.id}
              type="button"
              onClick={() => handleLoad(pair)}
              onDoubleClick={() => handleDrop(pair)}
              disabled={!!loading}
              className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                loading
                  ? "border-geo-accent/60 bg-geo-accent/10 cursor-wait"
                  : loaded
                  ? "border-geo-success/50 bg-geo-success/5 hover:bg-red-50 hover:border-red-300 cursor-pointer"
                  : "border-geo-border bg-geo-surface hover:border-geo-accent hover:bg-geo-bg hover:-translate-y-px hover:shadow-sm cursor-pointer"
              }`}
              title={
                loading
                  ? `Loading — ${elapsedS}s elapsed. First click per backend lifetime takes ~30 s.`
                  : loaded
                  ? "On map — double-click to drop both A and B"
                  : "Click to load A and B onto the map"
              }
            >
              <div className="flex items-start gap-2">
                <span
                  className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${
                    loaded ? "bg-geo-success" : loading ? "bg-geo-accent" : "bg-geo-border"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-0.5 flex-wrap">
                    <span className="font-semibold text-geo-text text-[12px] truncate">
                      {pair.title}
                    </span>
                    <span
                      className={`text-[9px] px-1.5 py-px rounded-full border font-medium flex-shrink-0 ${etaClass}`}
                    >
                      {etaLabel}
                    </span>
                  </div>
                  <div className="text-geo-muted text-[10px] leading-snug">
                    {pair.blurb}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
      <div className="text-[9px] text-geo-muted italic">
        Source: OlmoEarth FT heads + Sentinel-2 L2A (Microsoft Planetary Computer STAC).
      </div>
    </div>
  );
}
