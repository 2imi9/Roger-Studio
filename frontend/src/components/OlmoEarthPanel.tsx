import { useEffect, useRef, useState } from "react";
import type { BBox } from "../types";
import {
  getOlmoEarthCatalog,
  getOlmoEarthCacheStatus,
  loadOlmoEarthRepo,
  unloadOlmoEarthRepo,
  startOlmoEarthInference,
  type OlmoEarthCatalog,
  type OlmoEarthModel,
  type OlmoEarthDatasetLive,
  type OlmoEarthRepoStatus,
} from "../api/client";
import { Panel, SectionTitle } from "./ui/Panel";
import type { ImageryLayer } from "./MapView";

interface OlmoEarthPanelProps {
  selectedArea: BBox | null;
  onAddImageryLayer?: (l: ImageryLayer) => void;
}

type SubTab = "encoder" | "ft" | "dataset";
const SS_OLMO_SUBTAB = "roger.olmo.subtab.v2";
const SUBTAB_LABEL: Record<SubTab, string> = {
  encoder: "Encoder",
  ft: "Finetune head",
  dataset: "Dataset",
};

// Repos where clicking Load without thinking would hurt. The map value is
// the warning text shown in the confirm dialog — be specific about size and
// intended audience so the user can make an informed call.
const HEAVY_REPOS: Record<string, string> = {
  "allenai/olmoearth_pretrain_dataset":
    "This is the pretraining corpus — ~hundreds of GB of Sentinel-1/2 image " +
    "chips. It's meant for researchers training foundation models from " +
    "scratch, not for Roger Studio runtime. The projects datasets " +
    "(mangrove, AWF) are the fine-tuning eval sets and are tiny — " +
    "load those instead if you want to inspect OlmoEarth training data.",
};

function formatNumber(n: number | undefined): string {
  if (n == null) return "–";
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatBytes(n: number | undefined): string {
  if (n == null) return "";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)} GB`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)} MB`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)} KB`;
  return `${n} B`;
}

interface LoadPortProps {
  status?: OlmoEarthRepoStatus;
  onLoad: () => void;
  onUnload?: () => void;
}

function LoadPort({ status, onLoad, onUnload }: LoadPortProps) {
  const state = status?.status;
  if (state === "cached") {
    return (
      <div className="flex items-center gap-1 shrink-0">
        <span
          className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-accent-soft text-geo-accent whitespace-nowrap"
          title={status?.path}
        >
          Cached{status?.size_bytes ? ` · ${formatBytes(status.size_bytes)}` : ""}
        </span>
        {onUnload && (
          <button
            onClick={onUnload}
            aria-label="Remove from cache"
            title="Remove from cache"
            className="w-5 h-5 flex items-center justify-center rounded-full bg-geo-elevated hover:bg-red-500 hover:text-white text-geo-muted cursor-pointer text-xs font-bold transition-colors"
          >
            ×
          </button>
        )}
      </div>
    );
  }
  if (state === "loading") {
    const elapsed = status?.started_ts
      ? Math.max(0, Math.round(Date.now() / 1000 - status.started_ts))
      : 0;
    return (
      <span className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-warn-soft text-geo-warn whitespace-nowrap">
        Loading… {elapsed}s
      </span>
    );
  }
  if (state === "error") {
    return (
      <button
        onClick={onLoad}
        className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-red-50 text-red-700 hover:bg-red-100 cursor-pointer whitespace-nowrap"
        title={status?.error ?? ""}
      >
        Retry
      </button>
    );
  }
  return (
    <button
      onClick={onLoad}
      className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-accent text-white hover:opacity-90 cursor-pointer whitespace-nowrap"
    >
      Load
    </button>
  );
}

function timeAgo(iso: string | undefined): string {
  if (!iso) return "unknown";
  const ms = Date.now() - new Date(iso).getTime();
  const d = Math.floor(ms / 86_400_000);
  if (d > 30) return `${Math.floor(d / 30)} mo ago`;
  if (d >= 1) return `${d} d ago`;
  const h = Math.floor(ms / 3_600_000);
  if (h >= 1) return `${h} h ago`;
  const m = Math.floor(ms / 60_000);
  return m >= 1 ? `${m} m ago` : "just now";
}

function ModelRow({
  model,
  status,
  onLoad,
  onUnload,
  onRun,
  canRun,
  showPort,
}: {
  model: OlmoEarthModel;
  status?: OlmoEarthRepoStatus;
  onLoad: () => void;
  onUnload: () => void;
  onRun?: () => void;
  canRun: boolean;
  showPort: boolean;
}) {
  const isCached = status?.status === "cached";
  // The Run button only makes sense for FT heads — encoders produce
  // embeddings not directly viewable without a post-hoc head.
  const showRun = showPort && isCached && model.type === "fine-tuned" && onRun;
  return (
    <div className="flex items-start gap-2 py-2 border-b border-geo-border last:border-0">
      <div className="flex-1 min-w-0">
        <a
          href={`https://huggingface.co/${model.repo_id}`}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-xs text-geo-accent hover:underline break-all"
        >
          {model.repo_id.replace(/^allenai\//, "")}
        </a>
        {model.task && <p className="text-[11px] text-geo-muted mt-0.5">{model.task}</p>}
        {model.size_tier && <p className="text-[11px] text-geo-muted mt-0.5">size: {model.size_tier}</p>}
        {showPort && status?.error && (
          <p className="text-[10px] mt-1" style={{ color: "#dc2626" }}>{status.error.slice(0, 120)}</p>
        )}
      </div>
      <div className="text-right text-[10px] text-geo-muted font-mono shrink-0">
        <div>↓ {formatNumber(model.downloads)}</div>
        <div>♥ {formatNumber(model.likes)}</div>
      </div>
      {showRun && (
        <button
          onClick={onRun}
          disabled={!canRun}
          title={canRun ? "Run on current bbox and add as map layer" : "Draw an area on the Map tab first"}
          className={`text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded whitespace-nowrap cursor-pointer transition-colors ${
            canRun
              ? "bg-geo-accent text-white hover:opacity-90"
              : "bg-geo-elevated text-geo-muted cursor-not-allowed"
          }`}
        >
          Run
        </button>
      )}
      {showPort && <LoadPort status={status} onLoad={onLoad} onUnload={onUnload} />}
    </div>
  );
}

function DatasetRow({
  dataset,
  status,
  onLoad,
  onUnload,
  showPort,
}: {
  dataset: OlmoEarthDatasetLive;
  status?: OlmoEarthRepoStatus;
  onLoad: () => void;
  onUnload: () => void;
  showPort: boolean;
}) {
  return (
    <div className="flex items-start gap-2 py-2 border-b border-geo-border last:border-0">
      <div className="flex-1 min-w-0">
        <a
          href={`https://huggingface.co/datasets/${dataset.repo_id}`}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-xs text-geo-accent hover:underline break-all"
        >
          {dataset.repo_id.replace(/^allenai\//, "")}
        </a>
        {dataset.task && <p className="text-[11px] text-geo-muted mt-0.5">{dataset.task}</p>}
        {dataset.coverage && <p className="text-[11px] text-geo-muted mt-0.5">coverage: {dataset.coverage}</p>}
        <div className="flex gap-3 text-[10px] text-geo-muted font-mono mt-1">
          <span>↓ {formatNumber(dataset.downloads)}</span>
          <span>♥ {formatNumber(dataset.likes)}</span>
          {dataset.license && <span>{dataset.license}</span>}
        </div>
        {showPort && status?.error && (
          <p className="text-[10px] mt-1" style={{ color: "#dc2626" }}>{status.error.slice(0, 120)}</p>
        )}
      </div>
      {showPort && <LoadPort status={status} onLoad={onLoad} onUnload={onUnload} />}
    </div>
  );
}

export function OlmoEarthPanel({ selectedArea, onAddImageryLayer }: OlmoEarthPanelProps) {
  const [catalog, setCatalog] = useState<OlmoEarthCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetch, setLastFetch] = useState<number | null>(null);
  const [cacheStatus, setCacheStatus] = useState<Record<string, OlmoEarthRepoStatus>>({});
  const pollRef = useRef<number | null>(null);
  const [sub, setSub] = useState<SubTab>(() => {
    try {
      const raw = sessionStorage.getItem(SS_OLMO_SUBTAB);
      if (raw === "encoder" || raw === "ft" || raw === "dataset") return raw;
    } catch {
      /* noop */
    }
    return "encoder";
  });
  useEffect(() => {
    try { sessionStorage.setItem(SS_OLMO_SUBTAB, sub); } catch { /* noop */ }
  }, [sub]);

  // Listen for the Sidebar OlmoEarth-tab dropdown picking a subtab. The
  // Sidebar dispatches `roger:olmo-subtab-change` with the new value as
  // event.detail; we accept the same union and update local state. Avoids
  // having to lift the subtab state up through App.tsx.
  useEffect(() => {
    const onChange = (e: Event) => {
      const next = (e as CustomEvent).detail;
      if (next === "encoder" || next === "ft" || next === "dataset") {
        setSub(next);
      }
    };
    window.addEventListener("roger:olmo-subtab-change", onChange);
    return () => window.removeEventListener("roger:olmo-subtab-change", onChange);
  }, []);

  const load = async (force = false) => {
    setLoading(true);
    setError(null);
    try {
      const c = await getOlmoEarthCatalog({ bbox: selectedArea ?? undefined, force });
      setCatalog(c);
      setLastFetch(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed");
    } finally {
      setLoading(false);
    }
  };

  const refreshCacheStatus = async () => {
    try {
      const s = await getOlmoEarthCacheStatus();
      setCacheStatus(s.repos);
    } catch {
      /* transient — ignored */
    }
  };

  const startLoadRepo = async (repoId: string, repoType: "model" | "dataset") => {
    // Heavy-repo guardrail. The pretraining corpus is hundreds of GB of
    // Sentinel shards intended for training foundation models from scratch,
    // not for runtime use here. Because snapshot_download runs in an OS
    // thread we can't cleanly interrupt, accidentally starting one and then
    // displacing it leaves an orphan thread writing shards forever. Force
    // an explicit confirm so nobody clicks into that trap by accident.
    if (HEAVY_REPOS[repoId] && !window.confirm(
      `⚠️ ${repoId}\n\n${HEAVY_REPOS[repoId]}\n\nProceed with download?`,
    )) {
      return;
    }
    // Optimistically flip to loading so the UI reacts immediately — the
    // server's authoritative status overwrites on the next poll.
    setCacheStatus((prev) => ({
      ...prev,
      [repoId]: { ...prev[repoId], status: "loading", started_ts: Date.now() / 1000 },
    }));
    try {
      await loadOlmoEarthRepo({ repoId, repoType });
      refreshCacheStatus();
    } catch (e) {
      setCacheStatus((prev) => ({
        ...prev,
        [repoId]: { status: "error", error: e instanceof Error ? e.message : "failed" },
      }));
    }
  };

  const runInference = async (model: OlmoEarthModel) => {
    if (!selectedArea || !onAddImageryLayer) return;
    try {
      const r = await startOlmoEarthInference({
        bbox: selectedArea,
        modelRepoId: model.repo_id,
      });
      onAddImageryLayer({
        id: `infer-${r.job_id}`,
        // Backend hands back a relative path; let MapLibre resolve it
        // against the same origin the frontend is served from.
        tileUrl: `${window.location.origin}${r.tile_url}`,
        label: `${model.repo_id.replace(/^allenai\//, "")} · ${r.task_type ?? r.kind}`,
        opacity: 0.8,
        inferenceMetadata: r,
      });
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(`Inference failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const unloadRepo = async (repoId: string) => {
    // Optimistic evict — remove from local state so the row flips back to
    // a Load button immediately; server confirms on next poll.
    setCacheStatus((prev) => {
      const next = { ...prev };
      delete next[repoId];
      return next;
    });
    try {
      await unloadOlmoEarthRepo(repoId);
    } finally {
      refreshCacheStatus();
    }
  };

  // Refetch on mount + whenever the selected area changes (project_coverage
  // and recommended_model both depend on the bbox).
  useEffect(() => {
    load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedArea?.west, selectedArea?.south, selectedArea?.east, selectedArea?.north]);

  // Poll cache status every 2s so "Loading…" → "Cached" transitions appear
  // without user action. Runs forever while the panel is mounted.
  useEffect(() => {
    refreshCacheStatus();
    pollRef.current = window.setInterval(refreshCacheStatus, 2000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const encoders = (catalog?.models ?? []).filter((m) => m.type === "encoder");
  const ftHeads = (catalog?.models ?? []).filter((m) => m.type === "fine-tuned");
  const other = (catalog?.models ?? []).filter((m) => m.type !== "encoder" && m.type !== "fine-tuned");

  const connected = !!catalog && !error;
  // With the by-type subtab IA there's no "browse-only" mode — Load + Run
  // pills are always visible so the tab is actionable on every row.
  const showPorts = true;

  const cachedRepos = Object.entries(cacheStatus).filter(([, s]) => s.status === "cached");
  const loadingRepos = Object.entries(cacheStatus).filter(([, s]) => s.status === "loading");
  const totalCachedBytes = cachedRepos.reduce(
    (acc, [, s]) => acc + (s.size_bytes ?? 0),
    0,
  );

  return (
    <div className="space-y-8">
      {/* Header + connection chip */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-[22px] font-bold text-geo-text">OlmoEarth</h2>
          <span
            className={`text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded ${
              connected
                ? "bg-geo-accent-soft text-geo-accent"
                : "bg-geo-warn-soft text-geo-warn"
            }`}
            title="HuggingFace Hub read access"
          >
            {loading ? "Syncing…" : connected ? "Connected" : "Offline"}
          </span>
        </div>
        <p className="text-xs text-geo-muted">
          Live from <span className="font-mono">huggingface.co/allenai</span>, cached 10 min.
        </p>
        <div className="mt-2 flex items-center gap-3 text-xs text-geo-muted">
          <span>
            {lastFetch ? `Refreshed ${Math.round((Date.now() - lastFetch) / 1000)}s ago` : "Not yet fetched"}
          </span>
          <button
            onClick={() => load(true)}
            disabled={loading}
            className={`px-2 py-0.5 rounded text-[11px] font-semibold cursor-pointer transition-colors ${
              loading ? "bg-geo-elevated text-geo-muted" : "bg-geo-accent text-white hover:opacity-90"
            }`}
          >
            {loading ? "…" : "Refresh"}
          </button>
        </div>
        {error && <p className="mt-2 text-xs" style={{ color: "#dc2626" }}>{error}</p>}
      </div>

      {/* Subtab switcher — by TYPE, not by view mode */}
      <div className="flex gap-1 p-1 bg-geo-elevated rounded-lg">
        {(Object.keys(SUBTAB_LABEL) as SubTab[]).map((key) => (
          <button
            key={key}
            onClick={() => setSub(key)}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-colors cursor-pointer ${
              sub === key
                ? "bg-geo-bg text-geo-accent shadow-sm"
                : "text-geo-muted hover:text-geo-text"
            }`}
          >
            {SUBTAB_LABEL[key]}
          </button>
        ))}
      </div>

      {/* Cache summary — shared across subtabs, makes disk usage always visible */}
      <Panel border>
        <div className="flex items-center justify-between text-xs">
          <div>
            <p className="font-semibold text-geo-text">
              {cachedRepos.length} cached
              {loadingRepos.length > 0 && `, ${loadingRepos.length} loading`}
            </p>
            <p className="text-[11px] text-geo-muted mt-0.5">
              {formatBytes(totalCachedBytes)} on disk · {`~/.cache/huggingface/`}
            </p>
          </div>
        </div>
      </Panel>

      {/* Project coverage + recommended model — shown on the Finetune head
          tab (FT heads are what you'd run for a specific area) and on the
          Dataset tab (datasets have spatial coverage metadata). The Encoder
          tab skips it — encoders are area-agnostic. */}
      {(sub === "ft" || sub === "dataset") && (
        <div>
          <SectionTitle>Coverage for this area</SectionTitle>
          {!selectedArea && (
            <p className="text-xs text-geo-muted">
              Draw an area on the Map tab to see which OlmoEarth projects overlap it.
            </p>
          )}
          {selectedArea && catalog && catalog.project_coverage.length === 0 && (
            <p className="text-xs text-geo-muted">
              This bbox does not overlap any OlmoEarth project-labelled regions (mangrove
              tropics, southern Kenya).
            </p>
          )}
          {catalog && catalog.project_coverage.length > 0 && (
            <div className="space-y-2">
              {catalog.project_coverage.map((c) => (
                <div key={c.repo_id} className="flex items-start gap-2 text-sm">
                  <span className="text-geo-accent shrink-0">·</span>
                  <span className="flex-1">
                    <a
                      href={`https://huggingface.co/datasets/${c.repo_id}`}
                      target="_blank"
                      rel="noreferrer"
                      className="font-mono text-xs text-geo-accent hover:underline break-all"
                    >
                      {c.repo_id.replace(/^allenai\//, "")}
                    </a>
                    {c.dataset?.task && (
                      <span className="text-geo-muted text-xs"> — {c.dataset.task}</span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
          {catalog?.recommended_model && (
            <Panel border className="mt-3">
              <p className="text-[11px] uppercase tracking-wider text-geo-muted mb-1">
                Recommended model
              </p>
              <a
                href={`https://huggingface.co/${catalog.recommended_model.repo_id}`}
                target="_blank"
                rel="noreferrer"
                className="font-mono text-xs text-geo-accent hover:underline break-all"
              >
                {catalog.recommended_model.repo_id}
              </a>
              {catalog.recommended_model.reason && (
                <p className="mt-1 text-[11px] text-geo-muted leading-relaxed">
                  {catalog.recommended_model.reason}
                </p>
              )}
            </Panel>
          )}
        </div>
      )}

      {/* Encoders */}
      {sub === "encoder" && encoders.length > 0 && (
        <div>
          <SectionTitle>Encoders ({encoders.length})</SectionTitle>
          <Panel border>
            {encoders.map((m) => (
              <ModelRow
                key={m.repo_id}
                model={m}
                status={cacheStatus[m.repo_id]}
                onLoad={() => startLoadRepo(m.repo_id, "model")}
                onUnload={() => unloadRepo(m.repo_id)}
                onRun={() => runInference(m)}
                canRun={!!selectedArea && !!onAddImageryLayer}
                showPort={showPorts}
              />
            ))}
          </Panel>
        </div>
      )}

      {/* Fine-tuned heads */}
      {sub === "ft" && ftHeads.length > 0 && (
        <div>
          <SectionTitle>Fine-tuned heads ({ftHeads.length})</SectionTitle>
          <Panel border>
            {ftHeads.map((m) => (
              <ModelRow
                key={m.repo_id}
                model={m}
                status={cacheStatus[m.repo_id]}
                onLoad={() => startLoadRepo(m.repo_id, "model")}
                onUnload={() => unloadRepo(m.repo_id)}
                onRun={() => runInference(m)}
                canRun={!!selectedArea && !!onAddImageryLayer}
                showPort={showPorts}
              />
            ))}
          </Panel>
        </div>
      )}

      {/* Unknown / future variants — shown on the Encoder tab as a catch-all
          since most forthcoming OlmoEarth variants are likely backbone tweaks. */}
      {sub === "encoder" && other.length > 0 && (
        <div>
          <SectionTitle>Other variants ({other.length})</SectionTitle>
          <Panel border>
            {other.map((m) => (
              <ModelRow
                key={m.repo_id}
                model={m}
                status={cacheStatus[m.repo_id]}
                onLoad={() => startLoadRepo(m.repo_id, "model")}
                onUnload={() => unloadRepo(m.repo_id)}
                onRun={() => runInference(m)}
                canRun={!!selectedArea && !!onAddImageryLayer}
                showPort={showPorts}
              />
            ))}
          </Panel>
        </div>
      )}

      {/* Datasets */}
      {sub === "dataset" && catalog && catalog.datasets.length > 0 && (
        <div>
          <SectionTitle>Datasets ({catalog.datasets.length})</SectionTitle>
          <Panel border>
            {catalog.datasets.map((d) => (
              <DatasetRow
                key={d.repo_id}
                dataset={d}
                status={cacheStatus[d.repo_id]}
                onLoad={() => startLoadRepo(d.repo_id, "dataset")}
                onUnload={() => unloadRepo(d.repo_id)}
                showPort={showPorts}
              />
            ))}
          </Panel>
        </div>
      )}

      {catalog && (
        <div>
          <p className="text-[10px] text-geo-muted italic leading-relaxed">
            {catalog.notes?.join(" ")}
          </p>
        </div>
      )}
    </div>
  );
}
