import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  gemmaChat,
  gemmaHealth,
  gemmaStart,
  gemmaStop,
  gemmaLogs,
  gemmaSetModel,
} from "../api/client";
import type { ChatMessage, LLMHealth } from "../api/client";
import { useCancellableSend, wasAborted } from "../hooks/useCancellableSend";
import type { AreaSelection, DatasetInfo } from "../types";
import { LLMSettings, MODEL_PRESETS, CUSTOM_MODEL } from "./LLMSettings";
import { MarkdownText } from "./ui/MarkdownText";
import { safeSetItem } from "../util/sessionStorage";

interface GemmaChatProps {
  selectedArea: AreaSelection | null;
  datasets: DatasetInfo[];
  autoLabelSummary?: Record<string, unknown> | null;
  subView?: "chat" | "settings";
  onSubViewChange?: (v: "chat" | "settings") => void;
  /** When provided, the messages scroll + composer DOM re-parents into
   * this target via React portal. Config header (API key, model picker,
   * status) stays inline in the sidebar. Used by the floating-chat
   * overlay so the chat body can grow wider than the 480 px sidebar. */
  chatBodyPortalTarget?: HTMLElement | null;
}

const SUGGESTIONS = [
  "What land-cover classes should I expect in this area?",
  "Why did TIPSv2 label this polygon as 'forest' at low confidence?",
  "What seasonal variation should I watch for?",
  "Which bands of Landsat are most useful for agricultural detection here?",
];

// MODEL_PRESETS and CUSTOM_MODEL are imported from ./LLMSettings.

const SS_SELECTED_MODEL = "geoenv.llm.selectedModel";
const SS_CUSTOM_MODEL = "geoenv.llm.customModel";

// sessionStorage keys — scoped to a browser tab, cleared when the tab closes.
// We use session (not local) storage for privacy on shared machines.
const SS_MESSAGES = "geoenv.llm.messages";
const SS_INPUT = "geoenv.llm.input";
const SS_HF_TOKEN = "geoenv.llm.hfToken";
const SS_USE_UNGATED = "geoenv.llm.useUngatedMirror";

function readSession<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function GemmaChat({
  selectedArea,
  datasets,
  autoLabelSummary,
  subView = "chat",
  onSubViewChange,
  chatBodyPortalTarget,
}: GemmaChatProps) {
  // Persist chat state in sessionStorage so tab switches + full page reloads
  // don't wipe the conversation. sessionStorage is per-browser-tab and clears
  // when the tab closes — never written to disk, never sent to the backend.
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    readSession<ChatMessage[]>(SS_MESSAGES, [])
  );
  const [input, setInput] = useState<string>(() => readSession(SS_INPUT, ""));
  const { sending, begin, abort, finish } = useCancellableSend();
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<LLMHealth | null>(null);
  // HF token + model toggle for the vLLM docker-command generator. Kept
  // client-side only: never sent to backend, never persisted to disk.
  const [hfToken, setHfToken] = useState<string>(() => readSession(SS_HF_TOKEN, ""));
  const [useUngatedMirror, setUseUngatedMirror] = useState<boolean>(() =>
    readSession(SS_USE_UNGATED, false)
  );
  const [selectedModel, setSelectedModel] = useState<string>(() =>
    readSession(SS_SELECTED_MODEL, "")
  );
  const [customModel, setCustomModel] = useState<string>(() =>
    readSession(SS_CUSTOM_MODEL, "")
  );
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Persist on every change. Messages go through safeSetItem with
  // trackForEviction so another chat panel hitting the quota wall can
  // reclaim bytes from stale histories instead of silently dropping writes.
  useEffect(() => {
    safeSetItem(SS_MESSAGES, JSON.stringify(messages), { trackForEviction: true });
  }, [messages]);
  useEffect(() => {
    safeSetItem(SS_INPUT, JSON.stringify(input));
  }, [input]);
  useEffect(() => {
    safeSetItem(SS_HF_TOKEN, JSON.stringify(hfToken));
  }, [hfToken]);
  useEffect(() => {
    safeSetItem(SS_USE_UNGATED, JSON.stringify(useUngatedMirror));
  }, [useUngatedMirror]);
  useEffect(() => {
    safeSetItem(SS_SELECTED_MODEL, JSON.stringify(selectedModel));
  }, [selectedModel]);
  useEffect(() => {
    safeSetItem(SS_CUSTOM_MODEL, JSON.stringify(customModel));
  }, [customModel]);

  // Resolved model = what we actually generate the docker command for and
  // tell the backend to use. Picker value wins; otherwise fall back to the
  // ungated toggle behavior; otherwise health.model from backend.
  const resolvedModel = useMemo<string>(() => {
    if (selectedModel === CUSTOM_MODEL && customModel.trim()) return customModel.trim();
    if (selectedModel && selectedModel !== CUSTOM_MODEL) return selectedModel;
    // Fallback: legacy ungated toggle behavior
    const fallback = health?.model || "unsloth/gemma-4-e4b-it";
    return useUngatedMirror ? fallback.replace(/^google\//, "unsloth/") : fallback;
  }, [selectedModel, customModel, useUngatedMirror, health?.model]);

  // When the picker or custom model changes, push to backend so /chat and
  // /validate call the right model on the running vLLM/Ollama endpoint.
  useEffect(() => {
    if (!resolvedModel || resolvedModel === health?.model) return;
    gemmaSetModel(resolvedModel).catch(() => { /* backend may be mid-reload */ });
  }, [resolvedModel, health?.model]);
  const [starting, setStarting] = useState(false);
  const [logTail, setLogTail] = useState<string>("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshHealth = async () => {
    try {
      const h = await gemmaHealth();
      setHealth(h);
      return h;
    } catch {
      return null;
    }
  };

  useEffect(() => {
    refreshHealth();
  }, []);

  // Poll while we're starting the local runtime until it becomes reachable.
  // Also exit the starting state if the model is pulled/running but the API
  // simply isn't reachable (port blocked, service not up, etc.) — in that
  // case stop showing the blue "starting" banner so the user sees the real
  // error state.
  useEffect(() => {
    if (!starting) return;
    const startedAt = Date.now();
    const timer = setInterval(async () => {
      const h = await refreshHealth();
      if (h?.reachable) {
        setStarting(false);
        setLogTail("");
        return;
      }
      // Stop the starting banner once the pull is definitively done
      // (status !== "missing"). If the API is still unreachable, main UI will
      // show the offline state with a clear error.
      const done = h?.container_status === "running" || h?.container_status === "pulled";
      const elapsedMin = (Date.now() - startedAt) / 60000;
      if (done || elapsedMin > 4) {
        setStarting(false);
        setLogTail("");
        return;
      }
      try {
        const l = await gemmaLogs(20);
        setLogTail(l.logs.slice(-1500));
      } catch {
        /* noop */
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [starting]);

  const handleStart = async () => {
    setError(null);
    try {
      // Pass the client-side HF token through so gated models work. Token
      // lives in this browser tab only; backend uses it for the docker -e
      // HF_TOKEN flag but doesn't persist it.
      const r = await gemmaStart(hfToken || undefined);
      setStarting(true);
      if (r.note) setLogTail(r.note);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Start failed");
    }
  };

  const handleStop = async () => {
    setError(null);
    try {
      await gemmaStop();
      await refreshHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stop failed");
    }
  };

  // Clear the chat conversation (messages + draft). Keeps HF token and model
  // selection so the user doesn't have to re-enter them.
  const handleClearChat = () => {
    if (messages.length === 0 && !input) return;
    if (!window.confirm("Clear the entire conversation? This can't be undone.")) return;
    setMessages([]);
    setInput("");
    setError(null);
    try {
      sessionStorage.removeItem(SS_MESSAGES);
      sessionStorage.removeItem(SS_INPUT);
    } catch {
      /* noop */
    }
  };

  // Per-message copy: track which index was last copied, flash "Copied" for 1.5s
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const handleCopyMessage = async (idx: number, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx((cur) => (cur === idx ? null : cur)), 1500);
    } catch {
      setError("Clipboard write failed. Select + Ctrl+C the message manually.");
    }
  };

  // Build the vLLM docker-run command with the user's HF token baked in (if any).
  // Token stays in React state only; never hits the backend or disk.
  //
  // Default image is geoenv-vllm:latest — our local custom build that layers
  // pandas + transformers>=5.5.0 + modelopt on top of vllm/vllm-openai:cu130-nightly
  // (needed because the stock nightly image is missing pandas and the :latest
  // tag's Transformers is too old for the gemma4 architecture).
  //
  // Port mapping: container's internal 8000 → host 8001, because our FastAPI
  // backend already owns host-8000. Mismatching these causes a rogue-proxy
  // hijack that breaks /api/analyze with mysterious 500s and empty replies.
  const vllmDockerCommand = useMemo(() => {
    const tokenLine = hfToken.trim() ? `  -e HF_TOKEN=${hfToken.trim()} \\\n` : "";
    return (
      `docker run --rm --gpus all --ipc=host \\\n` +
      `  -v $HOME/.cache/huggingface:/root/.cache/huggingface \\\n` +
      `  -v $HOME/.cache/vllm:/root/.cache/vllm \\\n` +
      tokenLine +
      `  -p 8001:8000 geoenv-vllm:latest \\\n` +
      `  --model ${resolvedModel} \\\n` +
      `  --max-model-len 16384 \\\n` +
      `  --gpu-memory-utilization 0.75 \\\n` +
      `  --enable-auto-tool-choice \\\n` +
      `  --tool-call-parser gemma4 --reasoning-parser gemma4`
    );
  }, [hfToken, resolvedModel]);

  const handleCopyCommand = async () => {
    try {
      await navigator.clipboard.writeText(vllmDockerCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Clipboard write failed. Select + Ctrl+C the command manually.");
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, sending]);

  const sceneContext = () => {
    const ctx: Record<string, unknown> = {};
    if (selectedArea) {
      if ("west" in selectedArea) {
        ctx.area = `bbox W${selectedArea.west.toFixed(3)} S${selectedArea.south.toFixed(3)} E${selectedArea.east.toFixed(3)} N${selectedArea.north.toFixed(3)}`;
        // Parseable bbox object — the tool calls (run_olmoearth_inference,
        // query_ndvi_timeseries, etc.) expect a {west,south,east,north} dict,
        // so surfacing it in scene_context saves the LLM from re-parsing the
        // human string + prevents "I need the bbox coordinates" confusion.
        ctx.area_bbox = {
          west:  selectedArea.west,
          south: selectedArea.south,
          east:  selectedArea.east,
          north: selectedArea.north,
        };
      } else {
        ctx.area = `polygon (${selectedArea.coordinates.length} vertices)`;
      }
    }
    if (datasets.length) {
      ctx.datasets = datasets.map((d) => `${d.filename} (${d.format})`);
    }
    if (autoLabelSummary) {
      ctx.auto_label_summary = autoLabelSummary;
    }
    return ctx;
  };

  const send = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || sending) return;
    const userMsg: ChatMessage = { role: "user", content };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");
    setError(null);
    const signal = begin();
    try {
      const reply = await gemmaChat(next, sceneContext(), { signal });
      setMessages([...next, {
        role: "assistant",
        content: reply.content,
        reasoning_content: reply.reasoning_content,
        phantom_tool_call_fixed: reply.phantom_tool_call_fixed,
        empty_retry_fixed: reply.empty_retry_fixed,
        stopped_reason: reply.stopped_reason,
        artifacts: reply.artifacts,
      }]);
    } catch (e) {
      if (wasAborted(e)) return;   // user clicked Stop — not an error
      setError(e instanceof Error ? e.message : "Chat failed");
    } finally {
      finish();
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <div>
          <h3 className="m-0 text-sm font-semibold text-geo-text">LLM</h3>
          <p className="m-0 mt-0.5 text-[11px] text-geo-dim font-mono">
            {health?.model || "…"}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`text-[10px] font-mono px-2 py-1 rounded ${
              starting
                ? "bg-blue-100 text-blue-700"
                : health?.reachable === true
                ? "bg-green-100 text-green-700"
                : health?.reachable === false
                ? "bg-amber-100 text-amber-700"
                : "bg-gray-100 text-gray-500"
            }`}
            title={
              starting
                ? `Container: ${health?.container_status || "starting"}`
                : health?.reachable === true
                ? `vLLM online at :${health?.port}`
                : !health?.docker_available
                ? "Docker not installed"
                : "Container not running"
            }
          >
            {starting
              ? "starting…"
              : health?.reachable === true
              ? "online"
              : health?.reachable === false
              ? "offline"
              : "…"}
          </span>
          {health && !health.reachable && !starting && (
            <button
              onClick={handleStart}
              disabled={!health.docker_available}
              className="text-[10px] font-semibold px-2 py-1 rounded text-white cursor-pointer shadow-sm transition-all hover:shadow-md hover:-translate-y-px hover:brightness-110 active:translate-y-0 active:brightness-95 disabled:bg-geo-border disabled:text-geo-dim disabled:cursor-not-allowed disabled:shadow-none disabled:hover:translate-y-0 disabled:hover:brightness-100"
              style={
                health.docker_available
                  ? { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
                  : undefined
              }
              title={
                !health.docker_available
                  ? (health.runtime === "ollama" ? "Install Ollama first" : "Install Docker first")
                  : health.runtime === "ollama"
                  ? `ollama pull ${health.model}`
                  : health.runtime === "vllm"
                  ? `Launch vLLM Docker container (see card below for full command)`
                  : `Cloud runtime — nothing to start locally`
              }
            >
              Start
            </button>
          )}
          {health?.reachable && (
            <button
              onClick={handleStop}
              className="text-[10px] font-semibold px-2 py-1 rounded bg-geo-surface border border-geo-border text-geo-text cursor-pointer transition-all hover:border-geo-accent hover:bg-geo-bg active:bg-geo-border active:scale-95"
              title={
                health.runtime === "vllm"
                  ? `docker stop the vLLM container on :${health.port} — weights stay cached, restart is ~30-60 s`
                  : health.runtime === "ollama"
                  ? `ollama stop ${health.model} — unloads from VRAM, disk cache kept`
                  : "Cloud runtime — nothing to stop locally"
              }
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Starting status / log tail */}
      {starting && (
        <div className="flex-shrink-0 mb-3 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-900">
          <div className="font-semibold mb-1">Starting LLM…</div>
          <div className="text-[10px] text-blue-800 mb-1 leading-relaxed">
            {health?.runtime === "vllm" ? (
              <>
                Backend started the docker container. vLLM is loading{" "}
                <span className="font-mono">{health?.model}</span>:
                weights (~25 s if cached), torch.compile (~10-40 s), CUDA
                graph capture (~10 s). <span className="font-semibold">
                Warm restart ≈ 30-60 s</span>; first launch with weight
                download is several minutes.
              </>
            ) : (
              <>
                Ollama is pulling <span className="font-mono">{health?.model}</span>.
                First launch downloads weights; subsequent launches are
                near-instant.
              </>
            )}
            {" "}Polling health every 5 s.
          </div>
          {logTail && (
            <pre className="text-[10px] font-mono text-blue-900/80 bg-white/60 rounded p-2 overflow-auto max-h-24 whitespace-pre-wrap">
              {logTail}
            </pre>
          )}
        </div>
      )}

      {/* Late-surface docker run failures (e.g. bad flags, GPU not available) */}
      {!starting && health?.last_start_error && !health.reachable && (
        <div className="flex-shrink-0 mb-3 bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-[11px] text-red-900">
          <div className="font-semibold mb-1">docker run failed</div>
          <pre className="text-[10px] font-mono text-red-900/80 bg-white/60 rounded p-2 overflow-auto max-h-32 whitespace-pre-wrap">
            {health.last_start_error}
          </pre>
        </div>
      )}

      {/* Model pulled but Ollama service isn't responding on the expected port. */}
      {!starting &&
        !health?.reachable &&
        !health?.last_start_error &&
        (health?.container_status === "pulled" ||
          health?.container_status === "running") && (
          <div className="flex-shrink-0 mb-3 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-900">
            <div className="font-semibold mb-1">
              Model is pulled, but the API isn't reachable on port {health?.port}
            </div>
            <div className="text-[11px] leading-relaxed mb-1">
              <span className="font-mono">{health?.model}</span> is on disk but
              Ollama isn't answering at{" "}
              <span className="font-mono">{health?.base_url}</span>.
            </div>
            <div className="text-[11px] leading-relaxed">
              <span className="font-semibold">Try:</span> check the Ollama
              Windows service is running, or run{" "}
              <span className="font-mono">ollama serve</span> in a terminal.
              Alternative: set <span className="font-mono">GEMMA_BASE_URL</span>{" "}
              + <span className="font-mono">GEMMA_API_KEY</span> to a cloud
              provider. See <span className="font-mono">docs/llm-setup.md</span>.
            </div>
          </div>
        )}

      {/* Settings sub-view — all config (model, token, docker cmd) lives here
          now, no longer clutters the Chat view. */}
      {subView === "settings" && (
        <LLMSettings
          health={health}
          hfToken={hfToken}
          setHfToken={setHfToken}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          customModel={customModel}
          setCustomModel={setCustomModel}
          useUngatedMirror={useUngatedMirror}
          setUseUngatedMirror={setUseUngatedMirror}
          resolvedModel={resolvedModel}
          vllmDockerCommand={vllmDockerCommand}
        />
      )}

      {/* CHAT sub-view — scene context + messages + composer. Rendered in
          two cases: (1) sidebar is on the Local chat subview; (2) the LLM
          pane has been popped out AND the popout target is routed here, in
          which case the chat body lives in the float while Settings /
          Examples / etc stays docked in the sidebar. Without (2) the
          floating popout goes empty whenever the user flips to Settings. */}
      {(subView === "chat" || !!chatBodyPortalTarget) && (
      <>
      {/* Scene context chip */}
      <div className="flex-shrink-0 mb-3 bg-geo-surface border border-geo-border rounded-lg px-3 py-2 text-[11px] text-geo-muted">
        <div className="font-semibold text-geo-text mb-0.5">Scene context</div>
        <div>
          {selectedArea
            ? "Area: selected"
            : "Area: none (draw one on Map tab)"}
          {" · "}
          {datasets.length
            ? `${datasets.length} dataset${datasets.length > 1 ? "s" : ""} loaded`
            : "no datasets"}
          {autoLabelSummary ? " · auto-label available" : ""}
        </div>
      </div>

      {/* Runtime setup block — stays DOCKED in the sidebar and never
          portals. Pop-out is a UI affordance for the chat itself; users
          still need the Start-LLM / HF-token / Docker command flow
          visible where they opened the tab. Gated the same as before:
          only when there's no active chat AND the LLM is unreachable. */}
      {messages.length === 0 && health && !health.reachable && !starting && (
        <div className="flex-shrink-0 text-[12px] text-geo-muted">
          {/* Prominent Start CTA — varies by runtime. Only ollama has a
              one-click start; vllm + cloud are managed externally. */}
          {health.runtime === "ollama" && (
              <div
                className="mb-3 rounded-xl p-4 text-white"
                style={{
                  background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)",
                }}
              >
                <div className="text-[13px] font-semibold mb-1">
                  LLM is not running
                </div>
                <div className="text-[11px] opacity-90 mb-3 leading-relaxed">
                  Click below to pull{" "}
                  <span className="font-mono">{health.model}</span> via Ollama
                  (auto-uses your GPU). First launch downloads the weights;
                  subsequent starts are near-instant.
                </div>
                <button
                  onClick={handleStart}
                  disabled={!health.docker_available}
                  className="w-full py-2.5 rounded-lg text-[13px] font-semibold bg-white cursor-pointer shadow-sm transition-all hover:shadow-lg hover:-translate-y-0.5 hover:bg-geo-bg active:translate-y-0 active:shadow-sm active:bg-geo-surface disabled:bg-white/30 disabled:text-white/60 disabled:cursor-not-allowed disabled:shadow-none disabled:hover:translate-y-0"
                  style={{ color: "#3a6690" }}
                >
                  {health.docker_available
                    ? "▶  Start LLM"
                    : "Ollama not installed"}
                </button>
                {!health.docker_available && (
                  <div className="mt-2 text-[10px] opacity-80">
                    Install Ollama from ollama.com/download, then refresh.
                  </div>
                )}
              </div>
            )}

            {/* vLLM — backend doesn't manage lifecycle; generate docker cmd */}
            {health.runtime === "vllm" && (
              <div className="mb-3 rounded-xl p-4 bg-geo-surface border border-geo-border text-geo-text">
                <div className="text-[13px] font-semibold mb-1">
                  Waiting for vLLM at{" "}
                  <span className="font-mono">{health.base_url}</span>
                </div>
                <div className="text-[11px] text-geo-muted mb-3 leading-relaxed">
                  Click <span className="font-semibold text-geo-text">▶ Start LLM</span> to launch
                  the container from here (easiest), or copy the generated command
                  and run it yourself in WSL (for logs / debugging).
                </div>

                {/* Quick-start: backend runs docker for you. Advanced config
                    (model / HF token / command preview) lives in Settings. */}
                <div className="flex gap-2 mb-3">
                  <button
                    onClick={handleStart}
                    className="flex-1 py-2.5 rounded-lg text-[13px] font-semibold text-white cursor-pointer shadow-sm transition-all hover:shadow-lg hover:-translate-y-0.5 hover:brightness-110 active:translate-y-0 active:brightness-95 active:shadow-sm"
                    style={{
                      background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)",
                    }}
                    title="Backend runs `docker run -d geoenv-vllm:latest ...` on your behalf"
                  >
                    ▶ Start LLM container
                  </button>
                  {onSubViewChange && (
                    <button
                      onClick={() => onSubViewChange("settings")}
                      className="px-3 rounded-lg text-[12px] font-semibold bg-geo-bg border border-geo-border text-geo-text cursor-pointer transition-all hover:border-geo-accent hover:bg-geo-surface active:scale-95"
                      title="Open Settings to change model, enter HF token, or copy the docker command"
                    >
                      ⚙ Settings
                    </button>
                  )}
                </div>

                {/* First-time-vs-quick-restart explainer */}
                <details className="mb-3 text-[10px] text-geo-muted" open>
                  <summary className="cursor-pointer font-semibold text-geo-text mb-1">
                    First-time setup vs quick restart
                  </summary>
                  <div className="mt-2 leading-relaxed space-y-1.5 pl-1">
                    <div>
                      <span className="font-semibold text-geo-text">
                        First launch (~10 min):
                      </span>{" "}
                      builds <span className="font-mono">geoenv-vllm:latest</span>{" "}
                      image if missing (~3 min), pulls model weights from HF
                      (~7 min for E4B, longer for 26B), captures CUDA graphs
                      (~1 min).
                    </div>
                    <div>
                      <span className="font-semibold text-geo-text">
                        Subsequent starts (~30–60 s):
                      </span>{" "}
                      image cached, weights cached in{" "}
                      <span className="font-mono">~/.cache/huggingface</span>,
                      compiled kernels cached. Just re-run the same docker
                      command.
                    </div>
                    <div>
                      <span className="font-semibold text-geo-text">Stop:</span>{" "}
                      Ctrl-C in the WSL terminal OR click the "Stop" button up
                      top (stops the running container; model stays on disk).
                    </div>
                  </div>
                </details>

                {/* HF token input (password-masked) */}
                <label className="block text-[10px] font-semibold text-geo-text mb-1">
                  HuggingFace token{" "}
                  <span className="text-geo-muted font-normal">
                    — required only for gated models like{" "}
                    <span className="font-mono">google/gemma-4-*</span>
                  </span>
                </label>
                <input
                  type="password"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_... (stored in this tab only, never sent to backend)"
                  autoComplete="off"
                  spellCheck={false}
                  className="w-full px-2 py-1.5 mb-2 bg-geo-bg border border-geo-border rounded text-[11px] font-mono text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
                />

                {/* Model picker — presets sized for the common RTX 5090 Laptop
                    (24 GB VRAM). Selecting a preset also pushes the model ID
                    to the backend so chat/validate target it. */}
                <label className="block text-[10px] font-semibold text-geo-text mb-1 mt-2">
                  Model
                </label>
                <select
                  value={selectedModel || CUSTOM_MODEL}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-2 py-1.5 mb-1 bg-geo-bg border border-geo-border rounded text-[11px] text-geo-text focus:outline-none focus:border-geo-accent cursor-pointer"
                >
                  <option value="">— use backend default —</option>
                  <optgroup label="Unsloth (ungated, no HF token needed)">
                    {MODEL_PRESETS.filter((m) => !m.gated).map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label} · {m.vram} {m.fitsLaptop ? "✓ fits 24GB" : "✗ too big"}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Google (gated — needs HF token + license accept)">
                    {MODEL_PRESETS.filter((m) => m.gated).map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label} · {m.vram} {m.fitsLaptop ? "✓ fits 24GB" : "✗ too big"}
                      </option>
                    ))}
                  </optgroup>
                  <option value={CUSTOM_MODEL}>Custom…</option>
                </select>

                {selectedModel === CUSTOM_MODEL && (
                  <input
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="e.g. Qwen/Qwen3.6-35B-A3B or any HF model ID"
                    autoComplete="off"
                    spellCheck={false}
                    className="w-full px-2 py-1.5 mb-1 bg-geo-bg border border-geo-border rounded text-[11px] font-mono text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
                  />
                )}

                {/* Warn when selected model likely doesn't fit */}
                {(() => {
                  const preset = MODEL_PRESETS.find((m) => m.value === selectedModel);
                  if (preset && !preset.fitsLaptop) {
                    return (
                      <div className="mb-2 text-[10px] text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
                        ⚠ {preset.vram} won't fit a typical 24 GB laptop GPU.
                        Expect OOM during weight loading. Pick the 4-bit variant
                        or add <span className="font-mono">--quantization fp8</span>.
                      </div>
                    );
                  }
                  return null;
                })()}

                {/* Legacy ungated toggle kept as a shortcut; the picker above
                    is the primary control. */}
                <label className="flex items-center gap-1.5 mb-2 text-[10px] text-geo-muted cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={useUngatedMirror}
                    onChange={(e) => setUseUngatedMirror(e.target.checked)}
                    className="cursor-pointer"
                  />
                  Shortcut: swap google/ → unsloth/ in the backend default
                </label>

                {/* Command preview */}
                <pre className="text-[10px] font-mono bg-geo-bg border border-geo-border rounded p-2 overflow-auto whitespace-pre-wrap mb-2 max-h-48">
                  {vllmDockerCommand}
                </pre>

                {/* Copy button */}
                <button
                  onClick={handleCopyCommand}
                  className="w-full py-2 rounded-lg text-[12px] font-semibold text-white cursor-pointer transition-all hover:shadow-md hover:-translate-y-px active:translate-y-0 active:brightness-95"
                  style={{
                    background:
                      "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)",
                  }}
                >
                  {copied ? "✓ Copied — paste into your WSL shell" : "Copy Docker command"}
                </button>

                <div className="mt-2 text-[10px] text-geo-muted leading-relaxed">
                  Paste into a WSL bash shell (not Windows CMD — backslash line
                  continuations are bash syntax). Once vLLM prints{" "}
                  <span className="font-mono">Uvicorn running</span>, this
                  panel auto-flips green.
                </div>

                {/* Security hygiene note */}
                <div className="mt-2 text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 leading-relaxed">
                  <span className="font-semibold">Security:</span> the token
                  stays in this browser tab's React state only — not sent to the
                  backend, not written to disk, not logged. Refreshing the page
                  clears it. Generate tokens with minimum <code>read</code>{" "}
                  scope at huggingface.co/settings/tokens.
                </div>
              </div>
            )}

            {/* Cloud — need an API key */}
            {health.runtime === "cloud" && (
              <div className="mb-3 rounded-xl p-4 bg-geo-surface border border-geo-border text-geo-text">
                <div className="text-[13px] font-semibold mb-1">
                  Cloud endpoint unreachable
                </div>
                <div className="text-[11px] text-geo-muted leading-relaxed">
                  Base URL: <span className="font-mono">{health.base_url}</span>
                  <br />
                  API key: <span className="font-mono">{health.api_key_set ? "set" : "NOT SET"}</span>
                  <br />
                  Drop <span className="font-mono">GEMMA_API_KEY</span> into{" "}
                  <span className="font-mono">backend/.env</span> and restart
                  the backend.
                </div>
              </div>
            )}
        </div>
      )}

      {/* Chat body (welcome + suggestions + messages + composer). Wrapped
          so it can portal into a floating panel while the setup block
          (above) stays docked in the sidebar. When chatBodyPortalTarget
          is null, renders inline in the sidebar like a normal chat. */}
      {(() => { const chatBody = (<>
      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1"
      >
        {messages.length === 0 && (
          <div className="text-[12px] text-geo-muted">
            <div className="mb-2">
              Ask anything about the current scene — land-cover interpretation,
              pipeline choices, seasonal effects, or "why is this polygon wrong?"
            </div>
            <div className="space-y-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  disabled={!health?.reachable}
                  className="w-full text-left px-3 py-2 bg-geo-surface border border-geo-border rounded-lg text-[12px] text-geo-text cursor-pointer transition-all hover:border-geo-accent hover:bg-geo-bg hover:-translate-y-px hover:shadow-sm active:translate-y-0 active:shadow-none active:bg-geo-border/50 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-none disabled:hover:bg-geo-surface"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`text-[13px] leading-relaxed ${
              m.role === "user" ? "text-right" : ""
            }`}
          >
            <div
              className={`inline-block max-w-[90%] px-3 py-2 rounded-lg relative group ${
                m.role === "user"
                  ? "bg-geo-accent text-white"
                  : "bg-geo-surface border border-geo-border text-geo-text"
              }`}
              style={
                m.role === "user"
                  ? { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
                  : undefined
              }
            >
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <div className="text-[10px] opacity-70 font-semibold uppercase tracking-wide">
                  {m.role === "user" ? "You" : "Gemma 4"}
                </div>
                <button
                  onClick={() => handleCopyMessage(i, m.content)}
                  className={`text-[9px] font-semibold px-1.5 py-0.5 rounded cursor-pointer transition-all active:scale-95 ${
                    m.role === "user"
                      ? "bg-white/15 hover:bg-white/25 text-white/90 hover:text-white"
                      : "bg-geo-bg hover:bg-geo-border/50 text-geo-muted hover:text-geo-text border border-geo-border"
                  }`}
                  title="Copy this message"
                >
                  {copiedIdx === i ? "✓ Copied" : "Copy"}
                </button>
              </div>
              <MarkdownText>{m.content}</MarkdownText>

              {/* Auto-retry chip — small model needed a nudge to emit a real
                  tool_call or to fill an empty reply. Shown only on the
                  assistant bubble where the retry actually fired. */}
              {m.role === "assistant" && (m.phantom_tool_call_fixed || m.empty_retry_fixed) && (
                <div
                  className="mt-1.5 text-[10px] font-mono uppercase tracking-wider inline-block px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300"
                  title={
                    m.phantom_tool_call_fixed
                      ? "The model first narrated a tool it didn't call. Auto-retried with a stricter nudge; this turn is the fixed response."
                      : "The model first returned empty content. Auto-retried at lower temperature; this turn is the fixed response."
                  }
                >
                  {m.phantom_tool_call_fixed ? "⚠ auto-retry: phantom tool call fixed" : "⚠ auto-retry: empty reply fixed"}
                </div>
              )}

              {/* Artifact pills — long tabular / geo outputs (CSV, GeoJSON,
                  PNG) are saved to /tmp by the tool and surfaced here as a
                  download instead of being stuffed into chat text. System
                  prompt tells the model to summarize + cite, not paste. */}
              {m.role === "assistant" && m.artifacts && m.artifacts.length > 0 && (
                <div className="mt-2 space-y-1">
                  {m.artifacts.map((a) => (
                    <a
                      key={a.id}
                      href={a.download_url}
                      download={a.filename}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-geo-bg hover:bg-geo-accent/10 border border-geo-border hover:border-geo-accent text-[11px] font-mono text-geo-text hover:text-geo-accent transition-colors cursor-pointer no-underline"
                      title={a.summary}
                      data-testid="artifact-pill"
                    >
                      <span className="text-[13px]">📄</span>
                      <span className="flex-1 truncate">{a.filename}</span>
                      <span className="text-geo-muted">
                        {a.size_bytes < 1024
                          ? `${a.size_bytes} B`
                          : a.size_bytes < 1024 * 1024
                            ? `${(a.size_bytes / 1024).toFixed(1)} KB`
                            : `${(a.size_bytes / 1024 / 1024).toFixed(1)} MB`}
                      </span>
                      <span className="text-[10px] uppercase tracking-wider text-geo-accent">download</span>
                    </a>
                  ))}
                </div>
              )}

              {/* Collapsible reasoning block — vLLM's --reasoning-parser gemma4
                  captures the model's <think> tokens in reasoning_content.
                  Makes "why did it narrate instead of tool-call" visible. */}
              {m.role === "assistant" && m.reasoning_content && m.reasoning_content.trim() && (
                <details className="mt-2 text-[11px]">
                  <summary className="cursor-pointer text-geo-muted hover:text-geo-text uppercase tracking-wider text-[10px] font-mono select-none">
                    reasoning ({m.reasoning_content.length} chars)
                  </summary>
                  <pre className="mt-1.5 whitespace-pre-wrap text-geo-muted bg-geo-bg border border-geo-border rounded p-2 font-mono text-[11px] leading-snug max-h-60 overflow-y-auto">
                    {m.reasoning_content}
                  </pre>
                </details>
              )}
            </div>
          </div>
        ))}

        {sending && (
          <div className="text-[12px] text-geo-muted italic">Gemma is thinking…</div>
        )}

        {error && (
          <div className="text-[12px] text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {/* Clear-chat action row — shown above composer when there's history */}
      {messages.length > 0 && (
        <div className="flex-shrink-0 mt-2 flex justify-end">
          <button
            onClick={handleClearChat}
            className="text-[10px] font-semibold px-2 py-1 rounded text-geo-muted cursor-pointer transition-all hover:text-red-700 hover:bg-red-50 active:scale-95"
            title="Clear conversation — keeps HF token + model selection"
          >
            🗑  Clear {messages.length} message{messages.length > 1 ? "s" : ""}
          </button>
        </div>
      )}

      {/* Offline banner — surfaces ABOVE the composer whenever the local
          LLM isn't reachable so users don't spam a disabled send button
          wondering why nothing responds. Previously the only signal was
          the textarea's greyed-out placeholder ("Start vLLM server to
          enable chat") which is easy to miss mid-chat if docker
          crashes after a healthy start. The runtime-specific copy
          points at the actual next action (start container, paste API
          key, etc.). Only shown when messages.length > 0 — the
          "new-chat" empty state already carries the big Start CTAs. */}
      {health && !health.reachable && !starting && messages.length > 0 && (
        <div
          className="flex-shrink-0 mt-2 flex items-start gap-2 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-[11px] text-red-900"
          role="alert"
          data-testid="gemma-offline-banner"
        >
          <span className="text-[13px] leading-none flex-shrink-0" aria-hidden>⚠</span>
          <div className="flex-1 min-w-0">
            <span className="font-semibold">Local LLM is offline.</span>{" "}
            <span className="text-red-800">
              {health.runtime === "ollama"
                ? "Ollama isn't responding. Restart the daemon or re-pull the model, then click below."
                : health.runtime === "vllm"
                  ? `vLLM container '${health.container_name ?? "geoenv-llm"}' isn't responding at ${health.base_url}. Check docker logs, or click below to try to start it again.`
                  : health.runtime === "cloud"
                    ? `Cloud endpoint ${health.base_url} is unreachable. API key is ${health.api_key_set ? "set" : "NOT set"}; verify connectivity or paste a key in backend/.env.`
                    : `Upstream ${health.base_url} unreachable.`}
              {health.last_start_error ? ` Last start error: ${health.last_start_error}.` : null}
            </span>
          </div>
          {/* Actionable: the same start button the empty-state setup
              block offers, but surfaced inline so the user doesn't
              have to scroll up / clear the chat to find it. Only for
              runtimes we can actually kick off from the backend —
              cloud endpoints need manual .env edits, not a button. */}
          {(health.runtime === "vllm" || health.runtime === "ollama") && health.docker_available && (
            <button
              type="button"
              onClick={handleStart}
              className="flex-shrink-0 px-2 py-1 rounded bg-white border border-red-300 text-red-900 text-[10px] font-semibold uppercase tracking-wider hover:bg-red-100 cursor-pointer transition-colors"
              data-testid="gemma-offline-restart"
            >
              ▶ Start
            </button>
          )}
        </div>
      )}

      {/* Composer */}
      <div className="flex-shrink-0 mt-2 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder={
            health?.reachable
              ? "Ask about this scene… (Enter to send, Shift+Enter for newline)"
              : "Start vLLM server to enable chat"
          }
          disabled={!health?.reachable || sending}
          rows={2}
          className="flex-1 px-3 py-2 bg-geo-bg border border-geo-border rounded-lg text-[13px] text-geo-text resize-none focus:outline-none focus:border-geo-accent disabled:opacity-50"
        />
        <button
          onClick={sending ? abort : () => send()}
          disabled={sending ? false : (!health?.reachable || sending || !input.trim())}
          className={`px-4 rounded-lg text-[13px] font-semibold transition-all ${
            !health?.reachable || sending || !input.trim()
              ? "bg-geo-border text-geo-dim cursor-not-allowed"
              : "text-white cursor-pointer shadow-sm hover:shadow-lg hover:-translate-y-0.5 hover:brightness-110 active:translate-y-0 active:brightness-95 active:shadow-sm"
          }`}
          style={
            !health?.reachable || sending || !input.trim()
              ? undefined
              : { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
          }
        >{sending ? "Stop" : "Send"}</button>
      </div>
      </>);
      return chatBodyPortalTarget ? createPortal(chatBody, chatBodyPortalTarget) : chatBody;
      })()}
      </>
      )}
    </div>
  );
}
