import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { openaiChat, openaiHealth, openaiSetModel } from "../api/client";
import { useCancellableSend, wasAborted } from "../hooks/useCancellableSend";
import { ModelInput } from "./ModelInput";
import type { AgentArtifact, ChatMessage, OpenAIHealth } from "../api/client";
import type { AreaSelection, DatasetInfo } from "../types";
import { StatusOnline, StatusLoading, StatusError, StatusNoKey } from "./icons";
import { safeSetItem } from "../util/sessionStorage";
import { MarkdownText } from "./ui/MarkdownText";

// OpenAI model presets surfaced in the Settings picker. Default is GPT-5
// Mini — best tool-calling-to-cost ratio. Swap to GPT-5 for hardest tasks,
// o3-mini for reasoning-heavy turns. API keys scoped to the user's own
// OpenAI project, never persisted server-side.
export interface OpenAIModelPreset {
  value: string;
  label: string;
  size: string;
  notes: string;
}

export const OPENAI_MODEL_PRESETS: OpenAIModelPreset[] = [
  { value: "gpt-5",        label: "GPT-5",        size: "top",  notes: "Flagship; best tool-calling + reasoning" },
  { value: "gpt-5-mini",   label: "GPT-5 Mini",   size: "mid",  notes: "Default — fast, cheap, same tool quality" },
  { value: "gpt-4.1",      label: "GPT-4.1",      size: "long", notes: "1M-token context; proven tool loops" },
  { value: "gpt-4o",       label: "GPT-4o",       size: "base", notes: "Stable legacy baseline (text + vision)" },
  { value: "gpt-4o-mini",  label: "GPT-4o Mini",  size: "cheap",notes: "Lightest for high-volume short chats" },
  { value: "o3-mini",      label: "o3 Mini",      size: "reason",notes: "Reasoning-tuned; slower, best for hard tasks" },
];

export const CUSTOM_OPENAI_MODEL = "__custom_openai__";

interface OpenAIChatProps {
  selectedArea: AreaSelection | null;
  datasets: DatasetInfo[];
  autoLabelSummary?: Record<string, unknown> | null;
  /** When provided, the messages scroll + composer DOM re-parents into
   * this target via React portal. Config header (API key, model picker,
   * status) stays inline in the sidebar. Used by the floating-chat
   * overlay so the chat body can grow wider than the 480 px sidebar. */
  chatBodyPortalTarget?: HTMLElement | null;
}

const SUGGESTIONS = [
  "What land-cover classes should I expect in this area?",
  "Compare OlmoEarth and TIPSv2 for crop-type mapping here.",
  "Which Sentinel-2 bands distinguish mangroves from other coastal vegetation?",
  "Summarize the trade-offs between cloud inference and local inference for this workbench.",
];

// sessionStorage keys — per-tab, cleared when the tab closes. The API key
// NEVER touches disk; it's forwarded per-request to the backend proxy only.
const SS_API_KEY = "geoenv.openai.apiKey";
const SS_MODEL = "geoenv.openai.model";
const SS_CUSTOM = "geoenv.openai.customModel";
const SS_MESSAGES = "geoenv.openai.messages";
const SS_INPUT = "geoenv.openai.input";

function readSession<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function OpenAIChat({ selectedArea, datasets, autoLabelSummary, chatBodyPortalTarget }: OpenAIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    readSession<ChatMessage[]>(SS_MESSAGES, []),
  );
  const [input, setInput] = useState<string>(() => readSession(SS_INPUT, ""));
  const { sending, begin, abort, finish } = useCancellableSend();
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<OpenAIHealth | null>(null);
  const [apiKey, setApiKey] = useState<string>(() => readSession(SS_API_KEY, ""));
  const [selectedModel, setSelectedModel] = useState<string>(() =>
    readSession(SS_MODEL, ""),
  );
  const [customModel, setCustomModel] = useState<string>(() =>
    readSession(SS_CUSTOM, ""),
  );
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Persist on every change — message history is the one write big enough
  // to hit the per-origin quota after a long session, so it's the tracked
  // eviction candidate. See util/sessionStorage.ts for the retry path.
  useEffect(() => {
    safeSetItem(SS_MESSAGES, JSON.stringify(messages), { trackForEviction: true });
  }, [messages]);
  useEffect(() => {
    safeSetItem(SS_INPUT, JSON.stringify(input));
  }, [input]);
  useEffect(() => {
    safeSetItem(SS_API_KEY, JSON.stringify(apiKey));
  }, [apiKey]);
  useEffect(() => {
    safeSetItem(SS_MODEL, JSON.stringify(selectedModel));
  }, [selectedModel]);
  useEffect(() => {
    safeSetItem(SS_CUSTOM, JSON.stringify(customModel));
  }, [customModel]);

  const resolvedModel = useMemo<string>(() => {
    return customModel.trim() || health?.model || "";
  }, [customModel, health?.model]);

  const refreshHealth = async () => {
    try {
      const h = await openaiHealth(apiKey || undefined);
      setHealth(h);
      return h;
    } catch {
      return null;
    }
  };

  useEffect(() => {
    refreshHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-check health when the user types a new key (debounced lightly).
  useEffect(() => {
    const t = setTimeout(() => { refreshHealth(); }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey]);

  // Push model changes to the backend default so tool loops target it.
  useEffect(() => {
    if (!resolvedModel || resolvedModel === health?.model) return;
    openaiSetModel(resolvedModel).catch(() => { /* ignore — next health call refreshes */ });
  }, [resolvedModel, health?.model]);

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
      const reply = await openaiChat(next, sceneContext(), {
        apiKey: apiKey || undefined,
        model: resolvedModel || undefined,
        signal,
      });
      setMessages([...next, {
        role: "assistant",
        content: reply.content,
        artifacts: reply.artifacts,
      }]);
    } catch (e) {
      if (wasAborted(e)) return;   // user clicked Stop — not an error
      setError(e instanceof Error ? e.message : "Chat failed");
    } finally {
      finish();
    }
  };

  const handleClearChat = () => {
    if (messages.length === 0 && !input) return;
    if (!window.confirm("Clear the entire OpenAI conversation? This can't be undone.")) return;
    setMessages([]);
    setInput("");
    setError(null);
    try {
      sessionStorage.removeItem(SS_MESSAGES);
      sessionStorage.removeItem(SS_INPUT);
    } catch { /* noop */ }
  };

  const handleCopyMessage = async (idx: number, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx((cur) => (cur === idx ? null : cur)), 1500);
    } catch {
      setError("Clipboard write failed. Select + Ctrl+C the message manually.");
    }
  };

  const statusLabel = health?.reachable && health?.auth_ok
    ? "online"
    : health?.api_key_set === false
      ? "no key"
      : health?.auth_ok === false
        ? "bad key"
        : health
          ? "offline"
          : "…";

  const statusClass = health?.reachable && health?.auth_ok
    ? "bg-green-100 text-green-700"
    : health?.api_key_set === false
      ? "bg-amber-100 text-amber-700"
      : health?.auth_ok === false
        ? "bg-red-100 text-red-700"
        : "bg-gray-100 text-gray-500";

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <div>
          <h3 className="m-0 text-sm font-semibold text-geo-text">ChatGPT</h3>
          <p className="m-0 mt-0.5 text-[11px] text-geo-dim font-mono">
            {resolvedModel || health?.model || "…"} · OpenAI
          </p>
        </div>
        <span
          className={`text-[10px] font-mono px-2 py-1 rounded inline-flex items-center gap-1 ${statusClass}`}
          title={health?.error || health?.base_url || "OpenAI ChatGPT"}
        >
          {statusLabel === "online" ? (
            <StatusOnline size={10} />
          ) : statusLabel === "no key" ? (
            <StatusNoKey className="w-3 h-3" />
          ) : statusLabel === "bad key" ? (
            <StatusError className="w-3 h-3" />
          ) : (
            <StatusLoading className="w-3 h-3" />
          )}
          {statusLabel}
        </span>
      </div>

      {/* API key + model picker */}
      <div className="flex-shrink-0 mb-3 bg-geo-surface border border-geo-border rounded-lg p-3 space-y-2">
        <div>
          <label className="block text-[10px] font-semibold text-geo-text mb-1">
            ChatGPT / OpenAI API key{" "}
            <span className="text-geo-muted font-normal">
              — browser tab only, never stored server-side
            </span>
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={
              health?.api_key_set
                ? "env var is set on the backend — paste to override"
                : "nvapi-... (grab from platform.openai.com/api-keys)"
            }
            autoComplete="off"
            spellCheck={false}
            className="w-full px-2 py-1.5 bg-geo-bg border border-geo-border rounded text-[11px] font-mono text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
          />
        </div>

        <ModelInput
            value={customModel}
            onChange={setCustomModel}
            placeholder="e.g. gpt-5-mini"
            docsUrl="https://platform.openai.com/docs/models"
            docsLabel="OpenAI model docs"
            suggestions={[
              { value: "gpt-5", label: "GPT-5" }, { value: "gpt-5-mini", label: "GPT-5 Mini" }, { value: "gpt-4.1", label: "GPT-4.1" }, { value: "o3-mini", label: "o3 Mini" }
            ]}
          availableModels={health?.available_models}
          />

        {health?.error && (
          <div className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
            {health.error}
          </div>
        )}
      </div>

      {/* Scene context chip */}
      <div className="flex-shrink-0 mb-3 bg-geo-surface border border-geo-border rounded-lg px-3 py-2 text-[11px] text-geo-muted">
        <div className="font-semibold text-geo-text mb-0.5">Scene context</div>
        <div>
          {selectedArea ? "Area: selected" : "Area: none (draw one on Map tab)"}
          {" · "}
          {datasets.length
            ? `${datasets.length} dataset${datasets.length > 1 ? "s" : ""} loaded`
            : "no datasets"}
          {autoLabelSummary ? " · auto-label available" : ""}
        </div>
      </div>

      {/* Chat body (messages + clear chip + composer). Wrapped so it can
          portal into a floating panel while the config stays in the
          sidebar. When chatBodyPortalTarget is null, renders inline. */}
      {(() => { const chatBody = (<>
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1">
        {messages.length === 0 && (
          <div className="text-[12px] text-geo-muted">
            <div className="mb-2">
              Ask anything about the current scene. ChatGPT runs on the OpenAI
              API (no local GPU required) and has access to the same geo tools
              as Local Chat.
            </div>
            <div className="space-y-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  disabled={!health?.reachable || !health?.auth_ok}
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
            className={`text-[13px] leading-relaxed ${m.role === "user" ? "text-right" : ""}`}
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
                  {m.role === "user" ? "You" : "ChatGPT"}
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

              {/* Artifact pills — long tabular / geo tool outputs surfaced
                  as downloads (/api/artifacts/{id}) instead of pasted into
                  the bubble. System prompt tells the model to summarize +
                  cite, not dump the data. */}
              {m.role === "assistant" && m.artifacts && m.artifacts.length > 0 && (
                <div className="mt-2 space-y-1">
                  {m.artifacts.map((a: AgentArtifact) => (
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
            </div>
          </div>
        ))}

        {sending && (
          <div className="text-[12px] text-geo-muted italic">OpenAI is thinking…</div>
        )}

        {error && (
          <div className="text-[12px] text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {messages.length > 0 && (
        <div className="flex-shrink-0 mt-2 flex justify-end">
          <button
            onClick={handleClearChat}
            className="text-[10px] font-semibold px-2 py-1 rounded text-geo-muted cursor-pointer transition-all hover:text-red-700 hover:bg-red-50 active:scale-95"
            title="Clear OpenAI conversation"
          >
            🗑  Clear {messages.length} message{messages.length > 1 ? "s" : ""}
          </button>
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
            health?.reachable && health?.auth_ok
              ? "Ask ChatGPT about this scene… (Enter to send, Shift+Enter for newline)"
              : "Paste your OpenAI API key above to enable ChatGPT"
          }
          disabled={!health?.reachable || !health?.auth_ok || sending}
          rows={2}
          className="flex-1 px-3 py-2 bg-geo-bg border border-geo-border rounded-lg text-[13px] text-geo-text resize-none focus:outline-none focus:border-geo-accent disabled:opacity-50"
        />
        <button
          onClick={sending ? abort : () => send()}
          disabled={sending ? false : (!health?.reachable || !health?.auth_ok || sending || !input.trim())}
          className={`px-4 rounded-lg text-[13px] font-semibold transition-all ${
            !health?.reachable || !health?.auth_ok || sending || !input.trim()
              ? "bg-geo-border text-geo-dim cursor-not-allowed"
              : "text-white cursor-pointer shadow-sm hover:shadow-lg hover:-translate-y-0.5 hover:brightness-110 active:translate-y-0 active:brightness-95 active:shadow-sm"
          }`}
          style={
            !health?.reachable || !health?.auth_ok || sending || !input.trim()
              ? undefined
              : { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
          }
        >{sending ? "Stop" : "Send"}</button>
      </div>
      </>);
      return chatBodyPortalTarget ? createPortal(chatBody, chatBodyPortalTarget) : chatBody;
      })()}
    </div>
  );
}
