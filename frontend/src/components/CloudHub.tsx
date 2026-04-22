/**
 * CloudHub — single Cloud subtab under the LLM tab that multiplexes across
 * three cloud providers: NVIDIA NIM (CloudChat), Anthropic Claude (ClaudeChat),
 * and Google Gemini (GeminiChat).
 *
 * Each provider pane keeps its own state (messages / API key / model picker
 * / chat history) — we simply mount all three and toggle visibility via
 * display:none so switching providers doesn't wipe an in-progress chat.
 *
 * The provider pick persists to sessionStorage (``geoenv.cloudhub.provider``)
 * so a tab refresh lands you back on the same provider.
 */
import { useEffect, useState } from "react";

import type { AreaSelection, DatasetInfo } from "../types";
import { CloudChat } from "./CloudChat";
import { ClaudeChat } from "./ClaudeChat";
import { GeminiChat } from "./GeminiChat";
import { OpenAIChat } from "./OpenAIChat";

type Provider = "nim" | "claude" | "gemini" | "openai";

const SS_PROVIDER = "geoenv.cloudhub.provider";

const PROVIDER_INFO: Record<Provider, { label: string; tagline: string }> = {
  nim:    { label: "NVIDIA NIM", tagline: "GPT-OSS / Llama / Nemotron / DeepSeek / Qwen" },
  claude: { label: "Claude",      tagline: "Anthropic Opus / Sonnet / Haiku" },
  gemini: { label: "Gemini",      tagline: "Google 2.5 Pro / Flash / Flash-Lite / 2.0" },
  openai: { label: "ChatGPT",     tagline: "OpenAI ChatGPT — GPT-5 / GPT-4.1 / GPT-4o / o3-mini" },
};

interface CloudHubProps {
  selectedArea: AreaSelection | null;
  datasets: DatasetInfo[];
  autoLabelSummary?: Record<string, unknown> | null;
  /** Portal target for the chat body (messages + composer). When set,
   * each provider pane keeps its config header inline but re-parents
   * the scroll + composer into the floating panel via React portal. */
  chatBodyPortalTarget?: HTMLElement | null;
}

function readProvider(): Provider {
  try {
    const raw = sessionStorage.getItem(SS_PROVIDER);
    if (raw === "nim" || raw === "claude" || raw === "gemini" || raw === "openai") return raw;
  } catch { /* noop */ }
  return "nim";
}

export function CloudHub({ selectedArea, datasets, autoLabelSummary, chatBodyPortalTarget }: CloudHubProps) {
  const [provider, setProvider] = useState<Provider>(readProvider);

  useEffect(() => {
    try { sessionStorage.setItem(SS_PROVIDER, provider); } catch { /* noop */ }
  }, [provider]);

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Provider picker — sits above each provider's own header so the
          two headers stay distinct (chat status pill is rendered by the
          inner pane, not here). */}
      <div
        className="flex-shrink-0 mb-3 flex items-center gap-1 p-0.5 bg-geo-bg border border-geo-border rounded-lg"
        data-testid="cloudhub-provider-picker"
      >
        {(Object.keys(PROVIDER_INFO) as Provider[]).map((p) => {
          const active = p === provider;
          const info = PROVIDER_INFO[p];
          return (
            <button
              key={p}
              type="button"
              onClick={() => setProvider(p)}
              className={`flex-1 px-3 py-1.5 rounded-md text-[11px] font-semibold uppercase tracking-wider transition-colors cursor-pointer ${
                active
                  ? "bg-geo-accent text-white shadow-soft"
                  : "text-geo-muted hover:text-geo-text hover:bg-geo-surface"
              }`}
              title={info.tagline}
              data-testid={`cloudhub-provider-${p}`}
            >
              {info.label}
            </button>
          );
        })}
      </div>

      {/* Mount all three so each keeps its own chat history across
          provider switches. display:none preserves React state better
          than conditional-mounting. */}
      <div
        className="flex-1 min-h-0 flex flex-col"
        style={{ display: provider === "nim" ? "flex" : "none" }}
      >
        <CloudChat
          selectedArea={selectedArea}
          datasets={datasets}
          autoLabelSummary={autoLabelSummary}
          chatBodyPortalTarget={provider === "nim" ? chatBodyPortalTarget : null}
        />
      </div>
      <div
        className="flex-1 min-h-0 flex flex-col"
        style={{ display: provider === "claude" ? "flex" : "none" }}
      >
        <ClaudeChat
          selectedArea={selectedArea}
          datasets={datasets}
          autoLabelSummary={autoLabelSummary}
          chatBodyPortalTarget={provider === "claude" ? chatBodyPortalTarget : null}
        />
      </div>
      <div
        className="flex-1 min-h-0 flex flex-col"
        style={{ display: provider === "gemini" ? "flex" : "none" }}
      >
        <GeminiChat
          selectedArea={selectedArea}
          datasets={datasets}
          autoLabelSummary={autoLabelSummary}
          chatBodyPortalTarget={provider === "gemini" ? chatBodyPortalTarget : null}
        />
      </div>
      <div
        className="flex-1 min-h-0 flex flex-col"
        style={{ display: provider === "openai" ? "flex" : "none" }}
      >
        <OpenAIChat
          selectedArea={selectedArea}
          datasets={datasets}
          autoLabelSummary={autoLabelSummary}
          chatBodyPortalTarget={provider === "openai" ? chatBodyPortalTarget : null}
        />
      </div>
    </div>
  );
}
