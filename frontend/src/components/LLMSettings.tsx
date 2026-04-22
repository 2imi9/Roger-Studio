import { useMemo, useState } from "react";
import type { LLMHealth } from "../api/client";

// Model presets sized for common GPU memory budgets. VRAM estimates are for
// BF16 unless the tag says otherwise. fitsLaptop = RTX 5090 Laptop (24 GB).
interface ModelPreset {
  value: string;
  label: string;
  vram: string;
  fitsLaptop: boolean;
  gated: boolean;
}

export const MODEL_PRESETS: ModelPreset[] = [
  { value: "unsloth/gemma-4-26b-a4b-it-bnb-4bit", label: "Unsloth 26B A4B · 4-bit", vram: "~13 GB", fitsLaptop: true, gated: false },
  { value: "unsloth/gemma-4-26b-a4b-it",          label: "Unsloth 26B A4B · BF16", vram: "~52 GB", fitsLaptop: false, gated: false },
  { value: "unsloth/gemma-4-e4b-it",              label: "Unsloth E4B",            vram: "~8 GB",  fitsLaptop: true, gated: false },
  { value: "unsloth/gemma-4-e2b-it",              label: "Unsloth E2B",            vram: "~4 GB",  fitsLaptop: true, gated: false },
  { value: "google/gemma-4-26b-a4b-it",           label: "Google 26B A4B",         vram: "~52 GB", fitsLaptop: false, gated: true },
  { value: "google/gemma-4-31b-it",               label: "Google 31B dense",       vram: "~62 GB", fitsLaptop: false, gated: true },
  { value: "google/gemma-4-e4b-it",               label: "Google E4B",             vram: "~16 GB", fitsLaptop: true, gated: true },
];

export const CUSTOM_MODEL = "__custom__";

interface LLMSettingsProps {
  health: LLMHealth | null;
  hfToken: string;
  setHfToken: (v: string) => void;
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  customModel: string;
  setCustomModel: (v: string) => void;
  useUngatedMirror: boolean;
  setUseUngatedMirror: (v: boolean) => void;
  resolvedModel: string;
  vllmDockerCommand: string;
}

/**
 * Inline settings panel — rendered as a sub-view inside the LLM tab, not a
 * modal overlay. Contains everything that used to clutter the offline card:
 * HF token, model picker, Unsloth toggle, docker command preview.
 */
export function LLMSettings({
  health,
  hfToken,
  setHfToken,
  selectedModel,
  setSelectedModel,
  customModel,
  setCustomModel,
  useUngatedMirror,
  setUseUngatedMirror,
  resolvedModel,
  vllmDockerCommand,
}: LLMSettingsProps) {
  const [copied, setCopied] = useState(false);

  const oomWarning = useMemo(() => {
    const preset = MODEL_PRESETS.find((m) => m.value === selectedModel);
    return preset && !preset.fitsLaptop ? preset : null;
  }, [selectedModel]);

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(vllmDockerCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="flex-1 min-h-0 overflow-y-auto space-y-5 pr-1">
      {/* Runtime summary */}
      <section className="bg-geo-surface border border-geo-border rounded-lg p-3 text-[11px] text-geo-muted">
        <div className="font-semibold text-geo-text mb-1">Runtime</div>
        <div>
          Mode: <span className="font-mono text-geo-text">{health?.runtime ?? "…"}</span>
        </div>
        <div>
          Endpoint: <span className="font-mono text-geo-text">{health?.base_url ?? "…"}</span>
        </div>
        <div>
          Status:{" "}
          <span
            className={
              health?.reachable ? "font-semibold text-green-700" : "text-amber-700"
            }
          >
            {health?.reachable ? "online" : "offline"}
          </span>
        </div>
      </section>

      {/* HF Token */}
      <section>
        <label className="block text-[12px] font-semibold text-geo-text mb-1">
          HuggingFace token{" "}
          <span className="text-geo-muted font-normal">
            — required only for gated models (<span className="font-mono">google/gemma-4-*</span>)
          </span>
        </label>
        <input
          type="password"
          value={hfToken}
          onChange={(e) => setHfToken(e.target.value)}
          placeholder="hf_... (browser tab only, never sent to backend storage)"
          autoComplete="off"
          spellCheck={false}
          className="w-full px-3 py-2 bg-geo-surface border border-geo-border rounded-lg text-[12px] font-mono text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
        />
        <div className="mt-1 text-[10px] text-geo-muted">
          Stored in sessionStorage only (cleared when you close the browser tab).
          Passed as <span className="font-mono">-e HF_TOKEN=</span> when Start
          launches the container.
        </div>
      </section>

      {/* Model picker */}
      <section>
        <label className="block text-[12px] font-semibold text-geo-text mb-1">
          Model
        </label>
        <select
          value={selectedModel || CUSTOM_MODEL}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full px-3 py-2 bg-geo-surface border border-geo-border rounded-lg text-[12px] text-geo-text focus:outline-none focus:border-geo-accent cursor-pointer"
        >
          <option value="">— use backend default —</option>
          <optgroup label="Unsloth (ungated, no HF token needed)">
            {MODEL_PRESETS.filter((m) => !m.gated).map((m) => (
              <option key={m.value} value={m.value}>
                {m.label} · {m.vram} {m.fitsLaptop ? "✓ fits 24GB" : "✗ too big"}
              </option>
            ))}
          </optgroup>
          <optgroup label="Google (gated — needs HF token + license)">
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
            placeholder="e.g. Qwen/Qwen3.6-35B-A3B or any HF repo ID"
            autoComplete="off"
            spellCheck={false}
            className="w-full mt-2 px-3 py-2 bg-geo-surface border border-geo-border rounded-lg text-[12px] font-mono text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
          />
        )}

        {oomWarning && (
          <div className="mt-2 text-[11px] text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
            <span className="font-semibold">⚠ VRAM warning:</span> {oomWarning.vram} is too big
            for a typical 24 GB laptop GPU. Expect OOM during weight loading.
            Pick the 4-bit variant or add <span className="font-mono">--quantization fp8</span>.
          </div>
        )}

        <label className="flex items-center gap-2 mt-2 text-[11px] text-geo-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={useUngatedMirror}
            onChange={(e) => setUseUngatedMirror(e.target.checked)}
            className="cursor-pointer"
          />
          Shortcut: swap <span className="font-mono">google/</span> →{" "}
          <span className="font-mono">unsloth/</span> in the backend default (ungated mirror)
        </label>

        <div className="mt-2 text-[11px] text-geo-muted">
          Resolved model: <span className="font-mono text-geo-text">{resolvedModel}</span>
        </div>
      </section>

      {/* Docker command preview */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <label className="text-[12px] font-semibold text-geo-text">
            Docker command
          </label>
          <button
            onClick={copyCommand}
            className="text-[11px] font-semibold px-2.5 py-1 rounded text-white cursor-pointer shadow-sm transition-all hover:shadow-md hover:-translate-y-px active:translate-y-0 active:brightness-95"
            style={{ background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <pre className="text-[10px] font-mono bg-geo-surface border border-geo-border rounded-lg p-3 overflow-auto whitespace-pre-wrap max-h-64 text-geo-text">
          {vllmDockerCommand}
        </pre>
        <div className="mt-2 text-[10px] text-geo-muted leading-relaxed">
          The in-app Start button runs this command for you. Copy it only if
          you want to run it in WSL to watch logs directly, or to debug
          startup issues.
        </div>
      </section>

      {/* Security note */}
      <section className="text-[11px] text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 leading-relaxed">
        <span className="font-semibold">Security:</span> Token stays in this
        browser tab only — not written to disk, not committed to repo, not
        logged. Refreshing the tab preserves it via sessionStorage; closing
        the tab clears it. To rotate, visit{" "}
        <a
          href="https://huggingface.co/settings/tokens"
          target="_blank"
          rel="noreferrer"
          className="underline hover:text-amber-900"
        >
          huggingface.co/settings/tokens
        </a>
        .
      </section>
    </div>
  );
}
