/**
 * ModelInput — shared model-id picker for every cloud provider chat pane.
 *
 * Rationale: curated preset dropdowns go stale fast (new GPT / Claude /
 * Gemini versions ship monthly, old names get deprecated). Better to let
 * the user type any model id they want + point them at the provider's
 * official model-list doc so they can look up the current name
 * themselves. Optional suggestions render as clickable chips below for
 * the common cases.
 *
 * Each provider pane passes:
 *   - ``value`` / ``onChange``  — controlled text field
 *   - ``placeholder``           — the provider's default model id, so the
 *     pane shows "use backend default (gpt-5-mini)" when the field is blank
 *   - ``docsUrl``               — link to the provider's "list of models"
 *     doc page; rendered as a tiny help link under the input
 *   - ``docsLabel``             — human-readable link text
 *   - ``suggestions`` (optional) — short list of clickable chip ids
 */
import { useState } from "react";

export interface ModelSuggestion {
  value: string;
  label?: string;
  notes?: string;
}

interface ModelInputProps {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  docsUrl: string;
  docsLabel: string;
  /** Hand-picked common examples. Curation-light; user can always type any id. */
  suggestions?: ModelSuggestion[];
  /** Authoritative list pulled from the provider's /v1/models at health-check
   * time. When present, rendered as a separate "live from server" chip row so
   * the user can pick one that's guaranteed to exist on their account. */
  availableModels?: string[];
  /** Small label shown above the input. Defaults to "Model id". */
  fieldLabel?: string;
}

export function ModelInput({
  value,
  onChange,
  placeholder,
  docsUrl,
  docsLabel,
  suggestions = [],
  availableModels = [],
  fieldLabel = "Model id",
}: ModelInputProps) {
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [liveModelsOpen, setLiveModelsOpen] = useState(false);

  return (
    <div className="space-y-1" data-testid="model-input">
      <label className="block text-[10px] uppercase tracking-wider text-geo-muted font-mono">
        {fieldLabel}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? "(use backend default)"}
        spellCheck={false}
        autoComplete="off"
        className="w-full px-2 py-1.5 bg-geo-bg border border-geo-border rounded text-[11px] font-mono text-geo-text focus:outline-none focus:border-geo-accent"
        data-testid="model-input-field"
      />
      <div className="flex items-center justify-between gap-2 text-[10px] text-geo-muted">
        <a
          href={docsUrl}
          target="_blank"
          rel="noreferrer"
          className="text-geo-accent hover:underline flex-shrink-0"
          title="Browse the provider's official model list"
        >
          ↗ {docsLabel}
        </a>
        <div className="flex items-center gap-2 flex-shrink-0">
          {availableModels.length > 0 && (
            <button
              type="button"
              onClick={() => setLiveModelsOpen((v) => !v)}
              className="text-geo-muted hover:text-geo-text cursor-pointer"
              title="Models your API key actually has access to, pulled from the provider's /v1/models endpoint at last health-check"
              data-testid="model-input-live-toggle"
            >
              {liveModelsOpen ? "hide live" : `live (${availableModels.length})`}
            </button>
          )}
          {suggestions.length > 0 && (
            <button
              type="button"
              onClick={() => setSuggestionsOpen((v) => !v)}
              className="text-geo-muted hover:text-geo-text cursor-pointer"
            >
              {suggestionsOpen ? "hide examples" : `examples (${suggestions.length})`}
            </button>
          )}
        </div>
      </div>
      {liveModelsOpen && availableModels.length > 0 && (
        <div
          className="flex flex-wrap gap-1 pt-1 max-h-32 overflow-y-auto border-t border-geo-border"
          data-testid="model-input-live"
        >
          {availableModels.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => onChange(id)}
              title={`Verified live on ${docsLabel} (from /v1/models)`}
              className="px-2 py-0.5 rounded text-[10px] font-mono bg-geo-bg border border-geo-accent/40 hover:bg-geo-accent hover:text-white hover:border-geo-accent transition-colors cursor-pointer"
            >
              {id}
            </button>
          ))}
        </div>
      )}
      {suggestionsOpen && suggestions.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1" data-testid="model-input-suggestions">
          {suggestions.map((s) => (
            <button
              key={s.value}
              type="button"
              onClick={() => onChange(s.value)}
              title={s.notes || s.label || s.value}
              className="px-2 py-0.5 rounded text-[10px] font-mono bg-geo-surface border border-geo-border hover:bg-geo-accent/10 hover:border-geo-accent hover:text-geo-accent transition-colors cursor-pointer"
            >
              {s.label || s.value}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
