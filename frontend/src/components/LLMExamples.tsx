/**
 * LLM Examples subtab — curated copy-paste prompts that drive the full
 * Roger Studio agent loop (OlmoEarth FT + supporting tools) via any of the
 * three chat backends (Local Gemma, NIM, Claude).
 *
 * Every prompt here corresponds to a concrete user turn in
 * docs/LLM-CONVERSATIONS.md. The goal is to give a new user a known-good
 * starting point — tap Copy, switch to any chat pane, paste, send.
 *
 * Each conversation is a list of numbered user turns. Turns are sequential
 * within a conversation (each builds on the previous), but the first turn
 * of any conversation is a standalone entry point — start there.
 */
import { useState } from "react";

interface ExampleTurn {
  prompt: string;
  /** Optional hint shown next to the Copy button — what the LLM should do next. */
  hint?: string;
}

interface ExampleScenario {
  id: string;
  title: string;
  summary: string;
  setup: string;
  bestModel: string;
  turns: ExampleTurn[];
}

const SCENARIOS: ExampleScenario[] = [
  {
    id: "mangrove-kenya",
    title: "Mangrove mapping — Kenyan coast",
    summary:
      "4 turns. Catalog lookup → FT-Mangrove segmentation → NDVI seasonality check → FT-LFMC stress overlay. Exercises sliding-window FT + timeseries correlation.",
    setup: "Draw a rectangle over the Mombasa/Kilifi coast, or load the 'AWF Tsavo' pair from the Analysis tab's Sample Data demo pairs.",
    bestModel: "Any (small local Gemma works; Claude Sonnet 4.6 gives cleaner tool-call behavior).",
    turns: [
      {
        prompt:
          "What OlmoEarth FT models would be useful for this bbox, and which should I run first if I'm studying mangrove extent?",
        hint: "LLM should call query_olmoearth and recommend FT-Mangrove-Base.",
      },
      {
        prompt:
          "Run FT-Mangrove over this bbox for the dry season (2024-06-01 to 2024-09-30) with sliding_window=true so I get a spatial class map. Show me the class breakdown when it's done.",
        hint: "LLM calls run_olmoearth_inference with sliding_window=true; layer appears on the map.",
      },
      {
        prompt:
          "Now pull a 12-month NDVI timeseries over the same bbox so I can check for seasonal stress or die-off trends.",
        hint: "LLM calls query_ndvi_timeseries with months=12.",
      },
      {
        prompt:
          "Finally, run FT-LFMC over the same bbox with sliding_window=true so I can overlay live-fuel-moisture on the mangrove map. Then tell me which pixels look like stress candidates (classified as mangrove but with low LFMC).",
        hint: "LLM runs the regression FT head as a spatial layer + cross-references.",
      },
    ],
  },
  {
    id: "karst-forestloss-pa",
    title: "Karst forest-loss driver — PA Lancaster County",
    summary:
      "2 turns (agent chains multiple tools per turn). Polygon stats → FT-ForestLossDriver classification → NDVI delta vs. the prior year. Diagnoses a logging event uphill of a sinkhole.",
    setup: "Click the 'PA Karst Features' sample data card, then click any sinkhole polygon to select it.",
    bestModel: "Claude Sonnet 4.6 or NIM GPT-OSS 20B (two parallel tool calls per turn; small Gemma may split them).",
    turns: [
      {
        prompt:
          "This polygon is a confirmed sinkhole. First, tell me its area, elevation, and which county it's in. Then check if there's been any forest loss around it in the last two years — if yes, identify the driver.",
        hint: "LLM should call query_polygon, query_polygon_stats, and FT-ForestLossDriver in sequence.",
      },
      {
        prompt:
          "Pull a 24-month NDVI timeseries over the same bbox and compare May 2024 vs May 2023. If there's a drop > 0.15 NDVI, flag it as consistent with a selective-cut event.",
        hint: "LLM calls query_ndvi_timeseries(months=24) and does year-over-year delta.",
      },
    ],
  },
  {
    id: "sf-ecosystem-audit",
    title: "Ecosystem-type audit — SF Parks",
    summary:
      "1 turn, 7 parallel FT inferences. Maps each of 7 SF park polygons to an IUCN ecosystem type using FT-EcosystemTypeMapping-Base and ranks by class diversity.",
    setup: "Click the 'SF Parks & Landmarks' sample data card (loads 7 polygons at once).",
    bestModel: "Claude Opus 4.7 or NIM GPT-OSS 120B. On local Gemma, expect sequential execution — it'll take ~3–7 min on CPU.",
    turns: [
      {
        prompt:
          "For each of these 7 park polygons, classify the ecosystem type using the OlmoEarth EcosystemTypeMapping FT head with sliding_window=true. Aggregate the per-park class histograms and tell me which park has the most diverse set of ecosystem classes.",
        hint: "LLM emits one run_olmoearth_inference per polygon, then synthesizes a ranked table.",
      },
    ],
  },
  {
    id: "awf-vs-worldcover",
    title: "Compare FT-AWF vs. ESA WorldCover — southern Kenya",
    summary:
      "3 turns. Run FT-AWF segmentation → pull WorldCover histogram (Analysis tab) → diff the two to find where the 10-class AWF scheme adds detail that WorldCover misses.",
    setup: "Draw a rectangle anywhere in southern Kenya (≈0°S–5°S, 36°E–39°E).",
    bestModel: "Any — the per-turn tool calls are simple.",
    turns: [
      {
        prompt:
          "Run the FT-AWF model on this bbox with sliding_window=true for a dry-season date range (2024-07 to 2024-09). Tell me the breakdown of the 10 LULC classes when it's done.",
        hint: "LLM calls run_olmoearth_inference with FT-AWF-Base.",
      },
      {
        prompt:
          "Now pull an ESA WorldCover histogram for the same bbox from the Analysis tab (note: WorldCover isn't yet exposed as a chat tool — tell me to click the Analysis tab instead and summarize what I should expect to see there).",
        hint: "LLM correctly defers to the Analysis tab since no query_worldcover tool exists.",
      },
      {
        prompt:
          "Given the AWF classes you found, which ones would WorldCover likely MISS (classes AWF distinguishes that WorldCover lumps together)? Think about woodland_forest + montane_forest vs. WorldCover's single 'Tree cover' class.",
        hint: "Pure synthesis — no more tool calls. Good test of the LLM's ability to reason about schema differences.",
      },
    ],
  },
];


function CopyableTurn({
  n, prompt, hint,
}: { n: number; prompt: string; hint?: string }) {
  const [copied, setCopied] = useState(false);
  const doCopy = async () => {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Fallback: select the textarea so the user can Ctrl-C themselves.
      const ta = document.getElementById(`example-turn-${n}-${prompt.slice(0, 12)}`) as HTMLTextAreaElement | null;
      ta?.select();
    }
  };
  return (
    <div className="rounded-lg bg-geo-surface border border-geo-border p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-geo-muted">
          Turn {n}
        </span>
        <button
          type="button"
          onClick={doCopy}
          className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-geo-bg hover:bg-geo-accent hover:text-white border border-geo-border transition-colors cursor-pointer"
          title="Copy prompt to clipboard"
        >
          {copied ? "✓ copied" : "copy"}
        </button>
      </div>
      <p className="text-[12px] leading-relaxed text-geo-text whitespace-pre-wrap">
        {prompt}
      </p>
      {hint && (
        <p className="text-[10px] italic text-geo-muted leading-snug">
          → {hint}
        </p>
      )}
    </div>
  );
}


function ScenarioCard({ s }: { s: ExampleScenario }) {
  return (
    <article className="bg-gradient-panel rounded-xl p-4 border border-geo-border shadow-soft space-y-3" data-testid={`example-${s.id}`}>
      <header>
        <h4 className="text-sm font-semibold text-geo-text">{s.title}</h4>
        <p className="text-[11px] text-geo-muted leading-snug mt-1">{s.summary}</p>
      </header>
      <dl className="text-[11px] text-geo-muted space-y-1">
        <div>
          <dt className="inline uppercase tracking-wider font-mono">Setup: </dt>
          <dd className="inline">{s.setup}</dd>
        </div>
        <div>
          <dt className="inline uppercase tracking-wider font-mono">Best model: </dt>
          <dd className="inline">{s.bestModel}</dd>
        </div>
      </dl>
      <div className="space-y-2">
        {s.turns.map((t, i) => (
          <CopyableTurn key={i} n={i + 1} prompt={t.prompt} hint={t.hint} />
        ))}
      </div>
    </article>
  );
}


export function LLMExamples() {
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-shrink-0 mb-3">
        <h3 className="m-0 text-sm font-semibold text-geo-text">Multiround Conversation</h3>
        <p className="m-0 mt-0.5 text-[11px] text-geo-muted leading-snug">
          Copy a prompt, switch to any chat pane (Local / NIM / Claude / Gemini / ChatGPT), paste, send.
          Each scenario drives OlmoEarth FT inference + supporting tools end-to-end.
        </p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto space-y-4 pr-1">
        {SCENARIOS.map((s) => (
          <ScenarioCard key={s.id} s={s} />
        ))}
        <p className="text-[10px] text-geo-muted italic pt-2 pb-4">
          Full transcripts with tool-call JSON live in <code>docs/LLM-CONVERSATIONS.md</code>.
        </p>
      </div>
    </div>
  );
}
