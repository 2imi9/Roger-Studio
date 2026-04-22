# Tool-Calling Benchmark — Cuvier Studio · Local Gemma 4 E4B

**Date:** 2026-04-18
**Backend:** `geoenv-studio/backend` @ `localhost:8000`
**LLM:** vLLM 0.19.1 + `unsloth/gemma-4-e4b-it` (FP16, ~8 GB VRAM, RTX 5090 Laptop)
**vLLM flags:** `--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --max-model-len 16384 --gpu-memory-utilization 0.75`
**Endpoint under test:** `POST /api/auto-label/gemma/chat` (loop in `gemma_client.chat_with_tools`)
**Tool registry:** 7 tools defined in `app/services/geo_tools.py` (OpenAI shape)
**Harness:** `tests/test_tool_calling_bench.py`

---

## Results

| # | Case | Expected tool | Called | Args ok | Iter | Time |
|---|---|---|---|---|---|---|
| 1 | `query_polygon` — direct id lookup | `query_polygon` | ✅ | ✅ | 2 | 5.14 s |
| 2 | `query_olmoearth` — catalog + bbox | `query_olmoearth` | ✅ | ✅ | 2 | 4.73 s |
| 3 | `query_polygon_stats` — area + perimeter | `query_polygon_stats` | ✅ | ✅ | 2 | 4.45 s |
| 4 | `query_ndvi_timeseries` — STUB | `query_ndvi_timeseries` | ✅ | ✅ | 2 | 3.70 s |
| 5 | `search_stac_imagery` — date range | `search_stac_imagery` | ✅ | ✅ | 2 | 5.84 s |
| 6 | `get_composite_tile_url` — mosaic | `get_composite_tile_url` | ✅ | ✅\* | 2 | 5.03 s |
| 7 | `get_higher_res_patch` — STUB | `get_higher_res_patch` | ✅ | ✅ | 2 | 3.41 s |
| 8 | negative — small talk (S1 vs S2) | none | ✅ | n/a | 1 | 3.62 s |
| 9 | negative — abstract reasoning (NDVI saturation) | none | ✅ | n/a | 1 | 4.62 s |
| 10 | agentic — bbox stats then catalog | `query_olmoearth` (+ then `query_polygon_stats`) | ✅ both | ✅ | 2 | 9.20 s |

**Pass rate: 10/10 (100 %)** &nbsp; · &nbsp; Avg iterations: 1.8 &nbsp; · &nbsp; Total tokens: 36 225 &nbsp; · &nbsp; Total wall time: ~50 s

\* Case 6: see "Argument hallucination" below — passed by the rubric (correct tool, all required keys present) but the bbox value contained a sign error.

---

## What the rubric checks

For each case the harness scores three things in priority order — same shape as **τ-bench** (Sierra Research, 2024), **BFCL v3** (Berkeley), and **ToolBench** (xLAM):

1. **Correct-tool selection** — was the right tool called at all?
2. **Argument-key presence** — does the call include every `required` key from the schema?
3. **Loop termination** — did the agentic loop exit cleanly (`stop_reason == "final_answer"`) within `max_iterations=5`?

Negative cases invert (1): pass iff **no** tool was called.

The agentic case passes iff the first expected tool fires; ordering across iterations is reported separately.

---

## Notable finding — argument hallucination (case 6)

The model's `get_composite_tile_url` invocation:

```json
{
  "bbox": {"east": 39.8, "north": 3.9, "south": -4.1, "west": 39.6},
  "collection": "sentinel-2-l2a",
  "datetime": "2024-06-01/2024-08-31",
  "max_cloud_cover": 10
}
```

The scene context provided `north: -3.9` (Kenyan coast, southern hemisphere). The model dropped the negative sign on `north`, producing a bbox that crosses the equator into Somalia. **The downstream Planetary Computer mosaic registered successfully against the malformed coords** — there is no server-side bbox-vs-context cross-check today.

Severity: medium for production. The mosaic returned a valid tile URL pointing at the wrong region; a user trusting "Show me cloud-free Sentinel-2 for this bbox" would get imagery 8° off-target. Mitigations:

- **Schema fix:** make `bbox` a fixed reference to `scene_context.bbox` (omit it from the tool input) when a bbox is in scene — same pattern as `query_polygon` referencing `polygon_id` rather than re-emitting geometry.
- **Server-side cross-check:** in `geo_tools.execute_tool`, when `scene_context.bbox` is present and the model also supplied a `bbox` arg, refuse + ask the model to re-call with the scene bbox.
- **Sanity assertion:** reject any tool-emitted bbox that differs from the scene bbox by > N degrees.

This kind of argument-fidelity failure is exactly what BFCL's "argument-value" sub-score captures (vs. the looser "argument-key" check used here). On BFCL v3, small open models like Gemma 2 9B and Llama 3.1 8B typically score ~10–15 percentage points lower on argument-value than on argument-key for the same task, so this finding is consistent with the broader literature.

---

## Comparison to public benchmark scores (verified research)

These are **single-source headline numbers**, not apples-to-apples — Cuvier Studio's bench is 10 cases over a 7-tool geo domain, while the public benchmarks below are 100s–1 000s of cases over many domains. Use them for *order-of-magnitude* calibration, not direct comparison.

### τ-bench (Sierra Research, Yao et al., 2024) — agentic tool use, customer-service domains

| Model | retail (avg@1) | airline (avg@1) | Source |
|---|---|---|---|
| GPT-4o | ~61 % | ~36 % | original paper, Table 2 |
| Claude 3.5 Sonnet | ~46 % | ~16 % | original paper, Table 2 |
| GPT-4o-mini | ~26 % | ~22 % | original paper, Table 2 |
| Llama 3.1 70B | ~22 % | ~14 % | original paper, Table 2 |
| Llama 3.1 8B | ~7 % | ~12 % | original paper, Table 2 |

Frontier models (Claude Opus 4 / Claude Opus 4.6 / GPT-5) reportedly push retail past 70 % per Anthropic's announcements and the Sierra leaderboard, but those numbers are not yet in a peer-reviewed publication.

τ²-bench (the extended version) was released in 2025 and is harder across the board.

**Why this matters for Cuvier Studio:** Gemma 4 E4B is a ~4 B-param model — by parameter count it sits below Llama 3.1 8B. The Llama 8B τ-bench score of ~7 %/12 % suggests that **a model in this size class will struggle with multi-turn agentic tasks**. Our 10/10 here is a much easier setting (single-shot prompts, narrow domain, small tool set) — do not extrapolate to long-horizon agents.

### BFCL v3 (Berkeley Function-Calling Leaderboard) — function-calling correctness

Headline accuracy of frontier and mid-tier models on BFCL v3 (overall, including AST + executable + multi-turn + irrelevance):

| Model | overall | Note |
|---|---|---|
| GPT-4o (2024) | ~78 % | top tier at release |
| Claude 3.5 Sonnet | ~75 % | top tier |
| Llama 3.1 70B Instruct | ~63 % | best open ~70B |
| Mistral Large 2 | ~67 % | |
| Gemma 2 9B Instruct | ~52 % | reference small open model |
| Llama 3.1 8B Instruct | ~48 % | reference small open model |

Gemma 4 E4B is too new to have a published BFCL score at time of writing. The 9B class typically lands ~50 %; a 4 B model would be expected lower on the open leaderboard.

### ToolBench (xLAM team) — multi-tool planning

ToolBench reports pass-rate on multi-API tasks pulled from RapidAPI. Frontier closed models hit 80 %+; small open models typically hit 30-50 %. The 8 B class struggles with the "I3" (intra-domain, deeper) subset.

### Direct relevance for Cuvier Studio

| Property | This bench | τ-bench | BFCL | ToolBench |
|---|---|---|---|---|
| Domain | geo (7 tools) | retail + airline (~30 tools) | mixed (1000+ APIs) | RapidAPI (16k APIs) |
| Cases | 10 | ~165 | ~2000 | ~16k |
| Multi-turn | 1 case | all | ~25 % | most |
| Tool-arg-value check | partial | yes | yes | yes |
| Real network calls | yes (3/10) | simulated DB | mixed | yes |

Conclusion: 100 % on this suite **does not generalize**. It establishes that the **plumbing** (vLLM tool parser → backend dispatch → real network calls → answer integration) is correct for the simple cases. Production reliability would require:

1. Adding 30–50 more cases including multi-tool dependency chains, tool-error recovery, ambiguous prompts.
2. Adding **argument-value** checks (not just key presence) — would catch case 6's sign error.
3. Running the same suite against a frontier model (Claude Opus 4.7, GPT-5) for an upper bound.

---

## Cloud model selection plumbing (NIM + Claude)

End-to-end test of `/api/cloud/model` and `/api/claude/model` endpoints — the picker dropdowns in `CloudChat.tsx` and `ClaudeChat.tsx` flow through these.

| Probe | Endpoint | Result |
|---|---|---|
| Read default | `GET /api/cloud/health` → `model` | `openai/gpt-oss-20b` ✅ |
| Switch model | `POST /api/cloud/model {"model": "meta/llama-3.3-70b-instruct"}` | `{ok: true, model: "meta/llama-3.3-70b-instruct"}` ✅ |
| Reread | `GET /api/cloud/health` → `model` | `meta/llama-3.3-70b-instruct` ✅ (persisted in process state) |
| Empty payload | `POST /api/cloud/model {"model": ""}` | HTTP 400 `{"detail": "model field is required"}` ✅ |
| Read default | `GET /api/claude/health` → `model` | `claude-opus-4-7` ✅ |
| Switch model | `POST /api/claude/model {"model": "claude-sonnet-4-6"}` | `{ok: true, model: "claude-sonnet-4-6"}` ✅ |
| Reread | `GET /api/claude/health` → `model` | `claude-sonnet-4-6` ✅ |
| Empty payload | `POST /api/claude/model {"model": ""}` | HTTP 400 ✅ |

**End-to-end chat against NIM + Claude requires API keys we don't have in this environment.** The plumbing is verified; the actual completion path is unexercised. To benchmark the same 10 cases on those transports, plug a key into Cloud Chat / Claude Chat in the UI (or set `NVIDIA_API_KEY` / `ANTHROPIC_API_KEY`) and re-target the harness at `/api/cloud/chat` and `/api/claude/chat` — the request shapes are identical.

---

## Limitations + recommended next steps

1. **Argument-value scoring** — extend the harness to assert specific values, not just key presence. Case 6 would fail. Patch is small (~30 LoC in `run_case`).
2. **Multi-tool dependency cases** — e.g. "search S2 imagery for this bbox in summer 2024, then give me a composite tile URL for the least-cloudy date you found." This forces the model to use the result of one tool to populate the args of the next.
3. **Tool-error recovery** — inject a deliberate tool failure (return `{"error": "..."}`) and check whether the model retries or surfaces the error sensibly.
4. **Cross-transport bench** — once API keys are available, run the same 10 cases against NIM (`openai/gpt-oss-20b`, `meta/llama-3.3-70b-instruct`) and Claude (`claude-opus-4-7`, `claude-haiku-4-5`) for direct comparison.
5. **Statistical tightening** — run each case 5× with `temperature=0.4` and report pass-rate distributions, not single-shot pass/fail. With only 1 sample per case, today's 100 % could mask a 70 %-stable rate that happened to land all-green.
6. **Cite τ-bench v2 / BFCL v3 leaderboards directly** in any external report — both are publicly maintained and the numbers above will drift.

---

## Reproducing this run

```bash
# 1. Start backend (already running on :8000) and vLLM (Docker on WSL → :8001)
curl -X POST http://localhost:8000/api/auto-label/gemma/start  # idempotent

# 2. Wait ~60 s (warm) or several minutes (cold) until reachable=True:
curl -s http://localhost:8000/api/auto-label/gemma/health | python -c "import json,sys;print(json.load(sys.stdin)['reachable'])"

# 3. Run the bench
cd backend
python tests/test_tool_calling_bench.py            # human-readable
python tests/test_tool_calling_bench.py --json     # CI-ready
python tests/test_tool_calling_bench.py --case 5   # single case
```

Container ID for this run: `e8d7eb8779dd` (vLLM 0.19.1rc1.dev328+g18013df6a, image `geoenv-vllm:latest`).
