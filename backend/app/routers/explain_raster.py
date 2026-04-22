"""POST /api/explain-raster — LLM-backed explainer for inference rasters.

The UI used to dump every inference layer's full class list (~110 rows for
EcosystemTypeMapping) in a floating legend. Users asked for an agent-style
explanation instead: "tell me what this raster represents" in plain English,
with a compact summary rather than a wall of class names. This route takes
the raster's metadata + optionally the predicted class distribution and
returns a 2–3 paragraph natural-language explanation.

Provider chain (first available wins):
  1. NVIDIA NIM (``cloud_llm_client``) — default. Free-tier endpoint
     ``minimaxai/minimax-m2.7`` (best agentic / reasoning model on the
     free tier). Text-only is fine because the agent calls
     ``query_raster_histogram`` / ``query_raster_scalar_stats`` to read
     the raster's real pixel distribution — no vision encoder required.
  2. Claude (Anthropic) — fallback when NIM is unreachable / unconfigured.
  3. Gemma (local vLLM / Ollama) — only if the local runtime is running.
  4. Deterministic fallback — templated summary from metadata alone,
     labeled ``source="fallback"`` so the UI can surface that the answer
     is not LLM-generated.

Request-scoped API keys are respected so the UI can pass the user's
sessionStorage-stored key without requiring server-side env vars.

The endpoint is ``async`` and bounded — meant to answer a user's click
on a raster pill, so the chained LLM attempts cap at ~30 s total.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from app.services import (
    claude_client,
    cloud_llm_client,
    gemma_client,
    geo_tools,
    olmoearth_inference,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ExplainRasterRequest(BaseModel):
    model_repo_id: str
    task_type: str | None = None           # classification / segmentation / regression / embedding
    colormap: str | None = None            # e.g. "ecosystem", "mangrove"
    # Bbox used in the inference call — helps the LLM reason about location.
    bbox: dict[str, float] | None = None
    scene_id: str | None = None
    scene_datetime: str | None = None
    scene_cloud_cover: float | None = None
    # Class metadata (optional; present for classification / segmentation)
    class_names: list[str] | None = None
    # Top-N class → score pairs if the caller pre-computed them. Lets the
    # LLM focus on the classes that actually show up on the map instead of
    # narrating the full 110-class catalog.
    top_classes: list[dict[str, Any]] | None = None
    # Regression / scalar prediction summary.
    prediction_value: float | None = None
    units: str | None = None
    # When the inference produced a preview stub (kind="stub"), include
    # the reason so the LLM can explain why the colors aren't a real model
    # output rather than misleading the user.
    stub_reason: str | None = None
    # Per-provider API keys from the user's sessionStorage. Optional —
    # server-side env vars are tried when absent. The UI passes whichever
    # it has configured so the explainer can succeed even when the user
    # only set up one provider.
    nim_api_key: str | None = None
    claude_api_key: str | None = None
    # Back-compat alias; some earlier callers passed ``api_key`` for
    # Claude. Treat as claude_api_key when claude_api_key is absent.
    api_key: str | None = None
    # Inference job_id. Populated in scene_context so the agent's raster
    # tools (query_raster_histogram, query_raster_scalar_stats) default
    # to the right job when the LLM forgets to pass it explicitly.
    job_id: str | None = None


class ExplainRasterToolCall(BaseModel):
    """Summary of one tool the agent called while producing the explanation.
    Shown in the UI so users can verify the LLM actually inspected the AOI
    rather than paraphrasing metadata."""

    name: str
    ok: bool
    summary: str | None = None  # brief human-readable result summary


class ExplainRasterResponse(BaseModel):
    explanation: str
    source: str  # "nim" / "claude" / "gemma" / "fallback"
    model: str | None = None  # which specific model ID answered (for debug / attribution)
    tool_calls: list[ExplainRasterToolCall] = []  # agent's real tool usage


def _eager_raster_evidence(
    req: ExplainRasterRequest,
) -> tuple[str | None, dict[str, Any] | None]:
    """Call the raster-reader tool directly so the LLM can't skip it.

    The mcp-builder skill's design guidance: tools whose output the
    answer MUST cite shouldn't be optional. We were handing the model
    ``query_raster_histogram`` as a callable tool, but MiniMax M2.7 and
    similar reasoning models sometimes decline to call tools when they
    think they have enough context from the prompt — they produce a
    confident-sounding "from the 110 ecosystem classes, you'd likely
    see…" answer that's pure hallucination over the bbox.

    Fix: fetch the evidence server-side BEFORE the LLM runs, inject it
    into the user prompt as a fact block, and disable tool-calling on
    the main turn. The LLM's only job is to narrate the facts in plain
    prose. Returns ``(markdown_block, raw_result)`` — the block is
    inlined into the prompt, the raw_result is surfaced as a synthetic
    tool_call entry in the UI's trace so users see what grounded the
    answer.
    """
    if not req.job_id:
        return None, None
    task = (req.task_type or "").lower()
    if task in ("classification", "segmentation"):
        hist = olmoearth_inference.raster_class_histogram(req.job_id, top_n=10)
        if "error" in hist:
            return None, {"name": "query_raster_histogram", "result": hist}
        classes = hist.get("classes") or []
        if not classes:
            return None, {"name": "query_raster_histogram", "result": hist}
        lines = [
            f"- {c['name']} (id={c['index']}): {c['percent']}% ({c['pixel_count']} pixels)"
            for c in classes
        ]
        block = (
            f"PIXEL HISTOGRAM OVER THE AOI (ground truth from the rendered "
            f"raster; {hist.get('total_pixels')} total pixels, "
            f"{hist.get('num_classes_present')} classes present out of "
            f"{hist.get('num_classes_total')}):\n" + "\n".join(lines)
        )
        return block, {"name": "query_raster_histogram", "result": hist}
    if task == "regression":
        stats = olmoearth_inference.raster_scalar_stats(req.job_id)
        if "error" in stats:
            return None, {"name": "query_raster_scalar_stats", "result": stats}
        u = stats.get("units") or ""
        block = (
            "SCALAR STATS OVER THE AOI (ground truth from the rendered "
            f"raster; {stats.get('valid_pixels')} valid pixels):\n"
            f"- mean: {stats.get('mean')} {u}\n"
            f"- min: {stats.get('min')} {u}\n"
            f"- p10: {stats.get('p10')} {u}\n"
            f"- p50 (median): {stats.get('p50')} {u}\n"
            f"- p90: {stats.get('p90')} {u}\n"
            f"- max: {stats.get('max')} {u}"
        )
        return block, {"name": "query_raster_scalar_stats", "result": stats}
    return None, None


def _build_prompt(req: ExplainRasterRequest) -> str:
    """Assemble a single-turn user prompt packed with the raster metadata.

    Keeping the system instruction short (separate ``SYSTEM_PROMPT``
    below) and dropping ALL the raster context into the user turn means
    we can swap providers without re-tuning prompts. The LLM is asked to
    produce 2–3 short paragraphs for a non-expert; the top_classes list
    (when present) is capped at 8 entries so a 110-class ecosystem map
    doesn't blow the context window.
    """
    parts = [f"Model: {req.model_repo_id}"]
    if req.task_type:
        parts.append(f"Task type: {req.task_type}")
    if req.colormap:
        parts.append(f"Colormap key: {req.colormap}")
    if req.bbox:
        parts.append(
            f"AOI bbox (WGS84): west={req.bbox.get('west')}, "
            f"south={req.bbox.get('south')}, east={req.bbox.get('east')}, "
            f"north={req.bbox.get('north')}"
        )
    if req.scene_id or req.scene_datetime:
        scene_bits = []
        if req.scene_id:
            scene_bits.append(f"id={req.scene_id}")
        if req.scene_datetime:
            scene_bits.append(f"date={req.scene_datetime}")
        if req.scene_cloud_cover is not None:
            scene_bits.append(f"cloud={req.scene_cloud_cover:.1f}%")
        parts.append("Sentinel-2 scene: " + " · ".join(scene_bits))
    if req.prediction_value is not None:
        unit_str = f" {req.units}" if req.units else ""
        parts.append(f"Prediction value: {req.prediction_value:.3f}{unit_str}")
    if req.top_classes:
        top_trimmed = req.top_classes[:8]
        lines: list[str] = []
        for c in top_trimmed:
            name = c.get("name", "?")
            idx = c.get("index", "?")
            score = c.get("score")
            score_str = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
            lines.append(f"  - {name} (id={idx}{score_str})")
        parts.append("Top classes on this tile:\n" + "\n".join(lines))
    elif req.class_names:
        # No per-class scores, but a class catalog — include the first few
        # so the LLM knows the output space. Cap at 10.
        head = req.class_names[:10]
        tail = len(req.class_names) - len(head)
        parts.append(
            "Class catalog (first 10 of "
            f"{len(req.class_names)}):\n"
            + "\n".join(f"  - {n}" for n in head)
            + (f"\n  …{tail} more" if tail > 0 else "")
        )
    if req.stub_reason:
        parts.append(
            "IMPORTANT: This raster is a PREVIEW STUB — real inference "
            f"failed with: {req.stub_reason}. The colors are a placeholder "
            "gradient, NOT a true model prediction."
        )

    context = "\n".join(parts)
    return (
        "Explain in 2–3 short paragraphs what this OlmoEarth raster "
        "represents. Cover: (1) what the model does at a high level, "
        "(2) what the colors / classes on the map actually mean for the "
        "user's AOI, (3) any caveats (training-distribution bias, "
        "uncalibrated softmax scores, stub output, etc.). Keep it "
        "accessible to a non-ML-expert earth scientist.\n\n"
        f"Raster metadata:\n{context}"
    )


SYSTEM_PROMPT = (
    "You are Roger, an Earth-observation copilot. The user has clicked a "
    "raster result and wants to know what it actually contains. Ground "
    "your entire answer in tool output — NEVER paraphrase metadata and "
    "NEVER invent ground truth you don't have.\n\n"
    "MANDATORY FIRST STEP: call ``query_raster_histogram`` (for "
    "classification / segmentation) or ``query_raster_scalar_stats`` "
    "(for regression / embedding) before writing anything. This gives "
    "you the REAL pixel distribution over the AOI — which classes "
    "actually appear and in what percentages. Don't describe classes "
    "that aren't in the histogram.\n\n"
    "OPTIONAL SECOND TOOL for extra context (call at most one):\n"
    "- ``query_ndvi_timeseries(bbox)`` — recent Sentinel-2 NDVI over "
    "the AOI; useful to reality-check a vegetation classification.\n"
    "- ``query_olmoearth(bbox)`` — whether an OlmoEarth project region "
    "overlaps the AOI; useful to flag out-of-distribution inference.\n"
    "- ``query_polygon_stats`` — AOI area / elevation if size matters.\n\n"
    "Then write 2–3 short paragraphs covering: (1) what the model does "
    "in one sentence, (2) a concrete summary of what's on THIS tile "
    "citing the histogram percentages (e.g. 'about 48% class X, 23% "
    "class Y'), (3) one caveat that actually applies (training-"
    "distribution bias, uncalibrated softmax, stub output — pick the "
    "most relevant one).\n\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Plain prose only. NO markdown headings (``##``), NO bold "
    "(``**...**``), NO bullet lists, NO 'Paragraph 1:' scaffolding.\n"
    "- Do NOT emit <think> / <thinking> tags — think silently.\n"
    "- Start the first sentence directly with content the user reads. "
    "Finish your answer (don't stop mid-sentence — if you'd exceed the "
    "token budget, write tighter paragraphs)."
)

# Tools the explain-raster agent is allowed to call. ``query_raster_
# histogram`` / ``query_raster_scalar_stats`` are the primary grounding
# tools — they read the actual prediction raster server-side and return
# real pixel counts / stats. The others provide supporting AOI context
# (NDVI timeseries, OlmoEarth coverage, polygon stats, STAC scenes).
# Destructive / stateful tools (run_inference, get_higher_res_patch)
# are excluded — the user already ran inference.
_EXPLAINER_TOOL_NAMES = {
    "query_raster_histogram",
    "query_raster_scalar_stats",
    "query_olmoearth",
    "query_ndvi_timeseries",
    "query_polygon_stats",
    "search_stac_imagery",
}
_EXPLAINER_TOOLS = [
    t
    for t in geo_tools.TOOL_SCHEMAS
    if t.get("function", {}).get("name") in _EXPLAINER_TOOL_NAMES
]


def _scene_context_from_req(req: "ExplainRasterRequest") -> dict[str, Any]:
    """Build the ``scene_context`` the geo_tools dispatcher expects.

    Tools like ``query_olmoearth`` accept an explicit ``bbox`` argument,
    but others fall back to the scene's bbox / area / job_id when the
    LLM forgets to pass one. Packing this into the scene_context means
    the agent can call a tool with zero args and still get a meaningful
    result scoped to the user's AOI + raster.
    """
    ctx: dict[str, Any] = {}
    if req.bbox:
        ctx["area"] = req.bbox
        ctx["bbox"] = req.bbox
    if req.job_id:
        ctx["job_id"] = req.job_id
    return ctx


def _fallback_explanation(req: ExplainRasterRequest) -> str:
    """Templated summary when no LLM is reachable. Honest + informative,
    but not as conversational as an LLM-generated answer."""
    lines: list[str] = []
    task = req.task_type or "inference"
    model = req.model_repo_id.split("/")[-1]
    lines.append(
        f"This layer is a {task} output from {model}. "
    )
    if req.stub_reason:
        lines.append(
            f"Warning: the raster is a PREVIEW STUB — real inference "
            f"failed ({req.stub_reason}). The colors are a placeholder "
            f"gradient, not real model output."
        )
    elif req.task_type == "regression" and req.prediction_value is not None:
        unit_str = f" {req.units}" if req.units else ""
        lines.append(
            f"The mean predicted value over your AOI is "
            f"{req.prediction_value:.2f}{unit_str}. "
            "Brighter / warmer colors indicate higher values; darker "
            "colors indicate lower values."
        )
    elif req.task_type in ("classification", "segmentation") and req.class_names:
        n = len(req.class_names)
        lines.append(
            f"Each pixel is tagged with one of {n} classes; the color "
            "encodes the argmax class id. Colors with higher saturation "
            "typically reflect higher softmax confidence, but scores are "
            "UNCALIBRATED — treat them as rankings, not probabilities."
        )
    else:
        lines.append(
            "The colors encode the raw model output over your AOI. See "
            "the full class list below for details."
        )
    lines.append(
        "(LLM explanation unavailable — paste an NVIDIA NIM key in Cloud "
        "Chat Settings for a free-tier AI-generated explanation, or set "
        "NVIDIA_API_KEY / ANTHROPIC_API_KEY in backend/.env.)"
    )
    return " ".join(lines)


# Matches <think>...</think>, <thinking>...</thinking>, and <reasoning>...
# </reasoning> blocks — the reasoning-model prefix convention that leaked
# the model's planning into the user-facing explanation card. DOTALL so
# the block can span newlines; non-greedy so multiple blocks don't collapse
# into one giant match.
_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Fallback for the "model opened <think> but never closed it" case —
# happens when the model hits the max_tokens limit mid-reasoning and the
# closing tag never shows up. Without this, the regex above misses the
# whole block and the raw chain-of-thought leaks into the UI. Matches
# from the opening tag to end-of-string; applied AFTER the balanced
# regex so a properly-closed block is always preferred.
_UNCLOSED_THINK_RE = re.compile(
    r"<(think|thinking|reasoning)>.*",
    re.DOTALL | re.IGNORECASE,
)

# Kill common meta-narrative openers reasoning models tend to emit AFTER
# the think tag ("Let me write a clear...", "Paragraph 1:", "Okay, here's
# my answer..."). These are signs the model escaped its reasoning bubble
# without actually producing the answer. When detected at the start of
# the reply, we strip up to the first double-newline that looks like
# actual content.
_PLANNING_PREFIX_RE = re.compile(
    r"^\s*(let me (write|draft|provide|give|produce)|okay[,.]?\s*(here'?s|let'?s)|"
    r"here'?s (my |the )?(explanation|answer|response)|"
    r"i'?ll (write|draft|produce))",
    re.IGNORECASE,
)

# "Paragraph N:" / "Paragraph N —" labels the model sometimes emits in
# FRONT of actual content. We strip the prefix inline and keep the body —
# separate from ``_PLANNING_PREFIX_RE`` which drops the whole paragraph.
_PARAGRAPH_LABEL_RE = re.compile(
    r"^\s*paragraph\s*\d+\s*[:\-—]\s*",
    re.IGNORECASE,
)


def _strip_meta(text: str) -> str:
    """Remove reasoning / planning meta-narrative from LLM replies.

    Reasoning models (``minimaxai/minimax-m2.7`` in particular) emit
    ``<think>…</think>`` blocks before their actual answer, and sometimes
    follow with "Let me write… Paragraph 1:… Paragraph 2:…" planning
    that never materializes the paragraphs themselves. Both land the
    user in a wall of chain-of-thought instead of the explanation they
    asked for. This scrub:

      1. Drops every ``<think>`` / ``<thinking>`` / ``<reasoning>`` block.
      2. Trims leading "Let me write / Paragraph 1:" openers by
         advancing to the first non-meta line.
      3. Collapses the remaining whitespace.

    If after scrubbing there's less than ~30 chars of content left, the
    raw input is returned so we don't silently destroy an entire answer
    that just happened to open with a suspect phrase.
    """
    if not text:
        return ""
    # First pass: strip properly-closed <think>…</think> blocks.
    # Second pass: strip any orphan opening <think> tag that never got
    # closed (model ran out of tokens or simply forgot the close). The
    # second pass catches things the first misses — e.g. a response that
    # is 100% reasoning with no user-visible answer after the closing
    # tag, which we want to nuke so the fallback "< 30 chars → return
    # raw" safety net trips correctly below.
    stripped = _THINK_BLOCK_RE.sub("", text)
    stripped = _UNCLOSED_THINK_RE.sub("", stripped).strip()
    # Skip TRUE planning preambles only — "Let me write…", "Okay, here's…",
    # "I'll produce…". These are scaffolding with no content. "Paragraph
    # N:" is labeled content (real answer with a label on top) and is
    # handled below by stripping just the label, not the paragraph.
    for _ in range(3):  # hard cap
        if not _PLANNING_PREFIX_RE.match(stripped):
            break
        parts = stripped.split("\n\n", 1)
        if len(parts) < 2 or len(parts[1].strip()) < 30:
            break
        stripped = parts[1].strip()
    # Strip "Paragraph N:" labels inline, keeping the body. Applied line-
    # by-line because each paragraph can carry its own label.
    lines = [_PARAGRAPH_LABEL_RE.sub("", line) for line in stripped.split("\n")]
    stripped = "\n".join(lines).strip()
    # Markdown syntax safety net — the system prompt says "plain prose
    # only", but reasoning models sometimes still emit ``##`` headers or
    # ``**bold**``. The UI renders text with ``whitespace-pre-wrap`` and
    # no markdown parser, so literal ``##`` and ``**`` show up as junk
    # characters to the user. Strip them server-side:
    #   - ``^## Heading`` and ``^### Heading`` become plain bold prose
    #     lines with the hashes removed.
    #   - ``**word**`` → ``word`` (keep emphasis text, drop the markers).
    #   - Leading list markers ``- `` / ``* `` drop to plain sentences.
    stripped = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", stripped)
    stripped = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
    stripped = re.sub(r"(?m)^\s*[-*]\s+", "", stripped)
    # Collapse runs of 3+ newlines to double newlines (paragraph sep).
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    # Safety net: if the scrub blew away almost everything, the model's
    # "answer" was pure chain-of-thought with nothing user-facing. Return
    # a polite failure string so the UI shows something actionable
    # rather than an empty box or the raw reasoning dump.
    if len(stripped) < 30 <= len(text):
        return (
            "The model didn't produce a finished answer (its response was "
            "all reasoning, no final explanation). Click the pill again to "
            "retry, or switch the provider in Cloud Chat Settings."
        )
    return stripped


def _extract_reply(result: dict[str, Any] | None) -> str:
    """Pull the last assistant text from a ``chat_with_tools`` result.

    Different providers store the final reply under different keys —
    ``reply`` (NIM, Gemma), top-level ``content`` (Claude), or the last
    ``messages[-1]`` entry. Probe in that order so we don't force each
    caller to know the provider's serialization. Scrubs reasoning-model
    ``<think>…</think>`` blocks + meta-narrative openers before
    returning so the UI never renders chain-of-thought as the answer.
    """
    if not result:
        return ""
    raw = ""
    for k in ("reply", "content", "text"):
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            raw = v
            break
    if not raw:
        messages = result.get("messages") or []
        for m in reversed(messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                raw = m["content"]
                break
    return _strip_meta(raw)


def _synthetic_tool_call_to_ui(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize an eager-fetch tool result into the UI ``tool_calls``
    shape. Summary string picks out the top-3 class entries (histogram)
    or the mean + units (scalar) so the UI pill reads as a tight
    one-liner instead of dumping JSON."""
    name = entry.get("name") or "tool"
    res = entry.get("result") or {}
    if "error" in res:
        return {"name": name, "ok": False, "summary": f"error: {res['error']}"}
    if name == "query_raster_histogram":
        classes = res.get("classes") or []
        head = ", ".join(f"{c['name']} {c['percent']}%" for c in classes[:3])
        extra = len(classes) - 3
        summary = head + (f" … (+{extra} more)" if extra > 0 else "")
        return {"name": name, "ok": True, "summary": summary[:200]}
    if name == "query_raster_scalar_stats":
        u = res.get("units") or ""
        summary = (
            f"mean={res.get('mean')}{(' ' + u) if u else ''} · "
            f"range=[{res.get('min')}, {res.get('max')}]"
        )
        return {"name": name, "ok": True, "summary": summary[:200]}
    return {"name": name, "ok": True, "summary": None}


def _extract_tool_calls(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pull the list of tool calls the agent made out of a chat result.

    All three providers surface ``tool_calls_made`` (NIM/Gemma) or embed
    ``tool_use`` blocks inside the assistant messages (Claude). Normalize
    to ``{name, ok, summary}`` dicts so the frontend can render one row
    per tool without caring which provider served the request.
    """
    out: list[dict[str, Any]] = []
    if not result:
        return out
    calls = result.get("tool_calls_made") or result.get("tool_calls") or []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or c.get("tool") or c.get("function", {}).get("name")
        if not name:
            continue
        # Treat explicit "error" keys in the tool result as failure; anything
        # else counts as success for the purpose of the UI indicator.
        resp = c.get("result") or c.get("response") or {}
        ok = not (isinstance(resp, dict) and resp.get("error"))
        summary: str | None = None
        if isinstance(resp, dict):
            # Pick a short human-readable snippet — first string-shaped
            # value under a semantic key, falling back to repr-truncation.
            for k in ("summary", "text", "message", "detail"):
                v = resp.get(k)
                if isinstance(v, str) and v.strip():
                    summary = v.strip()[:200]
                    break
            if not summary:
                # Condense the dict to "key=value · key=value …" for
                # quick at-a-glance inspection.
                flat = ", ".join(
                    f"{k}={str(v)[:40]}" for k, v in list(resp.items())[:4]
                )
                summary = flat[:200] if flat else None
        out.append({"name": name, "ok": ok, "summary": summary})
    return out


def _build_prompt_with_evidence(req: ExplainRasterRequest, evidence: str | None) -> str:
    """Standard metadata prompt + the pre-computed pixel evidence block.

    The evidence block is labeled GROUND TRUTH so the model understands
    it's not speculation. Placed AFTER the metadata so the LLM reads it
    as the most recent, authoritative context before writing.
    """
    base = _build_prompt(req)
    if not evidence:
        return base
    return (
        base
        + "\n\n=== GROUND TRUTH ===\n"
        + evidence
        + "\n\nUse the histogram / stats above to write your explanation. "
          "Cite specific percentages or values. Do NOT invent classes that "
          "aren't in the histogram."
    )


async def _try_nim(
    req: ExplainRasterRequest,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Tool-augmented NIM call. Returns (text, model_id, tool_calls) or None.

    NIM is the PRIMARY provider per the 2026-04 product direction — free-
    tier ``minimaxai/minimax-m2.7`` gets called first so a user with only
    a NVIDIA developer key still sees LLM-backed explanations.
    """
    try:
        # Eagerly fetch the raster histogram / scalar stats and INJECT
        # them into the user prompt before the LLM runs. MiniMax M2.7
        # with tool_choice=auto regularly skipped the histogram call and
        # hallucinated class percentages; making it a fact in the prompt
        # means the LLM has no way to avoid it. The remaining tools
        # (NDVI, OlmoEarth coverage) stay available for optional extras
        # but aren't required for a grounded answer.
        evidence_block, synthetic_call = _eager_raster_evidence(req)
        scene_ctx = _scene_context_from_req(req)
        execute = geo_tools.execute_tool_with_timeout

        async def dispatch(name: str, args: Any) -> dict[str, Any]:
            return await execute(name, args, scene_ctx)

        result = await cloud_llm_client.chat_with_tools(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt_with_evidence(req, evidence_block)},
            ],
            tools=_EXPLAINER_TOOLS,
            execute_tool_fn=dispatch,
            api_key_override=req.nim_api_key,
            temperature=0.3,
            # 700 tokens ≈ 500 words — leaves headroom for 3 tight
            # paragraphs + tool-result thinking without cutting off
            # mid-sentence like the 500-token budget did in testing.
            max_tokens=700,
            max_iterations=4,
            tool_choice="auto",
        )
    except cloud_llm_client.NvidiaNimUnavailable:
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("explain-raster: NIM failed: %s", e)
        return None
    text = _extract_reply(result)
    if not text:
        return None
    model = (result or {}).get("model") or cloud_llm_client.NVIDIA_MODEL
    tool_calls = _extract_tool_calls(result)
    # Prepend the synthetic "eager histogram fetch" entry so the UI
    # trace shows what grounded the answer even though the LLM itself
    # didn't explicitly call the tool.
    if synthetic_call:
        tool_calls.insert(0, _synthetic_tool_call_to_ui(synthetic_call))
    return text, str(model), tool_calls


async def _try_claude(
    req: ExplainRasterRequest,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Tool-augmented Claude fallback. Same tool surface as NIM so the
    LLM provider can swap without changing what the agent has access to."""
    claude_key = req.claude_api_key or req.api_key
    evidence_block, synthetic_call = _eager_raster_evidence(req)
    scene_ctx = _scene_context_from_req(req)
    execute = geo_tools.execute_tool_with_timeout

    async def dispatch(name: str, args: Any) -> dict[str, Any]:
        return await execute(name, args, scene_ctx)

    try:
        result = await claude_client.chat_with_tools(
            messages=[
                {"role": "user", "content": _build_prompt_with_evidence(req, evidence_block)},
            ],
            tools=_EXPLAINER_TOOLS,
            execute_tool_fn=dispatch,
            api_key_override=claude_key,
            system=SYSTEM_PROMPT,
            max_tokens=700,
            max_iterations=4,
            tool_choice="auto",
        )
    except claude_client.ClaudeUnavailable:
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("explain-raster: Claude failed: %s", e)
        return None
    text = _extract_reply(result)
    if not text:
        return None
    model = (result or {}).get("model") or claude_client.CLAUDE_MODEL
    tool_calls = _extract_tool_calls(result)
    if synthetic_call:
        tool_calls.insert(0, _synthetic_tool_call_to_ui(synthetic_call))
    return text, str(model), tool_calls


async def _try_gemma(
    req: ExplainRasterRequest,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Tool-augmented Gemma fallback via the local runtime."""
    evidence_block, synthetic_call = _eager_raster_evidence(req)
    scene_ctx = _scene_context_from_req(req)
    execute = geo_tools.execute_tool_with_timeout

    async def dispatch(name: str, args: Any) -> dict[str, Any]:
        return await execute(name, args, scene_ctx)

    try:
        if not await gemma_client.health_check():
            return None
        result = await gemma_client.chat_with_tools(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt_with_evidence(req, evidence_block)},
            ],
            tools=_EXPLAINER_TOOLS,
            execute_tool_fn=dispatch,
            max_iterations=4,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("explain-raster: Gemma failed: %s", e)
        return None
    text = _extract_reply(result)
    if not text:
        return None
    model = (result or {}).get("model") or gemma_client.GEMMA_MODEL
    tool_calls = _extract_tool_calls(result)
    if synthetic_call:
        tool_calls.insert(0, _synthetic_tool_call_to_ui(synthetic_call))
    return text, str(model), tool_calls


@router.post("/explain-raster", response_model=ExplainRasterResponse)
async def explain_raster(req: ExplainRasterRequest = Body(...)) -> dict[str, Any]:
    """Explain a raster result in natural language via an LLM.

    Tries NIM → Claude → Gemma → deterministic fallback. The ``source``
    field in the response tells the frontend which path won so it can
    show a provider tag + a hint when the fallback fires.
    """
    if not req.model_repo_id:
        raise HTTPException(400, "model_repo_id is required")

    nim = await _try_nim(req)
    if nim:
        text, model, tool_calls = nim
        return {
            "explanation": text,
            "source": "nim",
            "model": model,
            "tool_calls": tool_calls,
        }

    claude = await _try_claude(req)
    if claude:
        text, model, tool_calls = claude
        return {
            "explanation": text,
            "source": "claude",
            "model": model,
            "tool_calls": tool_calls,
        }

    gemma = await _try_gemma(req)
    if gemma:
        text, model, tool_calls = gemma
        return {
            "explanation": text,
            "source": "gemma",
            "model": model,
            "tool_calls": tool_calls,
        }

    return {
        "explanation": _fallback_explanation(req),
        "source": "fallback",
        "model": None,
        "tool_calls": [],
    }
