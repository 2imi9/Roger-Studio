"""Tool-calling benchmark for the Cuvier Studio LLM stack.

Runs a fixed set of prompts through ``/api/auto-label/gemma/chat`` (Local
Gemma 4 E4B), recording for each case:

  - Did the model call a tool at all?
  - Did it call the EXPECTED tool?
  - Did the arguments include the required keys?
  - Did the loop terminate cleanly (not hit max_iterations)?
  - How many iterations + tokens did it burn?

Cases cover all 7 registered tools:
  query_polygon, query_olmoearth, query_polygon_stats,
  query_ndvi_timeseries (STUB), search_stac_imagery,
  get_composite_tile_url, get_higher_res_patch (STUB)

Plus negative cases (small-talk that should NOT trigger any tool) and one
multi-tool case (planning + execution) to exercise the agentic loop.

References for benchmark design:
  - τ-bench (Sierra Research, 2024) — agentic tool-use eval; tracks
    correct-tool, correct-args, and end-to-end task success
  - Berkeley Function-Calling Leaderboard (BFCL v3) — same trio of metrics
    plus argument-type accuracy
  - ToolBench (xLAM team) — multi-tool planning eval

Run:
    python tests/test_tool_calling_bench.py            # full suite
    python tests/test_tool_calling_bench.py --case 3   # single case
    python tests/test_tool_calling_bench.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import httpx


BACKEND = "http://localhost:8000"
ENDPOINT = f"{BACKEND}/api/auto-label/gemma/chat"


# ---------- shared scene fixtures ---------- #

# A small bbox over the Kenyan coast (matches the existing demo button).
KENYA_BBOX = {"west": 39.6, "south": -4.1, "east": 39.8, "north": -3.9}

# Two sample polygon features for the polygon-lookup tool.
SAMPLE_POLYGONS = [
    {
        "type": "Feature",
        "properties": {"id": "0", "class": "forest", "confidence": 0.42},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [39.62, -4.05], [39.65, -4.05], [39.65, -4.02],
                [39.62, -4.02], [39.62, -4.05],
            ]],
        },
    },
    {
        "type": "Feature",
        "properties": {"id": "1", "class": "water", "confidence": 0.91},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [39.70, -4.00], [39.75, -4.00], [39.75, -3.95],
                [39.70, -3.95], [39.70, -4.00],
            ]],
        },
    },
]

SCENE_BBOX_ONLY = {
    "area": (
        f"bbox W{KENYA_BBOX['west']:.3f} S{KENYA_BBOX['south']:.3f} "
        f"E{KENYA_BBOX['east']:.3f} N{KENYA_BBOX['north']:.3f}"
    ),
    "datasets": [],
    "bbox": KENYA_BBOX,
}

SCENE_WITH_POLYGONS = {
    **SCENE_BBOX_ONLY,
    "polygon_features": SAMPLE_POLYGONS,
    "auto_label_summary": "TIPSv2 ran on 2 polygons (1 forest @ 0.42, 1 water @ 0.91)",
}


@dataclass
class Case:
    name: str
    prompt: str
    scene: dict[str, Any]
    expected_tool: str | None  # None == should NOT call any tool
    required_arg_keys: list[str] = field(default_factory=list)
    notes: str = ""


CASES: list[Case] = [
    # ---------- positive: each of the 7 tools ---------- #
    Case(
        name="query_polygon — direct id lookup",
        prompt=(
            "Polygon 0 in this scene was labelled 'forest' with confidence "
            "0.42. Look it up and tell me what its bbox is."
        ),
        scene=SCENE_WITH_POLYGONS,
        expected_tool="query_polygon",
        required_arg_keys=["polygon_id"],
    ),
    Case(
        name="query_olmoearth — catalog + bbox coverage",
        prompt=(
            "What OlmoEarth datasets and models are available for this area? "
            "Use the catalog tool to check."
        ),
        scene=SCENE_BBOX_ONLY,
        expected_tool="query_olmoearth",
        required_arg_keys=["bbox"],
    ),
    Case(
        name="query_polygon_stats — area + perimeter",
        prompt=(
            "What is the area in square kilometres and perimeter of polygon 1?"
        ),
        scene=SCENE_WITH_POLYGONS,
        expected_tool="query_polygon_stats",
        required_arg_keys=[],  # tool accepts polygon_id OR geometry, neither required
    ),
    Case(
        name="query_ndvi_timeseries — STUB",
        prompt=(
            "Plot the monthly Sentinel-2 NDVI mean over this bbox for the "
            "last 12 months."
        ),
        scene=SCENE_BBOX_ONLY,
        expected_tool="query_ndvi_timeseries",
        required_arg_keys=["bbox", "months"],
        notes="STUB — tool returns not_implemented; agent should report missing data, not invent numbers",
    ),
    Case(
        name="search_stac_imagery — date range",
        prompt=(
            "Find Sentinel-2 scenes over this bbox between 2024-06-01 and "
            "2024-09-01 with cloud cover under 20%."
        ),
        scene=SCENE_BBOX_ONLY,
        expected_tool="search_stac_imagery",
        required_arg_keys=["bbox", "datetime"],
    ),
    Case(
        name="get_composite_tile_url — cloud-free mosaic",
        prompt=(
            "Give me a cloud-free Sentinel-2 true-colour composite tile URL "
            "for this bbox in summer 2024 (June through August)."
        ),
        scene=SCENE_BBOX_ONLY,
        expected_tool="get_composite_tile_url",
        required_arg_keys=["bbox", "datetime"],
    ),
    Case(
        name="get_higher_res_patch — STUB",
        prompt=(
            "Show me a zoomed-in basemap tile for polygon 0 at zoom level 16."
        ),
        scene=SCENE_WITH_POLYGONS,
        expected_tool="get_higher_res_patch",
        required_arg_keys=["polygon_id", "zoom"],
        notes="STUB — tool returns not_implemented",
    ),
    # ---------- negative: should NOT call any tool ---------- #
    Case(
        name="negative — small talk",
        prompt="What's the difference between Sentinel-1 and Sentinel-2 in one sentence?",
        scene=SCENE_BBOX_ONLY,
        expected_tool=None,
        notes="Pure factual recall — no tool call needed",
    ),
    Case(
        name="negative — abstract reasoning",
        prompt=(
            "Explain in two bullets why NDVI saturates over dense vegetation."
        ),
        scene=SCENE_BBOX_ONLY,
        expected_tool=None,
        notes="Pure factual recall — no tool call needed",
    ),
    # ---------- multi-tool / agentic ---------- #
    Case(
        name="agentic — bbox stats then catalog",
        prompt=(
            "First, look up what OlmoEarth datasets cover this bbox. "
            "Then tell me the area in km² of polygon 1."
        ),
        scene=SCENE_WITH_POLYGONS,
        expected_tool="query_olmoearth",  # at minimum the first tool
        required_arg_keys=["bbox"],
        notes="Agentic — should call multiple tools across iterations",
    ),
]


# ---------- runner ---------- #


def run_case(case: Case, timeout: float = 180.0) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": case.prompt}],
        "scene_context": case.scene,
        "tools": "auto",
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(ENDPOINT, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {
            "case": case.name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "duration_s": round(time.monotonic() - t0, 2),
        }
    duration = round(time.monotonic() - t0, 2)

    tool_calls = data.get("tool_calls_made", []) or []
    called_names = [tc["name"] for tc in tool_calls]

    # Score
    if case.expected_tool is None:
        # Negative case — pass iff NO tool was called.
        passed = len(tool_calls) == 0
        verdict = "no-tool (correct)" if passed else f"unexpected tool(s): {called_names}"
        arg_ok: bool | None = None
    else:
        called_correct = case.expected_tool in called_names
        # Find the first invocation of the expected tool to inspect args.
        first = next((tc for tc in tool_calls if tc["name"] == case.expected_tool), None)
        if first is None:
            arg_ok = None
            verdict = f"missed expected tool '{case.expected_tool}', called {called_names or '<none>'}"
            passed = False
        else:
            # The arguments come back as a JSON string from gemma_client; parse.
            try:
                args = json.loads(first.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = first.get("arguments", {})
            if isinstance(args, str):
                # Sometimes double-encoded; try once more.
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            missing = [k for k in case.required_arg_keys if k not in (args or {})]
            arg_ok = len(missing) == 0
            passed = called_correct and arg_ok
            verdict = f"called {case.expected_tool} | missing args: {missing or 'none'}"

    return {
        "case": case.name,
        "ok": passed,
        "verdict": verdict,
        "expected_tool": case.expected_tool,
        "called_tools": called_names,
        "arg_ok": arg_ok,
        "iterations": data.get("iterations"),
        "stopped_reason": data.get("stopped_reason"),
        "usage": data.get("usage", {}),
        "answer_preview": (data.get("content", "") or "")[:200],
        "tool_calls_made": tool_calls,
        "duration_s": duration,
        "notes": case.notes,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("ok"))
    errored = sum(1 for r in results if "error" in r)
    avg_iter = sum((r.get("iterations") or 0) for r in results) / max(1, total)
    total_tokens = sum((r.get("usage") or {}).get("total_tokens", 0) for r in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "errored": errored,
        "pass_rate": round(passed / max(1, total), 3),
        "avg_iterations": round(avg_iter, 2),
        "total_tokens": total_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, help="Run only this case index (0-based)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    cases = [CASES[args.case]] if args.case is not None else CASES
    results: list[dict[str, Any]] = []
    for i, c in enumerate(cases):
        if not args.json:
            print(f"[{i + 1}/{len(cases)}] {c.name} ...", flush=True)
        r = run_case(c, timeout=args.timeout)
        results.append(r)
        if not args.json:
            mark = "PASS" if r.get("ok") else ("ERR " if "error" in r else "FAIL")
            print(f"    {mark}  iter={r.get('iterations')} tools={r.get('called_tools')}  ({r.get('duration_s')}s)")
            if "error" in r:
                print(f"    err: {r['error']}")
            elif not r.get("ok"):
                print(f"    why: {r.get('verdict')}")

    summary = summarize(results)
    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
    else:
        print()
        print("=" * 60)
        print(f"Pass rate: {summary['passed']}/{summary['total']} ({summary['pass_rate'] * 100:.1f}%)")
        print(f"Avg iterations: {summary['avg_iterations']}")
        print(f"Total tokens: {summary['total_tokens']}")
        if summary["errored"]:
            print(f"Errored cases: {summary['errored']}")

    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
