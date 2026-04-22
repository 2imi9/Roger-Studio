"""NVIDIA NIM cloud LLM client for Roger Studio.

Lets the backend dispatch a chat turn to NVIDIA's hosted OpenAI-compatible
endpoint at ``https://integrate.api.nvidia.com/v1`` so the LLM panel can
offer a **Cloud Chat** alongside the existing **Local Chat** (vLLM / Ollama
in ``gemma_client``). Pattern is borrowed from sibling project O3earth's
``CloudLLMClient`` and adapted to Roger's async httpx tool loop.

NIM speaks the OpenAI schema, so tool-calling just works with the same
``TOOL_SCHEMAS`` dispatched to Gemma — no adapter needed.

API key resolution order (first non-empty wins):
    1. ``api_key_override`` passed per-request (the UI pastes a key into
       sessionStorage and forwards it each call — never persisted server-side)
    2. ``NVIDIA_API_KEY`` environment variable (for server-owned deployments)

Never logged; the request header is constructed locally and not echoed.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from app.services._phantom_tool_detect import (
    looks_phantom,
    was_phantom_nudged,
    PHANTOM_RETRY_NUDGE,
)

logger = logging.getLogger(__name__)


NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
# MiniMax M2.7 is the default because a live April-2026 test against
# /v1/chat/completions showed it gave the cleanest tool-calling behavior
# on the free NIM tier: no phantom narration, correct argument shapes,
# reliable finish_reason=tool_calls. The previous default
# (openai/gpt-oss-20b) sometimes throttles under load, breaking the
# first-run experience. Override via the ``NVIDIA_MODEL`` env var or the
# Cloud Chat → NIM UI picker.
# Default NIM model for generic tool-calling / reasoning chats. MiniMax
# M2.7 is the best free-tier NIM for agentic workflows (tight tool-call
# format compliance, strong reasoning) — kept as the default for the
# main chat surface. Callers that need a MULTIMODAL model (e.g. the
# raster-explainer, which attaches a rendered tile PNG to the user
# message) pass ``model_override="mistralai/mistral-large-3-675b-
# instruct-2512"`` on a per-call basis rather than swapping the global.
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "minimaxai/minimax-m2.7")

# Multimodal NIM model for the raster-explainer path. Free-tier vision-
# language model ("Image-to-Text") — used only when we have an actual
# tile PNG to attach. Env-overridable for future model swaps without
# code changes.
NVIDIA_MULTIMODAL_MODEL = os.environ.get(
    "NVIDIA_MULTIMODAL_MODEL", "mistralai/mistral-large-3-675b-instruct-2512"
)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_TIMEOUT = float(os.environ.get("NVIDIA_TIMEOUT", "120"))


class NvidiaNimUnavailable(RuntimeError):
    """Raised when the NIM endpoint is unreachable or auth fails."""


def set_model(new_model: str) -> None:
    """Update the default NIM model at runtime (driven by the UI picker)."""
    global NVIDIA_MODEL
    new_model = (new_model or "").strip()
    if new_model:
        NVIDIA_MODEL = new_model


def _resolve_key(override: str | None) -> str:
    """Per-call override wins; fall back to the process env var."""
    key = (override or "").strip()
    return key or NVIDIA_API_KEY


async def health_check(api_key_override: str | None = None) -> dict[str, Any]:
    """Return reachability + auth status for the NIM endpoint.

    Never raises — always returns a dict with ``reachable`` / ``auth_ok`` /
    ``error``. Used by the frontend status badge in the Cloud Chat tab.
    """
    key = _resolve_key(api_key_override)
    out: dict[str, Any] = {
        "reachable": False,
        "auth_ok": False,
        "model": NVIDIA_MODEL,
        "base_url": NVIDIA_BASE_URL,
        "api_key_set": bool(key),
        "error": None,
    }
    if not key:
        out["error"] = (
            "NVIDIA_API_KEY not configured. Paste a key in Cloud Chat Settings, "
            "or set NVIDIA_API_KEY in backend/.env."
        )
        return out
    try:
        async with httpx.AsyncClient(
            timeout=8.0,
            headers={"Authorization": f"Bearer {key}"},
        ) as client:
            r = await client.get(f"{NVIDIA_BASE_URL}/models")
            r.raise_for_status()
            data = r.json()
        out["reachable"] = True
        out["auth_ok"] = True
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            ids = [m.get("id", "") for m in data["data"] if isinstance(m, dict)]
            out["available_models"] = [i for i in ids if i][:100]
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        out["reachable"] = True
        if code in (401, 403):
            out["auth_ok"] = False
            out["error"] = f"NIM rejected the API key (HTTP {code}). Rotate the key or check scopes."
        else:
            out["error"] = f"HTTP {code}: {e.response.text[:200]}"
    except httpx.HTTPError as e:
        out["error"] = f"Network error reaching NIM: {type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 — health is best-effort
        out["error"] = f"{type(e).__name__}: {e}"
    return out


async def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute_tool_fn: Any,
    api_key_override: str | None = None,
    model_override: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 800,
    max_iterations: int = 5,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Run a tool-augmented chat loop against NVIDIA NIM.

    Mirrors the return shape of ``gemma_client.chat_with_tools`` so the
    router can stitch either backend response into the same UI payload.
    """
    key = _resolve_key(api_key_override)
    if not key:
        raise NvidiaNimUnavailable(
            "NVIDIA_API_KEY is not set — paste a key in Cloud Chat Settings "
            "or export NVIDIA_API_KEY in backend/.env."
        )
    model = (model_override or "").strip() or NVIDIA_MODEL

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    history: list[dict[str, Any]] = [dict(m) for m in messages]
    tool_calls_made: list[dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    stopped_reason = "final_answer"
    last_text = ""
    last_usage: dict[str, Any] = {}
    iteration = 0

    async with httpx.AsyncClient(timeout=NVIDIA_TIMEOUT, headers=headers) as client:
        for iteration in range(1, max_iterations + 1):
            payload: dict[str, Any] = {
                "model": model,
                "messages": history,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools and tool_choice != "none":
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice

            try:
                r = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions", json=payload
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                detail = e.response.text[:400]
                if code in (401, 403):
                    raise NvidiaNimUnavailable(
                        f"NIM rejected the API key (HTTP {code}). "
                        "Rotate it at build.nvidia.com, or check that the key "
                        "has access to the selected model."
                    ) from e
                raise NvidiaNimUnavailable(
                    f"NIM returned {code}: {detail}"
                ) from e
            except httpx.HTTPError as e:
                raise NvidiaNimUnavailable(
                    f"Cannot reach NIM at {NVIDIA_BASE_URL}: {type(e).__name__}: {e}"
                ) from e

            choice = data["choices"][0]
            msg = choice["message"]
            usage = data.get("usage", {}) or {}
            for k, v in usage.items():
                if isinstance(v, int):
                    total_usage[k] = total_usage.get(k, 0) + v
            last_usage = usage
            last_text = msg.get("content") or ""

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Phantom-tool-call guard: the model sometimes narrates a
                # tool execution (or fabricates a fake job_id / pixel
                # histogram) without emitting tool_calls. If so, inject one
                # retry nudge and loop again before giving up.
                phantom = looks_phantom(last_text)
                # Hard cap: max ONE phantom-nudge retry per chat turn. The
                # ``was_phantom_nudged`` helper centralizes the check so
                # a future strategy change lands once across all clients.
                already_nudged = was_phantom_nudged(history)
                if phantom is not None and tools and tool_choice != "none" and not already_nudged:
                    logger.warning(
                        "phantom tool call detected (phrase=%r) — injecting retry nudge",
                        phantom.group(0),
                    )
                    history.append({"role": "assistant", "content": last_text})
                    history.append({"role": "system", "content": PHANTOM_RETRY_NUDGE})
                    stopped_reason = "phantom_tool_call_retry"
                    continue
                break

            history.append(
                {
                    "role": "assistant",
                    "content": last_text,
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                arguments = fn.get("arguments", "{}")
                call_id = (
                    call.get("id") or f"call_{iteration}_{len(tool_calls_made)}"
                )
                try:
                    result = await execute_tool_fn(name, arguments)
                except Exception as e:  # noqa: BLE001 — surface to model
                    logger.exception("Tool callback raised for %s", name)
                    result = {
                        "error": "tool_callback_raised",
                        "name": name,
                        "detail": str(e),
                    }
                tool_calls_made.append(
                    {
                        "id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

            logger.info(
                "NIM tool-loop iter %d: %d call(s) — %s",
                iteration,
                len(tool_calls),
                ", ".join(
                    c["name"] for c in tool_calls_made[-len(tool_calls) :]
                ),
            )
        else:
            stopped_reason = "max_iterations"
            logger.warning("cloud chat_with_tools hit max_iterations=%d", max_iterations)

    logger.info(
        "NIM tokens: %s prompt / %s total across %d iter",
        total_usage.get("prompt_tokens"),
        total_usage.get("total_tokens"),
        iteration,
    )

    return {
        "role": "assistant",
        "content": last_text,
        "tool_calls_made": tool_calls_made,
        "iterations": iteration,
        "usage": total_usage or last_usage,
        "stopped_reason": stopped_reason,
        "model": model,
        "provider": "nvidia_nim",
    }
