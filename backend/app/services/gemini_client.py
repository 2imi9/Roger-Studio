"""Google Gemini cloud LLM client for Roger Studio.

Dispatches chat turns to Google AI Studio's **OpenAI-compatible** endpoint
at ``https://generativelanguage.googleapis.com/v1beta/openai`` so we can
reuse the same ``TOOL_SCHEMAS`` + tool-loop pattern already used for NIM
(``cloud_llm_client``) and local Gemma (``gemma_client``) — no special
adapter needed.

Why the OpenAI-compat path (not Google's native REST)?
  - Same payload shape (messages, tools, tool_choice, temperature, …).
  - Same response shape (choices[0].message.tool_calls[]).
  - One httpx POST, no SDK dep, no bespoke adapter.

API key resolution (first non-empty wins):
    1. ``api_key_override`` passed per-request (UI pastes a key into
       sessionStorage and forwards it each call — never persisted server-side)
    2. ``GEMINI_API_KEY`` env var / ``GOOGLE_API_KEY`` env var (for
       server-owned deployments)

Model IDs follow Google's convention: ``gemini-2.5-pro``, ``gemini-2.5-flash``,
``gemini-2.5-flash-lite``, ``gemini-2.0-flash``. Pick in the UI picker;
the backend runtime-updates via ``set_model``.
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


GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY", "").strip()
    or os.environ.get("GOOGLE_API_KEY", "").strip()
)
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "120"))


class GeminiUnavailable(RuntimeError):
    """Raised when the Gemini endpoint is unreachable or auth fails."""


def set_model(new_model: str) -> None:
    """Update the default Gemini model at runtime (driven by the UI picker)."""
    global GEMINI_MODEL
    cleaned = (new_model or "").strip()
    if cleaned:
        GEMINI_MODEL = cleaned


def _resolve_key(override: str | None) -> str:
    key = (override or "").strip()
    return key or GEMINI_API_KEY


async def health_check(api_key_override: str | None = None) -> dict[str, Any]:
    """Return reachability + auth status for the Gemini endpoint. Never raises."""
    key = _resolve_key(api_key_override)
    out: dict[str, Any] = {
        "reachable": False,
        "auth_ok": False,
        "model": GEMINI_MODEL,
        "base_url": GEMINI_BASE_URL,
        "api_key_set": bool(key),
        "error": None,
    }
    if not key:
        out["error"] = (
            "GEMINI_API_KEY not configured. Paste a key in Cloud Chat → Gemini, "
            "or set GEMINI_API_KEY / GOOGLE_API_KEY in backend/.env. "
            "Get one at https://aistudio.google.com/apikey."
        )
        return out
    try:
        async with httpx.AsyncClient(
            timeout=8.0,
            headers={"Authorization": f"Bearer {key}"},
        ) as client:
            r = await client.get(f"{GEMINI_BASE_URL}/models")
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
            out["error"] = (
                f"Gemini rejected the API key (HTTP {code}). Rotate at "
                "aistudio.google.com/apikey or check scopes."
            )
        else:
            out["error"] = f"HTTP {code}: {e.response.text[:200]}"
    except httpx.HTTPError as e:
        out["error"] = f"Network error reaching Gemini: {type(e).__name__}: {e}"
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
    max_tokens: int = 1500,
    max_iterations: int = 5,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Run a tool-augmented chat loop against Google Gemini.

    Mirrors the return shape of ``gemma_client.chat_with_tools`` and
    ``cloud_llm_client.chat_with_tools`` so the Sidebar chat hub can route
    either backend response into the same UI payload (with artifacts,
    phantom-retry, empty-retry flags, etc.).
    """
    key = _resolve_key(api_key_override)
    if not key:
        raise GeminiUnavailable(
            "GEMINI_API_KEY is not set — paste a key in Cloud Chat → Gemini "
            "or export GEMINI_API_KEY in backend/.env. "
            "Get one at https://aistudio.google.com/apikey."
        )
    model = (model_override or "").strip() or GEMINI_MODEL

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

    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT, headers=headers) as client:
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
                    f"{GEMINI_BASE_URL}/chat/completions", json=payload
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                detail = e.response.text[:400]
                if code in (401, 403):
                    raise GeminiUnavailable(
                        f"Gemini rejected the API key (HTTP {code}). "
                        "Rotate at aistudio.google.com/apikey, or check that "
                        "the key has access to the selected model."
                    ) from e
                raise GeminiUnavailable(
                    f"Gemini returned {code}: {detail}"
                ) from e
            except httpx.HTTPError as e:
                raise GeminiUnavailable(
                    f"Cannot reach Gemini at {GEMINI_BASE_URL}: "
                    f"{type(e).__name__}: {e}"
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
                # Hard cap: max ONE phantom-nudge retry per chat turn (shared helper).
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
                    logger.exception("Gemini tool callback raised for %s", name)
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
                "Gemini tool-loop iter %d: %d call(s) — %s",
                iteration,
                len(tool_calls),
                ", ".join(c["name"] for c in tool_calls_made[-len(tool_calls):]),
            )
        else:
            stopped_reason = "max_iterations"
            logger.warning(
                "Gemini chat_with_tools hit max_iterations=%d", max_iterations,
            )

    logger.info(
        "Gemini tokens: %s prompt / %s total across %d iter",
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
    }
