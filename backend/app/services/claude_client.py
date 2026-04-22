"""Anthropic Claude client for Roger Studio's Claude Chat tab.

Sits alongside ``gemma_client`` (Local) and ``cloud_llm_client`` (NVIDIA NIM)
as a third LLM transport. Uses the official ``anthropic`` Python SDK rather
than raw HTTP — gives us typed exceptions, automatic retries on 429/5xx, and
the SDK's content-block model that's easier to reason about than the raw API
JSON.

Defaults follow the claude-api skill guidance:
  * model: ``claude-opus-4-7`` (highest-capability Anthropic model)
  * adaptive thinking via ``thinking={"type": "adaptive"}`` — Claude decides
    how much to think; ``budget_tokens`` is removed on Opus 4.7 (returns 400)
  * top-level ``cache_control={"type": "ephemeral"}`` for free prompt caching
    on repeat turns (silently no-ops if prefix < 4096 tokens on Opus 4.7)
  * NO ``temperature`` / ``top_p`` / ``top_k`` — also removed on Opus 4.7

Tool schema bridge: ``geo_tools.TOOL_SCHEMAS`` is OpenAI shape (used by both
local Gemma vLLM and NVIDIA NIM). Anthropic uses a flatter shape — we
convert in ``_to_anthropic_tools`` so this client stays geo-agnostic and the
tool registry stays single-sourced.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7").strip()
CLAUDE_TIMEOUT = float(os.environ.get("CLAUDE_TIMEOUT", "120"))


class ClaudeUnavailable(RuntimeError):
    """Raised when Anthropic isn't reachable, the key is missing/invalid, or
    the SDK rejected the request for a model-side reason."""


def set_model(new_model: str) -> None:
    """UI picker hook — same shape as gemma/NIM's ``set_model``."""
    global CLAUDE_MODEL
    new_model = (new_model or "").strip()
    if new_model:
        CLAUDE_MODEL = new_model


def _resolve_key(override: str | None) -> str:
    return (override or "").strip() or CLAUDE_API_KEY


def _to_anthropic_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-shape tool schemas to Anthropic shape.

    OpenAI:    ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Anthropic: ``{"name", "description", "input_schema"}``

    Tools already in Anthropic shape pass through untouched (so server-side
    Anthropic tools like ``{"type": "code_execution_20260120"}`` survive).
    """
    out: list[dict[str, Any]] = []
    for t in openai_tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        else:
            out.append(t)
    return out


async def health_check(api_key_override: str | None = None) -> dict[str, Any]:
    """Best-effort reachability + auth probe. Never raises.

    Hits ``GET /v1/models/{id}`` for the configured model — cheap, returns
    capability metadata, and gives a clean 401/403 if the key is bad.
    """
    key = _resolve_key(api_key_override)
    out: dict[str, Any] = {
        "reachable": False,
        "auth_ok": False,
        "model": CLAUDE_MODEL,
        "api_key_set": bool(key),
        "error": None,
    }
    if not key:
        out["error"] = (
            "ANTHROPIC_API_KEY not configured. Paste a key in Claude Chat "
            "Settings, or set ANTHROPIC_API_KEY in backend/.env."
        )
        return out
    try:
        client = anthropic.AsyncAnthropic(api_key=key, timeout=8.0)
        m = await client.models.retrieve(CLAUDE_MODEL)
        out["reachable"] = True
        out["auth_ok"] = True
        out["model"] = getattr(m, "id", CLAUDE_MODEL)
        out["display_name"] = getattr(m, "display_name", None)
        out["max_input_tokens"] = getattr(m, "max_input_tokens", None)
        out["max_output_tokens"] = getattr(m, "max_tokens", None)
        # Fetch the full list of models the account has access to so the UI
        # can show it as a "live" chip row — parity with NIM / Gemini /
        # OpenAI where /v1/models exposes the same. Best-effort: any error
        # here is swallowed and just means the UI shows an empty live list.
        try:
            listing = await client.models.list(limit=200)
            ids = [getattr(x, "id", None) for x in getattr(listing, "data", [])]
            out["available_models"] = [i for i in ids if i][:100]
        except Exception:  # noqa: BLE001 — live list is an affordance, not required
            pass
    except anthropic.AuthenticationError as e:
        out["reachable"] = True
        out["auth_ok"] = False
        out["error"] = f"Anthropic rejected the API key: {e.message}"
    except anthropic.PermissionDeniedError as e:
        out["reachable"] = True
        out["auth_ok"] = False
        out["error"] = f"API key lacks permission for {CLAUDE_MODEL}: {e.message}"
    except anthropic.NotFoundError as e:
        out["reachable"] = True
        out["auth_ok"] = True
        out["error"] = f"Model {CLAUDE_MODEL} not found: {e.message}"
    except anthropic.APIConnectionError as e:
        out["error"] = f"Cannot reach api.anthropic.com: {e}"
    except Exception as e:  # noqa: BLE001 — health is best-effort
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# Tool-augmented chat loop
#
# Anthropic's wire format differs from OpenAI's — the assistant's tool calls
# are blocks within ``content`` (not a sibling ``tool_calls`` array), and
# tool results come back as user-role blocks of type ``tool_result`` (not a
# dedicated ``role=tool`` message). We translate both directions here so the
# router doesn't need to care which transport it's talking to.
#
# Manual loop (rather than the SDK's beta ``tool_runner``) because we have a
# pre-existing async ``execute_tool_fn`` callback that's shared with the two
# OpenAI-shape transports — letting the SDK auto-execute tools would
# duplicate the registry and split the dispatch path.
# --------------------------------------------------------------------------- #


async def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute_tool_fn: Any,
    api_key_override: str | None = None,
    model_override: str | None = None,
    system: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 5,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Run a tool-augmented chat loop against Anthropic's Messages API.

    Returns a dict with the same keys as ``gemma_client`` /
    ``cloud_llm_client`` so the router can serialize them identically.

    ``messages`` may include a leading ``role: system`` entry (matching the
    OpenAI convention); we strip it and pass it as the ``system`` parameter,
    which is where Anthropic expects it.
    """
    key = _resolve_key(api_key_override)
    if not key:
        raise ClaudeUnavailable(
            "ANTHROPIC_API_KEY not configured. Paste a key in Claude Chat "
            "Settings or export ANTHROPIC_API_KEY in backend/.env."
        )
    model = (model_override or "").strip() or CLAUDE_MODEL

    # Split system prompt out of the message list — Anthropic puts it
    # top-level, not inside ``messages``.
    system_text = system or ""
    cleaned_messages: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            if not system_text:
                content = m.get("content")
                system_text = content if isinstance(content, str) else ""
            continue
        cleaned_messages.append(dict(m))

    anthropic_tools = _to_anthropic_tools(tools)
    anthropic_tool_choice: dict[str, Any] | None
    if tool_choice == "none":
        anthropic_tool_choice = {"type": "none"}
    else:
        anthropic_tool_choice = {"type": "auto"}

    client = anthropic.AsyncAnthropic(api_key=key, timeout=CLAUDE_TIMEOUT)

    history: list[dict[str, Any]] = list(cleaned_messages)
    tool_calls_made: list[dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    last_text = ""
    stopped_reason = "final_answer"
    iteration = 0
    final_stop_reason: str | None = None

    for iteration in range(1, max_iterations + 1):
        try:
            # Top-level ``cache_control`` auto-places on the last cacheable
            # block (tools + system are the most stable prefix). Free if the
            # prefix doesn't reach Opus 4.7's 4096-token minimum — silently
            # no-ops in that case.
            #
            # Adaptive thinking: Claude decides depth per request. NO
            # ``budget_tokens`` (removed on Opus 4.7). NO ``temperature``
            # (also removed on Opus 4.7).
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": history,
                "thinking": {"type": "adaptive"},
                "cache_control": {"type": "ephemeral"},
            }
            if system_text:
                kwargs["system"] = system_text
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
                if anthropic_tool_choice:
                    kwargs["tool_choice"] = anthropic_tool_choice

            response = await client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise ClaudeUnavailable(
                f"Anthropic rejected the API key: {e.message}"
            ) from e
        except anthropic.PermissionDeniedError as e:
            raise ClaudeUnavailable(
                f"API key lacks permission for {model}: {e.message}"
            ) from e
        except anthropic.BadRequestError as e:
            raise ClaudeUnavailable(
                f"Anthropic rejected the request: {e.message}"
            ) from e
        except anthropic.RateLimitError as e:
            raise ClaudeUnavailable(
                f"Anthropic rate limit hit: {e.message}"
            ) from e
        except anthropic.APIConnectionError as e:
            raise ClaudeUnavailable(
                f"Cannot reach api.anthropic.com: {e}"
            ) from e
        except anthropic.APIStatusError as e:
            raise ClaudeUnavailable(
                f"Anthropic returned {e.status_code}: {e.message}"
            ) from e

        usage = getattr(response, "usage", None)
        if usage is not None:
            for field in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            ):
                v = getattr(usage, field, None)
                if isinstance(v, int):
                    total_usage[field] = total_usage.get(field, 0) + v

        # Pull text + tool_use blocks out of the typed content list.
        text_chunks: list[str] = []
        tool_use_blocks: list[Any] = []
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_chunks.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_use_blocks.append(block)
        last_text = "".join(text_chunks)
        final_stop_reason = response.stop_reason

        if not tool_use_blocks:
            # No tool calls this turn — final answer.
            break

        # Echo the assistant turn back into history with the typed content
        # blocks intact (so tool_use IDs survive for matching tool_result).
        history.append({"role": "assistant", "content": response.content})

        tool_results_msg: list[dict[str, Any]] = []
        for tu in tool_use_blocks:
            name = getattr(tu, "name", "") or ""
            tool_input = getattr(tu, "input", None) or {}
            call_id = getattr(tu, "id", "") or f"toolu_{iteration}_{len(tool_calls_made)}"
            try:
                result = await execute_tool_fn(name, tool_input)
            except Exception as e:  # noqa: BLE001 — surface to the model
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
                    "arguments": tool_input,
                    "result": result,
                }
            )
            tool_results_msg.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result, default=str),
                }
            )

        history.append({"role": "user", "content": tool_results_msg})

        logger.info(
            "Claude tool-loop iter %d: %d call(s) — %s",
            iteration,
            len(tool_use_blocks),
            ", ".join(
                c["name"] for c in tool_calls_made[-len(tool_use_blocks) :]
            ),
        )
    else:
        stopped_reason = "max_iterations"
        logger.warning(
            "claude_client.chat_with_tools hit max_iterations=%d", max_iterations
        )

    if final_stop_reason and stopped_reason == "final_answer":
        # Surface Anthropic's stop_reason verbatim when we exited cleanly
        # (end_turn, max_tokens, tool_use, refusal, ...).
        stopped_reason = final_stop_reason

    logger.info(
        "Claude tokens: in=%s out=%s cache_read=%s cache_create=%s across %d iter",
        total_usage.get("input_tokens"),
        total_usage.get("output_tokens"),
        total_usage.get("cache_read_input_tokens"),
        total_usage.get("cache_creation_input_tokens"),
        iteration,
    )

    return {
        "role": "assistant",
        "content": last_text,
        "tool_calls_made": tool_calls_made,
        "iterations": iteration,
        "usage": total_usage,
        "stopped_reason": stopped_reason,
        "model": model,
        "provider": "anthropic",
    }
