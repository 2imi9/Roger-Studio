"""LLM client for the Roger Studio backend.

Speaks the OpenAI-compatible /v1/chat/completions protocol, which lets the
backend talk to any of three runtimes interchangeably — picked via the
`LLM_RUNTIME` env var (or auto-detected from `GEMMA_BASE_URL`):

  ollama  → local, `ollama pull` / `ollama ps` / `ollama stop` lifecycle
  vllm    → local, externally-managed long-running `vllm serve` or docker
  cloud   → OpenRouter, OpenAI, Anthropic, Google AI Studio, NVIDIA NIM, ...

The default is vLLM (docker) serving `unsloth/gemma-4-e4b-it` on :8001 —
~8 GB VRAM, fits any consumer GPU, ungated (no HF token required). Larger
Gemma 4 variants (26B A4B, etc.) are one click away in the UI model picker
but require more VRAM and/or an HF token for the gated Google repos.

This module is the thin I/O layer + local-runner lifecycle. The orchestration
and prompt engineering live in geo_agent.py so every geo framework (TIPSv2,
SamGeo, Spectral, OlmoEarth, Weather, Elevation) reaches the model through a
single choke point regardless of which runtime is configured.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any

import httpx

# Phrases that small tool-use models emit when they WANT to fire a tool but
# forget to actually emit the tool_calls field. If we see any of these in an
# assistant reply with no tool_calls, we treat the turn as a "phantom tool
# call" and auto-retry with a stricter nudge. Case-insensitive regex.
_PHANTOM_TOOL_CALL_PATTERNS = re.compile(
    r"\b("
    r"i\s+have\s+(initiated|started|begun|launched|kicked\s+off|triggered)"
    r"|i'?ll\s+(notify|let\s+you\s+know|update\s+you)"
    r"|please\s+stand\s+by"
    r"|please\s+wait"
    r"|the\s+process\s+(is\s+running|has\s+begun|has\s+started)"
    r"|running\s+the\s+(model|inference|tool)"
    r"|inference\s+is\s+(running|now\s+in\s+progress|underway)"
    r"|(process|inference|tool)\s+is\s+now\s+running"
    r"|i\s+will\s+(notify|update|provide\s+you)"
    r"|stand\s+by"
    r"|once\s+(the|it'?s)\s+(inference|process|tool|prediction|complete)"
    r"|tool\s+execution\s+failed\s+on\s+the\s+backend"
    r"|endpoint\s+(for\s+)?the\s+inference\s+call\s+was\s+not\s+found"
    r"|temporary\s+issue\s+with\s+the\s+service"
    r")\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# LLM_RUNTIME selects how we manage the local model:
#   "ollama" (default) — lifecycle via `ollama pull` / `ollama ps`. Zero config.
#   "vllm"             — vLLM is long-running (user starts `vllm serve` or
#                        `docker run vllm/vllm-openai:nightly` themselves).
#                        Backend only does health + reachability checks.
#   "cloud"            — GEMMA_BASE_URL points at OpenRouter/OpenAI/etc.
#                        Lifecycle helpers are no-ops; only health matters.
# Auto-detect: if GEMMA_BASE_URL points at a non-localhost host, treat as cloud.
_LLM_RUNTIME_ENV = os.environ.get("LLM_RUNTIME", "").strip().lower()


# Default points at vLLM (docker in WSL) serving Gemma 4 E4B (Unsloth variant).
# Port is 8001 (not 8000) because our FastAPI backend already owns 8000 —
# the container's internal 8000 is mapped to host 8001 with -p 8001:8000.
# Why E4B as the default:
#   - ~8 GB VRAM at FP16 — fits any modern consumer GPU (RTX 3060 and up).
#   - Ungated: Unsloth mirrors don't need an HF token, so first-run works
#     without a license-accept roundtrip.
#   - Text + image multimodal, same tool-calling / reasoning parsers as the
#     bigger Gemma 4 A4B MoE variants.
# Users with more VRAM (RTX 5090 class) can swap to unsloth/gemma-4-26b-a4b-it-bnb-4bit
# (~13 GB) or the full unsloth/gemma-4-26b-a4b-it (~52 GB) via the UI picker.
# For fully managed cloud, set LLM_RUNTIME=cloud + GEMMA_API_KEY + cloud base URL.
GEMMA_BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://localhost:8001/v1")

# Resolved-once cache for the model name the server actually reports from
# /v1/models. Lets us tolerate `unsloth/gemma-4-e4b-it` vs `google/gemma-4-e4b-it`
# vs user-picked variants without forcing the operator to thread the exact
# string through GEMMA_MODEL at every startup. Cleared when the server
# returns 404 on the currently-resolved name (model was swapped).
_resolved_served_model: str | None = None
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "unsloth/gemma-4-e4b-it")


def set_model(new_model: str) -> None:
    """Update GEMMA_MODEL at runtime (used by the UI model picker).
    Keeps .env unchanged; lives until next backend restart.
    """
    global GEMMA_MODEL, _resolved_served_model
    GEMMA_MODEL = new_model.strip() or GEMMA_MODEL
    # Invalidate the auto-resolved name so the next request re-queries /v1/models.
    _resolved_served_model = None
GEMMA_TIMEOUT = float(os.environ.get("GEMMA_TIMEOUT", "120"))
GEMMA_PORT = int(os.environ.get("GEMMA_PORT", "8001"))
# Optional bearer token for cloud providers. Empty for local Ollama.
GEMMA_API_KEY = os.environ.get("GEMMA_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Kept for backward compat so an older .env pointing at Docker Model Runner
# or a custom vLLM image still parses. None of these are used by the default
# Ollama path.
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "geoenv-vllm:latest")
VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", "geoenv-llm")
VLLM_DOCKERFILE_DIR = os.environ.get(
    "VLLM_DOCKERFILE_DIR",
    str(os.path.join(os.path.dirname(__file__), "..", "..", "docker")),
)
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR",
    os.path.expanduser("~/.cache/huggingface"),
)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _resolve_runtime() -> str:
    """Decide which runtime to drive based on LLM_RUNTIME + GEMMA_BASE_URL.

    Explicit LLM_RUNTIME wins. Otherwise, infer from base URL:
      - non-localhost host → cloud
      - port 11434 (Ollama default) → ollama
      - anything else → vllm (the remaining local option)
    """
    if _LLM_RUNTIME_ENV in {"ollama", "vllm", "cloud"}:
        return _LLM_RUNTIME_ENV
    base = GEMMA_BASE_URL.lower()
    if "localhost" not in base and "127.0.0.1" not in base:
        return "cloud"
    if ":11434" in base:
        return "ollama"
    return "vllm"


LLM_RUNTIME = _resolve_runtime()

# Tracks the most recent `docker run` subprocess so we can surface late
# failures (the command may take >3s to fail on Windows Docker Desktop).
_last_start_proc: subprocess.Popen | None = None
_last_start_error: str | None = None


class GemmaUnavailable(RuntimeError):
    """Raised when the vLLM server is not reachable."""


def _encode_image_png(image: Any) -> str:
    """PIL Image -> base64 PNG data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _auth_headers() -> dict[str, str]:
    """Return Authorization header if GEMMA_API_KEY is set, else empty."""
    if GEMMA_API_KEY:
        return {"Authorization": f"Bearer {GEMMA_API_KEY}"}
    return {}


async def health_check() -> bool:
    """Ping the configured OpenAI-compatible endpoint. Returns True if reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers()) as client:
            r = await client.get(f"{GEMMA_BASE_URL}/models")
            r.raise_for_status()
            models = r.json().get("data", [])
            return any(GEMMA_MODEL in m.get("id", "") for m in models) or len(models) > 0
    except Exception as e:
        logger.warning(f"LLM health check failed ({GEMMA_BASE_URL}): {e}")
        return False


async def chat_with_vision(
    system_prompt: str,
    user_prompt: str,
    images: list[Any] | None = None,
    response_format: str = "json_object",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a multimodal chat request to Gemma 4.

    Args:
        system_prompt: System instructions (geoscientist role).
        user_prompt: User task text.
        images: list of PIL images to attach (max 4 per vLLM config).
        response_format: "json_object" to force structured JSON, "text" otherwise.
        temperature: sampling temp (low for deterministic validation).
        max_tokens: cap on response length.

    Returns:
        Parsed JSON dict (when response_format=json_object) or {"text": str}.

    Raises:
        GemmaUnavailable: if the vLLM server is unreachable.
    """
    if images is None:
        images = []

    # Gemma 4 modality order: images BEFORE text (model card §4 / google-deepmind
    # gemma ChatSampler convention). This measurably improves vision grounding.
    content: list[dict[str, Any]] = []
    for img in images[:4]:  # backend limit-mm-per-prompt
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_png(img)},
        })
    content.append({"type": "text", "text": user_prompt})

    served_model = await _resolve_served_model_name()
    payload = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=GEMMA_TIMEOUT, headers=_auth_headers()) as client:
            r = await client.post(
                f"{GEMMA_BASE_URL}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        raise GemmaUnavailable(
            f"Cannot reach LLM at {GEMMA_BASE_URL}. "
            "Click Start in the LLM panel to launch the local model, or set "
            "GEMMA_BASE_URL + GEMMA_API_KEY for a cloud provider."
        ) from e
    except httpx.HTTPStatusError as e:
        raise GemmaUnavailable(f"Gemma returned {e.response.status_code}: {e.response.text[:300]}") from e

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    logger.info(f"Gemma tokens: {usage.get('total_tokens')} (prompt {usage.get('prompt_tokens')})")

    if response_format == "json_object":
        try:
            parsed = json.loads(text)
            parsed["_usage"] = usage
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"Gemma returned non-JSON despite json_object mode: {text[:200]}")
            return {"raw": text, "_parse_error": True, "_usage": usage}

    return {"text": text, "_usage": usage}


# ---------------------------------------------------------------------------
# Tool-calling loop
#
# vLLM exposes OpenAI-style function calling when started with
# `--enable-auto-tool-choice --tool-call-parser gemma4`. Assistant turns may
# carry a ``tool_calls`` array; each entry has ``id``, ``type="function"``,
# and ``function: {name, arguments}`` where ``arguments`` is a JSON string.
# We run one tool call per entry, append ``role=tool`` replies to the message
# list, and re-query until the model produces a turn with no tool_calls (the
# final answer) or we hit ``max_iterations``.
#
# ``execute_tool_fn`` is passed in as a callback so this module stays free of
# geo-specific imports — geo_tools.execute_tool is what the chat router wires
# in. Each call gets (name, arguments_json_str) and must return a
# JSON-serializable dict.
# ---------------------------------------------------------------------------


ToolExecutor = Any  # Callable[[str, str], Awaitable[dict]]


async def _resolve_served_model_name() -> str:
    """Return the model id the LLM server is actually serving.

    Preference order:
      1. Cached result from a previous call in the same process.
      2. GEMMA_MODEL env var, if the server actually serves it (verified via
         /v1/models).
      3. The first model id returned by /v1/models.
      4. GEMMA_MODEL env var as a last-resort fallback (let the request fail
         with a clear 404 so the operator sees the mismatch).

    This lets the operator swap models in the docker / vllm-serve command
    without also updating GEMMA_MODEL — a common foot-gun that surfaces as
    a 404 "model does not exist" once the user tries to chat.
    """
    global _resolved_served_model
    if _resolved_served_model:
        return _resolved_served_model
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers()) as client:
            r = await client.get(f"{GEMMA_BASE_URL}/models")
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        logger.warning(
            "/v1/models lookup failed (%s); falling back to GEMMA_MODEL=%s",
            type(e).__name__, GEMMA_MODEL,
        )
        return GEMMA_MODEL

    served = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    if not served:
        return GEMMA_MODEL
    # Prefer the configured model if the server actually serves it.
    if GEMMA_MODEL in served:
        _resolved_served_model = GEMMA_MODEL
    else:
        _resolved_served_model = served[0]
        if GEMMA_MODEL != _resolved_served_model:
            logger.info(
                "GEMMA_MODEL=%r not served; auto-resolved to %r (first of %d served)",
                GEMMA_MODEL, _resolved_served_model, len(served),
            )
    return _resolved_served_model


async def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute_tool_fn: ToolExecutor,
    temperature: float = 0.4,
    max_tokens: int = 800,
    max_iterations: int = 5,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Run a tool-augmented chat loop against the configured LLM.

    Args:
        messages: OpenAI-format chat history. Usually opens with a system
            message and ends with the latest user turn.
        tools: OpenAI function-calling tool schemas (see geo_tools.TOOL_SCHEMAS).
        execute_tool_fn: ``async (name: str, arguments_json: str) -> dict``.
            Raising is fine — the exception is caught and surfaced to the
            model as a structured ``role=tool`` error so it can recover.
        temperature / max_tokens: per-turn sampling settings.
        max_iterations: safety cap on the call → tool-reply → call loop.
        tool_choice: "auto" (model decides) or "none" (disable tools).

    Returns:
        ``{"role": "assistant", "content": str, "tool_calls_made": [...],
           "iterations": int, "usage": {...}, "stopped_reason": str}``

    Raises:
        GemmaUnavailable: if the LLM endpoint is unreachable at any step.
    """
    # Copy so we don't mutate the caller's list
    history: list[dict[str, Any]] = [dict(m) for m in messages]
    tool_calls_made: list[dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    stopped_reason = "final_answer"
    last_text = ""
    last_reasoning = ""
    last_usage: dict[str, Any] = {}
    iteration = 0
    # Flags surfaced to the caller so the UI can render warning chips:
    #   phantom_tool_call_fixed → one auto-retry took the model from narrative
    #                             to a real tool_call. Useful signal that the
    #                             small model is unstable on tool use.
    #   empty_retry_fixed       → reply was empty; single lower-temp retry
    #                             produced a valid answer.
    phantom_tool_call_fixed = False
    empty_retry_fixed = False
    empty_retry_used = False

    for iteration in range(1, max_iterations + 1):
        served_model = await _resolve_served_model_name()
        payload: dict[str, Any] = {
            "model": served_model,
            "messages": history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        try:
            async with httpx.AsyncClient(timeout=GEMMA_TIMEOUT, headers=_auth_headers()) as client:
                r = await client.post(f"{GEMMA_BASE_URL}/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError as e:
            raise GemmaUnavailable(
                f"Cannot reach LLM at {GEMMA_BASE_URL}. Start the local model "
                "from the LLM panel or set GEMMA_BASE_URL + GEMMA_API_KEY for "
                "a cloud provider."
            ) from e
        except httpx.HTTPStatusError as e:
            # 404 on "model" means the server swapped models since we resolved
            # its name. Clear the cache and retry once on the next iteration.
            if e.response.status_code == 404 and "does not exist" in e.response.text:
                global _resolved_served_model
                _resolved_served_model = None
                logger.warning(
                    "model %r not served anymore; re-resolving on next turn",
                    served_model,
                )
            raise GemmaUnavailable(
                f"LLM returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e

        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {}) or {}
        for k, v in usage.items():
            if isinstance(v, int):
                total_usage[k] = total_usage.get(k, 0) + v
        last_usage = usage
        last_text = msg.get("content") or ""
        last_reasoning = msg.get("reasoning_content") or ""

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Phantom-tool-call detection: the model described running a tool
            # but didn't emit the tool_calls field. Small tool-use models
            # (Gemma 4 E4B especially) do this routinely — they narrate
            # "I have initiated the inference" instead of firing the call.
            # One auto-retry with an in-line system nudge usually fixes it.
            phantom = _PHANTOM_TOOL_CALL_PATTERNS.search(last_text or "")
            if (
                phantom is not None
                and tool_choice != "none"
                and tools
                and not any(
                    m.get("role") == "system"
                    and "PHANTOM TOOL CALL DETECTED" in (m.get("content") or "")
                    for m in history
                )
            ):
                phantom_tool_call_fixed = True  # flipped back to False if the retry still fails
                logger.warning(
                    "phantom tool call detected (phrase=%r) — injecting retry nudge",
                    phantom.group(0),
                )
                history.append({
                    "role": "assistant",
                    "content": last_text,
                })
                history.append({
                    "role": "system",
                    "content": (
                        "PHANTOM TOOL CALL DETECTED. Your previous reply claimed "
                        "a tool was running but did not emit a tool_calls field, "
                        "so no tool actually ran. Re-read the user's most recent "
                        "request. If the request needs a tool, emit the real "
                        "tool_calls entry now — do not describe the tool in "
                        "prose, do not apologize, do not promise a future update."
                    ),
                })
                stopped_reason = "phantom_tool_call_retry"
                continue  # loop will re-request with the nudge appended

            # Empty-content retry: model returned nothing useful — no
            # tool_calls, no visible content, AND no reasoning block. This
            # narrower check avoids firing on normal short answers where
            # Gemma put everything in <think>...</think> and the
            # --reasoning-parser gemma4 split it out as reasoning_content
            # with a short content field. We only retry when the turn is
            # genuinely blank on all three channels.
            if (
                not last_text.strip()
                and not (last_reasoning or "").strip()
                and not empty_retry_used
            ):
                empty_retry_used = True
                logger.warning("empty model reply — retrying once at temperature=0.3")
                history.append({
                    "role": "system",
                    "content": (
                        "Your previous reply was empty. Answer the user in "
                        "plain text, or emit a tool_call if a tool is needed. "
                        "Do not return an empty message."
                    ),
                })
                temperature = 0.3  # local override for the retry turn
                stopped_reason = "empty_retry"
                continue

            if empty_retry_used and last_text.strip():
                empty_retry_fixed = True
            break  # final answer

        # Echo the assistant's tool-calling turn into history exactly as received
        history.append({
            "role": "assistant",
            "content": last_text,
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = (call.get("function") or {})
            name = fn.get("name", "")
            arguments = fn.get("arguments", "{}")
            call_id = call.get("id") or f"call_{iteration}_{len(tool_calls_made)}"

            try:
                result = await execute_tool_fn(name, arguments)
            except Exception as e:  # noqa: BLE001 — surfaced to model, not re-raised
                logger.exception("Tool callback raised for %s", name)
                result = {"error": "tool_callback_raised", "name": name, "detail": str(e)}

            tool_calls_made.append({
                "id": call_id,
                "name": name,
                "arguments": arguments,
                "result": result,
            })
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result),
            })

        logger.info(
            "tool-loop iteration %d: %d call(s) — %s",
            iteration, len(tool_calls), ", ".join(c["name"] for c in tool_calls_made[-len(tool_calls):]),
        )
    else:
        stopped_reason = "max_iterations"
        logger.warning("chat_with_tools hit max_iterations=%d", max_iterations)

    logger.info(
        "Gemma tokens (tool loop): %s prompt / %s total across %d iteration(s)",
        total_usage.get("prompt_tokens"), total_usage.get("total_tokens"), iteration,
    )

    # If we injected a retry nudge but the final turn still had no tool call,
    # revert the phantom flag — the UI should only flash "auto-retry fixed it"
    # when a tool actually fired in the repaired turn.
    if phantom_tool_call_fixed and not tool_calls_made:
        phantom_tool_call_fixed = False

    return {
        "role": "assistant",
        "content": last_text,
        "reasoning_content": last_reasoning,
        "tool_calls_made": tool_calls_made,
        "iterations": iteration,
        "usage": total_usage or last_usage,
        "stopped_reason": stopped_reason,
        "phantom_tool_call_fixed": phantom_tool_call_fixed,
        "empty_retry_fixed": empty_retry_fixed,
    }


# ---------------------------------------------------------------------------
# Ollama lifecycle — start / stop / status for the local model
#
# Ollama installs on Windows as a background service, binds localhost:11434,
# auto-detects NVIDIA GPUs, and exposes an OpenAI-compatible API at /v1. No
# Docker Desktop knobs, no custom Dockerfiles, no host-TCP toggles.
#
# The `DockerError` class name is preserved for backward compatibility with
# the routers — it just means "local runner command failed" now.
# ---------------------------------------------------------------------------


class DockerError(RuntimeError):
    """Raised when the local runner (ollama) isn't installed or a command fails."""


def _docker_available() -> bool:
    """Reports whether the *local* runner CLI is installed (ollama or docker).

    Name is kept as `_docker_available` so the health endpoint JSON schema is
    stable for the frontend.
    """
    if LLM_RUNTIME == "ollama":
        return shutil.which("ollama") is not None
    if LLM_RUNTIME == "vllm":
        return shutil.which("docker") is not None
    # cloud runtime — no local CLI needed
    return True


def _model_runner_available() -> bool:
    """Check whether the runtime is operable.

    - ollama: `ollama --version` returns 0
    - vllm:   `docker --version` returns 0 (the actual vllm container lifecycle
              is owned by the user; our health check tests the endpoint)
    - cloud:  always true (reachability is the real check)
    """
    if LLM_RUNTIME == "cloud":
        return True
    if not _docker_available():
        return False
    cmd = ["ollama", "--version"] if LLM_RUNTIME == "ollama" else ["docker", "--version"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _container_status() -> str:
    """Return 'running' | 'pulled' | 'missing' for the configured model.

    Semantics vary by runtime:
    - ollama: 'running' = loaded in VRAM (ollama ps), 'pulled' = on disk (ollama list)
    - vllm:   we don't manage lifecycle — return 'running' iff the OpenAI
              endpoint responds, else 'missing'
    - cloud:  same as vllm — reachability implies status
    Never raises.
    """
    if LLM_RUNTIME in ("vllm", "cloud"):
        # For external runtimes, map reachability to status.
        # (Caller can still tell the difference via health.reachable.)
        return "missing"  # the sync variant; real reachability is in /health
    if not _model_runner_available():
        return "missing"

    # Match on the model name, with and without the tag suffix. Ollama list
    # shows rows like "gemma4:26b    sha256...    15 GB    2 hours ago".
    full = GEMMA_MODEL.lower()
    needles = {full}
    if ":" in full:
        needles.add(full.split(":", 1)[0])

    def _matches(line: str) -> bool:
        low = line.lower()
        return any(n in low for n in needles if len(n) >= 3)

    # First check running (loaded into VRAM) models
    try:
        r = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines()[1:]:
                if _matches(line):
                    return "running"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Then check pulled (on-disk) models
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines()[1:]:
                if _matches(line):
                    return "pulled"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "missing"


def container_status() -> dict[str, Any]:
    """Public status helper used by the /llm/status endpoint."""
    global _last_start_error
    docker_ok = _docker_available()
    status = _container_status() if docker_ok else "docker_not_installed"

    # If a previous `docker run` Popen has since exited with an error, capture
    # that so the UI can show it instead of "pulling" forever.
    if _last_start_proc is not None and _last_start_proc.poll() is not None:
        rc = _last_start_proc.returncode
        if rc != 0 and _last_start_error is None:
            try:
                _, err = _last_start_proc.communicate(timeout=1)
                _last_start_error = (err or "").strip()[:1000] or f"exit {rc}"
            except Exception:
                _last_start_error = f"docker run exited with code {rc}"

    image_built = _image_present(VLLM_IMAGE) if docker_ok else False
    build_in_progress = (
        _last_start_proc is not None
        and _last_start_proc.poll() is None
        and not image_built
    )

    # "Local" = pointing at a docker-managed vLLM on this host. If the user
    # has pointed GEMMA_BASE_URL at a cloud provider, docker lifecycle is
    # irrelevant — just show the connection + key state.
    is_local = "localhost" in GEMMA_BASE_URL or "127.0.0.1" in GEMMA_BASE_URL

    return {
        "docker_available": docker_ok,
        "container_name": VLLM_CONTAINER_NAME,
        "container_status": status,
        "image": VLLM_IMAGE,
        "image_built": image_built,
        "building_image": build_in_progress,
        "port": GEMMA_PORT,
        "hf_cache": HF_CACHE_DIR,
        "hf_token_set": bool(HF_TOKEN),
        "last_start_error": _last_start_error,
        "provider_mode": "local" if is_local else "cloud",
        "api_key_set": bool(GEMMA_API_KEY),
        "runtime": LLM_RUNTIME,
    }


def _start_vllm_container(hf_token: str | None = None) -> dict[str, Any]:
    """Spawn the geoenv-vllm:latest docker container with the configured model.

    Runs `docker run -d` (detached) so the backend returns quickly. The UI
    polls /health to see when vLLM finishes loading (~30-60s on warm restart,
    much longer on first model download).
    """
    if not shutil.which("docker"):
        raise DockerError(
            "Docker is not installed or not on PATH. Install Docker Desktop "
            "with WSL2 + NVIDIA GPU support, then click Start again."
        )

    # Already running? No-op.
    try:
        r = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"publish={GEMMA_PORT}"],
            capture_output=True, text=True, timeout=10,
        )
        if (r.stdout or "").strip():
            return {
                "started": False,
                "already_running": True,
                "model": GEMMA_MODEL,
                "note": f"A container is already publishing port {GEMMA_PORT}.",
            }
    except subprocess.TimeoutExpired:
        pass

    # Ensure HF cache host path exists so Docker Desktop bind-mounts cleanly.
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    hf_cache = HF_CACHE_DIR.replace("\\", "/")

    # Also persist vLLM's torch.compile cache. Without this mount, every
    # restart re-compiles the graph (~38 s wasted on warm restart). With it,
    # warm restart drops from ~70 s to ~30 s.
    vllm_cache_host = os.path.expanduser("~/.cache/vllm")
    os.makedirs(vllm_cache_host, exist_ok=True)
    vllm_cache = vllm_cache_host.replace("\\", "/")

    cmd = [
        "docker", "run", "-d", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        "-v", f"{vllm_cache}:/root/.cache/vllm",
        "-p", f"{GEMMA_PORT}:8000",
    ]
    # HF token for gated models. Never logged, never written to disk — lives
    # only in the docker process's env vars.
    if hf_token and hf_token.strip():
        cmd += ["-e", f"HF_TOKEN={hf_token.strip()}"]

    cmd += [
        VLLM_IMAGE,
        "--model", GEMMA_MODEL,
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.75",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
        "--reasoning-parser", "gemma4",
    ]

    global _last_start_proc, _last_start_error
    _last_start_error = None

    # Redact token before logging
    log_cmd = [a if not a.startswith("HF_TOKEN=") else "HF_TOKEN=<redacted>" for a in cmd]
    logger.info(f"Launching vLLM container: {' '.join(log_cmd)}")

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        raise DockerError(
            f"docker run timed out after 60 s. Image pull may be slow; retry. "
            f"({e})"
        ) from e

    if r.returncode != 0:
        err = (r.stderr or "").strip() or (r.stdout or "").strip() or f"exit {r.returncode}"
        # Redact token from error output if present
        if hf_token:
            err = err.replace(hf_token, "<redacted>")
        _last_start_error = err[:1500]
        raise DockerError(f"docker run failed: {err[:500]}")

    container_id = (r.stdout or "").strip().split("\n")[0]
    return {
        "started": True,
        "already_running": False,
        "container_id": container_id[:12],
        "model": GEMMA_MODEL,
        "note": (
            "Container started. First launch pulls weights (~several min). "
            "The UI will auto-detect readiness — badge flips green when "
            "vLLM's /v1/models endpoint responds."
        ),
    }


def start_container(hf_token: str | None = None) -> dict[str, Any]:
    """Start the configured runtime's local model.

    - ollama: runs `ollama pull` for the configured model
    - vllm:   spawns geoenv-vllm:latest via `docker run -d` with the configured
              model + flags. Optional hf_token injected as -e HF_TOKEN= for
              gated models (token lives only in the docker process env, never
              written to disk or logged by us).
    - cloud:  no-op; returns a descriptive note
    """
    if LLM_RUNTIME == "cloud":
        return {
            "started": False,
            "already_running": True,
            "model": GEMMA_MODEL,
            "note": (
                "Cloud provider mode — nothing to start locally. Check that "
                "GEMMA_API_KEY is set and GEMMA_BASE_URL is reachable."
            ),
        }
    if LLM_RUNTIME == "vllm":
        return _start_vllm_container(hf_token=hf_token)

    # ollama branch
    if not _docker_available():
        raise DockerError(
            "Ollama is not installed or not on PATH. Install from "
            "https://ollama.com/download (Windows installer runs as a service)."
        )
    if not _model_runner_available():
        raise DockerError(
            "Ollama CLI is installed but not responding. Is the Ollama service "
            "running? Try: `ollama serve` in a terminal."
        )

    status = _container_status()
    if status == "running":
        return {
            "started": False,
            "already_running": True,
            "model": GEMMA_MODEL,
        }
    if status == "pulled":
        # Already on disk — Ollama will auto-load on first /v1 request.
        return {
            "started": True,
            "already_running": False,
            "model": GEMMA_MODEL,
            "note": (
                "Model is already pulled. Ollama auto-loads on the first "
                "/v1/chat/completions request — Send a message to warm it up."
            ),
        }

    global _last_start_proc, _last_start_error
    _last_start_error = None

    cmd = ["ollama", "pull", GEMMA_MODEL]
    logger.info(f"Pulling Ollama model: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _last_start_proc = proc
    except FileNotFoundError as e:
        raise DockerError(f"ollama binary not found: {e}") from e

    # Peek for fast failures (bad tag, daemon down). Ollama errors come back
    # quickly when they happen.
    try:
        stdout, stderr = proc.communicate(timeout=5)
        if proc.returncode != 0:
            _last_start_error = (
                (stderr or "").strip()[:1200] or (stdout or "").strip()[:1200]
                or f"exit {proc.returncode}"
            )
            raise DockerError(
                f"ollama pull failed (exit {proc.returncode}):\n{_last_start_error}"
            )
        return {
            "started": True,
            "already_running": False,
            "model": GEMMA_MODEL,
            "note": "Pull complete. Ollama will load the model on first request.",
        }
    except subprocess.TimeoutExpired:
        return {
            "started": True,
            "already_running": False,
            "model": GEMMA_MODEL,
            "pulling_image": True,
            "note": (
                f"ollama pull is fetching {GEMMA_MODEL} from ollama.com. "
                "For gemma4:26b expect ~15-18 GB; for gemma4:e4b ~5 GB. "
                "Subsequent launches are near-instant. Poll /health for readiness.\n\n"
                "If pull stalls, fallback options:\n"
                "  • Smaller tag: GEMMA_MODEL=gemma4:e4b\n"
                "  • Cloud: set GEMMA_BASE_URL + GEMMA_API_KEY"
            ),
        }


def stop_container() -> dict[str, Any]:
    """Stop the local model.

    - ollama: unloads the model from VRAM (`ollama stop <model>`)
    - vllm:   finds the container publishing our configured port and stops it
    - cloud:  no-op (nothing to stop)
    """
    if LLM_RUNTIME == "cloud":
        return {"stopped": False, "was_status": "cloud_runtime_nothing_to_stop"}

    if LLM_RUNTIME == "vllm":
        # Find containers publishing our GEMMA_PORT and stop them. This works
        # whether the user ran `docker run --rm` or detached.
        if not shutil.which("docker"):
            raise DockerError("Docker is not available to stop the vLLM container.")
        try:
            r = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"publish={GEMMA_PORT}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ids = [line.strip() for line in (r.stdout or "").splitlines() if line.strip()]
            if not ids:
                return {
                    "stopped": False,
                    "was_status": "no_vllm_container_on_port",
                    "port": GEMMA_PORT,
                    "note": f"No Docker container publishing port {GEMMA_PORT} was found.",
                }
            stop = subprocess.run(
                ["docker", "stop", *ids],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if stop.returncode != 0:
                raise DockerError(f"docker stop failed: {stop.stderr.strip() or stop.stdout.strip()}")
            return {
                "stopped": True,
                "container_ids": ids,
                "port": GEMMA_PORT,
                "note": (
                    "Stopped the vLLM container. Click Start (or re-run your "
                    "docker command in WSL) to bring it back — cached weights "
                    "make subsequent starts near-instant."
                ),
            }
        except subprocess.TimeoutExpired as e:
            raise DockerError(f"docker command timed out: {e}") from e

    if not _docker_available():
        raise DockerError("Ollama is not installed")
    if not _model_runner_available():
        raise DockerError("Ollama service not responding")

    status = _container_status()
    if status != "running":
        return {"stopped": False, "was_status": status}

    r = subprocess.run(
        ["ollama", "stop", GEMMA_MODEL],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise DockerError(f"ollama stop failed: {r.stderr}")

    return {"stopped": True, "model": GEMMA_MODEL}


def _image_present(image: str | None = None) -> bool:
    """Return True if the given image is already built/pulled to local daemon."""
    image = image or VLLM_IMAGE
    if not _docker_available():
        return False
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _build_custom_image() -> None:
    """Build geoenv-vllm:latest from our Dockerfile if it doesn't exist yet.

    Uses Popen so the FastAPI request returns fast; the build runs in the
    background and subsequent container launches wait for it via /health polling.
    """
    global _last_start_proc, _last_start_error
    dockerfile_dir = os.path.abspath(VLLM_DOCKERFILE_DIR)
    if not os.path.isdir(dockerfile_dir):
        raise DockerError(f"Dockerfile dir not found: {dockerfile_dir}")

    logger.info(f"Building {VLLM_IMAGE} from {dockerfile_dir}/Dockerfile.vllm")
    proc = subprocess.Popen(
        [
            "docker", "build",
            "-t", VLLM_IMAGE,
            "-f", os.path.join(dockerfile_dir, "Dockerfile.vllm"),
            dockerfile_dir,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _last_start_proc = proc
    _last_start_error = None


def _pull_progress() -> str | None:
    """Best-effort snapshot of an in-flight `docker pull`: size on disk so far.

    Returns a one-line human string or None if not in progress.
    """
    if not _docker_available():
        return None
    try:
        # `docker system df` gives a rough view of disk usage per category
        r = subprocess.run(
            ["docker", "system", "df", "--format", "{{.Type}}\t{{.Size}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.lower().startswith("images"):
                    return f"Local image cache: {line.split(chr(9))[1]}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def container_logs(tail: int = 100) -> str:
    """Return runtime-appropriate status / diagnostic output."""
    if LLM_RUNTIME == "cloud":
        return (
            "[cloud runtime]\n"
            f"Base URL: {GEMMA_BASE_URL}\n"
            f"Model:    {GEMMA_MODEL}\n"
            f"API key:  {'set' if GEMMA_API_KEY else 'NOT SET — cloud providers will reject requests'}"
        )
    if LLM_RUNTIME == "vllm":
        # Inspect the actual container state rather than printing stale advice.
        try:
            ps = subprocess.run(
                ["docker", "ps", "--filter", f"publish={GEMMA_PORT}",
                 "--format", "{{.ID}} {{.Image}} {{.Status}}"],
                capture_output=True, text=True, timeout=10,
            )
            running_lines = [l for l in (ps.stdout or "").splitlines() if l.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            running_lines = []

        header = (
            f"[vllm runtime]\n"
            f"Endpoint : {GEMMA_BASE_URL}\n"
            f"Model    : {GEMMA_MODEL}\n"
            f"Image    : {VLLM_IMAGE}\n"
            f"Port     : {GEMMA_PORT}\n"
        )
        if running_lines:
            log_blocks: list[str] = []
            lines_per_container = min(max(tail, 10), 200)
            for line in running_lines:
                # Grab the most recent log lines from the container for quick diag
                cid = line.split()[0]
                try:
                    lg = subprocess.run(
                        ["docker", "logs", "--tail", str(lines_per_container), cid],
                        capture_output=True, text=True, timeout=10,
                    )
                    log_blocks.append(
                        f"--- last log lines from {cid} ---\n"
                        + ((lg.stdout or "") + (lg.stderr or "")).strip()
                    )
                except subprocess.TimeoutExpired:
                    log_blocks.append(f"(log fetch timed out for {cid})")
            return (
                header
                + "\nContainer(s) up:\n  " + "\n  ".join(running_lines)
                + "\n\n" + ("\n\n".join(log_blocks) if log_blocks else "")
            )
        if _last_start_error:
            return header + "\nNo container running on this port.\n\nLast start error:\n" + _last_start_error
        return (
            header
            + "\nNo container running on this port yet.\n"
            + "Click ▶ Start LLM container to launch one (backend spawns docker run)."
        )
    # ollama branch
    if not _docker_available():
        return (
            "[Ollama not installed]\n"
            "Download from https://ollama.com/download and run the Windows "
            "installer. It registers as a background service on :11434."
        )
    if not _model_runner_available():
        return (
            "[Ollama CLI installed, but service not responding]\n"
            "Start it with `ollama serve` in a terminal, or restart the "
            "Ollama Windows service."
        )

    status = _container_status()

    if status == "missing":
        if _last_start_proc is not None and _last_start_proc.poll() is None:
            return (
                f"[pulling model] {GEMMA_MODEL} — ollama pull is downloading "
                "weights. gemma4:26b is ~15-18 GB; gemma4:e4b is ~5 GB.\n"
                "Watch progress in another terminal:\n"
                "    ollama list     (pulled models)\n"
                "    ollama ps       (running models)\n"
            )
        if _last_start_error:
            return (
                f"[last start failed]\n{_last_start_error}\n\n"
                "Common fixes:\n"
                "  • Retry — transient network resets often clear on retry\n"
                "  • Try a smaller tag: set GEMMA_MODEL=gemma4:e4b\n"
                "  • Or set GEMMA_BASE_URL + GEMMA_API_KEY for a cloud provider"
            )
        return (
            f"[model not pulled] {GEMMA_MODEL} is not on disk. "
            "Click Start in the LLM panel to pull it from ollama.com."
        )

    # Model is pulled or running — show `ollama list` + `ollama ps` side by side
    try:
        lst = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ps = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = "=== pulled models ===\n" + (lst.stdout or lst.stderr or "(empty)\n")
        out += "\n=== running models ===\n" + (ps.stdout or ps.stderr or "(none)\n")
        return out
    except subprocess.TimeoutExpired:
        return "[log fetch timed out]"
