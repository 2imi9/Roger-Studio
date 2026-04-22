"""Cloud chat router — NVIDIA NIM as an alternative to Local Chat (vLLM/Ollama).

Mounts under ``/api/cloud/*``. Endpoints intentionally mirror the gemma/*
shape so the frontend can keep one common ChatMessage type and one chat-loop
contract; only the underlying transport differs.
"""
from fastapi import APIRouter, Body, HTTPException, Query

from app.services import cloud_llm_client, geo_tools

router = APIRouter()


@router.get("/cloud/health")
async def cloud_health(
    api_key: str | None = Query(
        None,
        description=(
            "Optional NIM API key — sent only when the user has pasted a key "
            "in the UI Settings panel. Falls back to the NVIDIA_API_KEY env var."
        ),
    ),
) -> dict:
    """Report whether NVIDIA NIM is reachable + auth-OK with the given key.

    Never raises — the response always includes ``reachable`` / ``auth_ok``
    / ``error`` so the UI can render a status badge.

    NB: passing the key as a query param is fine here because the response
    is uncached and Roger's backend doesn't log query strings. The chat
    endpoint takes the key in the body to keep it out of access logs.
    """
    return await cloud_llm_client.health_check(api_key_override=api_key)


@router.post("/cloud/model")
async def cloud_set_model(payload: dict = Body(...)) -> dict:
    """Update the backend's active NIM model at runtime.

    Body: ``{"model": "openai/gpt-oss-120b"}``
    """
    new_model = (payload.get("model") or "").strip()
    if not new_model:
        raise HTTPException(400, "model field is required")
    cloud_llm_client.set_model(new_model)
    return {"ok": True, "model": cloud_llm_client.NVIDIA_MODEL}


@router.post("/cloud/chat")
async def cloud_chat(
    payload: dict = Body(
        ...,
        description=(
            "{messages: [{role, content}], "
            "scene_context?: {area, datasets, auto_label_summary, polygon_features}, "
            "api_key?: '<optional per-request key, never persisted>', "
            "model?: '<optional per-request model override>', "
            "tools?: 'auto' | 'none'}"
        ),
    ),
) -> dict:
    """Tool-augmented chat against NVIDIA NIM, anchored on the current scene.

    Same request/response shape as ``/auto-label/gemma/chat`` (Local Chat) so
    the UI just swaps the API call. ``api_key`` and ``model`` are optional
    per-request overrides — the backend uses ``NVIDIA_API_KEY`` /
    ``NVIDIA_MODEL`` env vars when not provided.
    """
    messages = payload.get("messages", [])
    scene = payload.get("scene_context") or {}
    api_key = payload.get("api_key") or None
    model = payload.get("model") or None
    tool_choice = payload.get("tools", "auto")
    if tool_choice not in ("auto", "none"):
        tool_choice = "auto"

    if not messages:
        raise HTTPException(400, "messages array is required")

    scene_block = ""
    if scene:
        area = scene.get("area")
        datasets = scene.get("datasets") or []
        summary = scene.get("auto_label_summary")
        polygon_count = len(scene.get("polygon_features") or [])
        scene_block = "\n\nCurrent scene context (grounds your answers):\n"
        area_bbox = scene.get("area_bbox")
        if area:
            scene_block += f"  selected area: {area}\n"
        if isinstance(area_bbox, dict) and all(k in area_bbox for k in ("west", "south", "east", "north")):
            # Dump as a JSON object the LLM can copy verbatim into any tool
            # call that expects a bbox argument. Avoids "what are the
            # coordinates?" hallucinations when the area was already drawn.
            import json as _json  # noqa: PLC0415
            scene_block += (
                f"  bbox (use this as the 'bbox' arg for tool calls): "
                f"{_json.dumps(area_bbox)}\n"
            )
        if datasets:
            scene_block += f"  loaded datasets: {', '.join(datasets)}\n"
        if summary:
            scene_block += f"  last auto-label run: {summary}\n"
        if polygon_count:
            scene_block += f"  polygons available for lookup: {polygon_count}\n"

    system_prompt = (
        "You are a cloud-hosted geo-assistant embedded in Roger Studio, a "
        "remote-sensing workbench. You are running on NVIDIA NIM. The user "
        "is a geoscientist. Give concise, technically precise answers. When "
        "they ask about a polygon or class, refer to the scene context "
        "provided or call the available tools. Prefer bulleted answers for "
        "multi-part questions. Refuse to invent satellite data you haven't "
        "been shown — if unsure, say so or call a tool to fetch real data.\n\n"
        "TOOL USE — NON-NEGOTIABLE:\n"
        "1. When the user asks you to run a model, map tiles, compute NDVI, "
        "look up the OlmoEarth catalog, or fetch polygon stats, you MUST "
        "emit a tool_call. Do NOT describe the tool in prose, and DO NOT "
        "write a fake 'Result' or 'Job ID' block — if you didn't emit a "
        "tool_call in THIS turn, the tool DID NOT RUN and there is no job.\n"
        "2. NEVER say 'I have initiated', 'please stand by', 'I will notify "
        "you', 'process complete', or any variant. There is no async "
        "notification channel — the result arrives in the SAME turn via a "
        "role=tool message after your tool_call.\n"
        "3. NEVER invent job_ids, scene_ids, tile URLs, or pixel-count "
        "tables. These only exist after a tool_call runs and returns real "
        "data in a role=tool message. If the user pastes what looks like a "
        "prior result, treat it as user-provided text, not your output.\n"
        "4. If the scene_context includes a 'bbox' line with a JSON object, "
        "USE THAT OBJECT verbatim as the 'bbox' argument — do NOT ask the "
        "user for coordinates they've already provided.\n"
        "5. If a tool result carries an 'artifacts' array, summarize in "
        "2-3 sentences and cite the download; do NOT paste the raw table."
        + scene_block
    )

    chat_messages = [{"role": "system", "content": system_prompt}, *messages]

    async def _execute(name: str, arguments: str) -> dict:
        # Bounded by ``TOOL_EXECUTION_TIMEOUT_S`` (120 s) so a hung upstream
        # (STAC, Open-Meteo, HuggingFace, OlmoEarth forward pass) can't
        # lock up the chat turn. On timeout the LLM sees a ``tool_timeout``
        # payload and can retry / summarize instead of hanging.
        return await geo_tools.execute_tool_with_timeout(name, arguments, scene)

    try:
        resp = await cloud_llm_client.chat_with_tools(
            messages=chat_messages,
            tools=geo_tools.TOOL_SCHEMAS,
            execute_tool_fn=_execute,
            api_key_override=api_key,
            model_override=model,
            temperature=0.4,
            max_tokens=800,
            tool_choice=tool_choice,
        )
    except cloud_llm_client.NvidiaNimUnavailable as e:
        raise HTTPException(503, str(e))

    return {
        "role": "assistant",
        "content": resp.get("content", ""),
        "tool_calls_made": resp.get("tool_calls_made", []),
        "iterations": resp.get("iterations", 1),
        "stopped_reason": resp.get("stopped_reason"),
        "usage": resp.get("usage", {}),
        "model": resp.get("model"),
        "provider": resp.get("provider", "nvidia_nim"),
    }
