"""Google Gemini cloud chat router — third provider in the Cloud umbrella.

Mounts under ``/api/gemini/*``. Mirrors the NIM (``/api/cloud``) +
Anthropic Claude (``/api/claude``) routes so the frontend's unified
Cloud panel can switch providers without changing the message flow.
"""
from fastapi import APIRouter, Body, HTTPException, Query

from app.services import gemini_client, geo_tools

router = APIRouter()


@router.get("/gemini/health")
async def gemini_health(
    api_key: str | None = Query(
        None,
        description=(
            "Optional Google AI Studio API key — sent only when the user "
            "has pasted one in the UI Settings panel. Falls back to the "
            "GEMINI_API_KEY / GOOGLE_API_KEY env var."
        ),
    ),
) -> dict:
    """Report whether Gemini is reachable + auth-OK with the given key.

    Never raises — the response always includes ``reachable`` / ``auth_ok``
    / ``error`` so the UI can render a status badge.
    """
    return await gemini_client.health_check(api_key_override=api_key)


@router.post("/gemini/model")
async def gemini_set_model(payload: dict = Body(...)) -> dict:
    """Update the backend's active Gemini model at runtime.

    Body: ``{"model": "gemini-2.5-pro"}``
    """
    new_model = (payload.get("model") or "").strip()
    if not new_model:
        raise HTTPException(400, "model field is required")
    gemini_client.set_model(new_model)
    return {"ok": True, "model": gemini_client.GEMINI_MODEL}


@router.post("/gemini/chat")
async def gemini_chat(
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
    """Tool-augmented chat against Google Gemini, anchored on the current scene.

    Same request/response shape as ``/api/cloud/chat`` (NIM) and
    ``/api/claude/chat`` so the UI can treat all three as a single Cloud
    pipeline with only the provider picker changing.
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
        "remote-sensing workbench. You are running on Google Gemini. The "
        "user is a geoscientist. Give concise, technically precise answers. "
        "When they ask about a polygon or class, refer to the scene context "
        "provided or call the available tools. Prefer bulleted answers for "
        "multi-part questions. Refuse to invent satellite data you haven't "
        "been shown — if unsure, say so or call a tool to fetch real data.\n\n"
        "TOOL USE — NON-NEGOTIABLE:\n"
        "1. When the user asks you to run a model, map tiles, compute NDVI, "
        "look up the OlmoEarth catalog, or fetch polygon stats, you MUST "
        "emit a tool_call. Do NOT describe the tool in prose.\n"
        "2. NEVER say 'I have initiated', 'please stand by', 'I will notify "
        "you', or any variant. There is no async channel; if you did not "
        "emit a tool_call in the SAME turn, the tool DID NOT RUN.\n"
        "3. If a tool result carries an 'artifacts' array, DO NOT paste the "
        "raw data. Summarize in 2-3 sentences and tell the user the file is "
        "available for download — the UI renders the download pill "
        "automatically.\n"
        + scene_block
    )

    chat_messages = [{"role": "system", "content": system_prompt}, *messages]

    async def _execute(name: str, arguments: str) -> dict:
        # Bounded tool-call ceiling — see ``cloud_chat.py`` for rationale.
        return await geo_tools.execute_tool_with_timeout(name, arguments, scene)

    try:
        resp = await gemini_client.chat_with_tools(
            messages=chat_messages,
            tools=geo_tools.TOOL_SCHEMAS,
            execute_tool_fn=_execute,
            api_key_override=api_key,
            model_override=model,
            temperature=0.4,
            max_tokens=1500,
            tool_choice=tool_choice,
        )
    except gemini_client.GeminiUnavailable as e:
        raise HTTPException(503, str(e))

    # Surface artifacts flat for the UI (same pattern as auto_label.py).
    artifacts_this_turn: list[dict] = []
    for call in resp.get("tool_calls_made", []):
        result = call.get("result") or {}
        for a in (result.get("artifacts") or []):
            artifacts_this_turn.append(a)

    return {
        "role": "assistant",
        "content": resp.get("content", ""),
        "tool_calls_made": resp.get("tool_calls_made", []),
        "iterations": resp.get("iterations", 1),
        "stopped_reason": resp.get("stopped_reason"),
        "usage": resp.get("usage", {}),
        "model": resp.get("model"),
        "provider": "google_gemini",
        "artifacts": artifacts_this_turn,
    }
