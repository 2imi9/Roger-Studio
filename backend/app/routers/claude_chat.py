"""Claude chat router — Anthropic's Messages API as a third LLM transport.

Mounts under ``/api/claude/*``. Endpoints intentionally mirror ``cloud_chat``
(NIM) and ``auto_label`` (Gemma local) so the frontend can swap transports
behind a single ChatMessage type and a single tool-loop contract.
"""
from fastapi import APIRouter, Body, HTTPException, Query

from app.services import claude_client, geo_tools

router = APIRouter()


@router.get("/claude/health")
async def claude_health(
    api_key: str | None = Query(
        None,
        description=(
            "Optional Anthropic API key — sent only when the user has pasted "
            "a key in Claude Chat Settings. Falls back to ANTHROPIC_API_KEY."
        ),
    ),
) -> dict:
    """Report whether Anthropic is reachable + auth-OK with the given key.

    Never raises; always returns ``reachable`` / ``auth_ok`` / ``error``.
    """
    return await claude_client.health_check(api_key_override=api_key)


@router.post("/claude/model")
async def claude_set_model(payload: dict = Body(...)) -> dict:
    """Update the backend's active Claude model at runtime.

    Body: ``{"model": "claude-sonnet-4-6"}``
    """
    new_model = (payload.get("model") or "").strip()
    if not new_model:
        raise HTTPException(400, "model field is required")
    claude_client.set_model(new_model)
    return {"ok": True, "model": claude_client.CLAUDE_MODEL}


@router.post("/claude/chat")
async def claude_chat(
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
    """Tool-augmented chat against Anthropic Claude, anchored on the scene.

    Same request/response shape as ``/api/cloud/chat`` (NIM) and
    ``/api/auto-label/gemma/chat`` (Local) so the UI just swaps the API call.
    ``api_key`` and ``model`` are optional per-request overrides — backend
    falls back to env vars when absent.
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
        "You are Claude, embedded in Roger Studio — a remote-sensing "
        "workbench for a geoscientist. Give concise, technically precise "
        "answers. When the user asks about a polygon or class, refer to the "
        "scene context provided or call the available tools. Prefer "
        "bulleted answers for multi-part questions. Refuse to invent "
        "satellite data you haven't been shown — if unsure, say so or call "
        "a tool to fetch real data.\n\n"
        "TOOL USE — NON-NEGOTIABLE:\n"
        "1. When the user asks you to run a model, map tiles, compute NDVI, "
        "look up the OlmoEarth catalog, or fetch polygon stats, you MUST "
        "emit a tool_use block. Do NOT describe the tool in prose, and DO "
        "NOT write a fake 'Result' or 'Job ID' section.\n"
        "2. NEVER say 'I have initiated', 'please stand by', 'I will notify "
        "you', or any variant. Real results arrive in the SAME turn via a "
        "tool_result block after your tool_use.\n"
        "3. NEVER invent job_ids, scene_ids, tile URLs, or pixel-count "
        "tables — these exist only after a tool_use actually runs.\n"
        "4. If the scene_context includes a 'bbox' line with a JSON object, "
        "USE THAT OBJECT verbatim as the 'bbox' tool argument — do not ask "
        "the user for coordinates they've already provided.\n"
        "5. If a tool result carries an 'artifacts' array, summarize in "
        "2-3 sentences and cite the download; do NOT paste the raw table."
        + scene_block
    )

    async def _execute(name: str, arguments) -> dict:
        # Bounded tool-call ceiling — see ``cloud_chat.py`` for rationale.
        return await geo_tools.execute_tool_with_timeout(name, arguments, scene)

    try:
        resp = await claude_client.chat_with_tools(
            messages=messages,
            tools=geo_tools.TOOL_SCHEMAS,
            execute_tool_fn=_execute,
            api_key_override=api_key,
            model_override=model,
            system=system_prompt,
            max_tokens=4096,
            tool_choice=tool_choice,
        )
    except claude_client.ClaudeUnavailable as e:
        raise HTTPException(503, str(e))

    return {
        "role": "assistant",
        "content": resp.get("content", ""),
        "tool_calls_made": resp.get("tool_calls_made", []),
        "iterations": resp.get("iterations", 1),
        "stopped_reason": resp.get("stopped_reason"),
        "usage": resp.get("usage", {}),
        "model": resp.get("model"),
        "provider": resp.get("provider", "anthropic"),
    }
