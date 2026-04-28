from fastapi import APIRouter, Body, HTTPException, Query

from app.models.schemas import AutoLabelResponse
from app.services.data_ingest import UPLOAD_DIR
from app.services import database as db
from app.services import geo_agent, gemma_client, geo_tools

router = APIRouter()

TIPSV2_AVAILABLE = False
try:
    from app.services.tipsv2_labeler import auto_label_geotiff_tipsv2, DEFAULT_CLASSES
    TIPSV2_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    import logging
    logging.getLogger(__name__).warning(f"TIPSv2 not available: {e}")
    DEFAULT_CLASSES = []

SAMGEO_AVAILABLE = False
try:
    from app.services.samgeo_labeler import auto_label_geotiff_samgeo
    SAMGEO_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    import logging
    logging.getLogger(__name__).warning(f"SamGeo not available: {e}")

from app.services.auto_label import auto_label_raster, auto_label_vector


@router.post("/auto-label/{filename}", response_model=AutoLabelResponse)
async def auto_label_dataset(
    filename: str,
    n_classes: int = Query(6, ge=2, le=12, description="Number of land cover classes"),
    min_segment_pixels: int = Query(200, ge=10, le=5000, description="Minimum segment size in pixels"),
    method: str = Query("auto", description="Method: 'auto', 'tipsv2', 'spectral', or 'samgeo'"),
    model: str = Query("google/tipsv2-b14", description="TIPSv2 model name"),
    sliding_window: bool = Query(True, description="Use overlapping-tile inference for pixel-accurate boundaries (TIPSv2 only)"),
    # User-supplied label classes for TIPSv2 zero-shot. Each item is
    # ``{"name": str, "prompt": str, "color": "#RRGGBB"}``. When omitted,
    # ``tipsv2_labeler.DEFAULT_CLASSES`` is used. The audit caught this:
    # the LabelClassEditor in the UI was a no-op because the prompts
    # never reached the backend — TIPSv2 always classified against the
    # built-in 8-class land-cover prompt set regardless of what the user
    # typed. Sending them through the body lets the user actually steer
    # the classifier.
    classes: list[dict] | None = Body(None, embed=True),
) -> dict:
    """Auto-label an uploaded dataset.

    Methods:
      - 'auto': Uses TIPSv2 zero-shot for rasters if available, spectral fallback otherwise
      - 'tipsv2': Force TIPSv2 zero-shot (requires transformers + GPU)
      - 'spectral': K-means on spectral bands (no GPU needed)

    TIPSv2 models (Apache 2.0, from Google DeepMind CVPR'26):
      - google/tipsv2-b14  (fast, ~300MB VRAM)
      - google/tipsv2-l14  (balanced, ~600MB)
      - google/tipsv2-g14  (best, ~2GB)
    """
    # Strict method validation. The audit caught this: unknown methods (e.g.
    # a typo like `method=spectal` or a future value the frontend added
    # without updating the backend) used to fall through the elif chain and
    # silently run spectral k-means. The user saw a result and assumed it
    # was the method they asked for. Reject unknown values explicitly so
    # the client sees the mismatch.
    allowed_methods = {"auto", "tipsv2", "spectral", "samgeo"}
    if method not in allowed_methods:
        raise HTTPException(
            400,
            f"Invalid method '{method}'. Must be one of: {sorted(allowed_methods)}",
        )

    info = db.get_dataset(filename)
    if info is None:
        raise HTTPException(404, f"Dataset '{filename}' not found")

    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"File not found on disk: {filename}")

    fmt = info.format

    try:
        if fmt in ("geotiff", "cog"):
            if method == "samgeo":
                if not SAMGEO_AVAILABLE:
                    raise HTTPException(422, "SamGeo not available. Install: pip install segment-geospatial")
                result = auto_label_geotiff_samgeo(str(filepath), n_clusters=n_classes)
            elif method == "tipsv2" or (method == "auto" and TIPSV2_AVAILABLE):
                if not TIPSV2_AVAILABLE:
                    raise HTTPException(422, "TIPSv2 not available. Install: pip install transformers torch")
                result = auto_label_geotiff_tipsv2(
                    str(filepath),
                    classes=classes,
                    model_name=model,
                    sliding_window=sliding_window,
                )
            else:
                result = auto_label_raster(str(filepath), n_classes=n_classes, min_segment_pixels=min_segment_pixels)

        elif fmt in ("geojson", "geopackage", "shapefile", "parquet", "geoparquet", "csv"):
            result = auto_label_vector(str(filepath), n_classes=n_classes)

        else:
            raise HTTPException(422, f"Auto-labeling not supported for format '{fmt}'")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Auto-labeling failed: {e}")

    return result


@router.post("/auto-label/{filename}/validate", response_model=AutoLabelResponse)
async def validate_labels(
    filename: str,
    geojson: dict = Body(..., description="FeatureCollection produced by a prior auto-label run"),
    pipeline: str = Query("tipsv2", description="Which upstream pipeline generated the features"),
    only_low_confidence: bool = Query(True, description="Validate only polygons with confidence < threshold"),
    low_conf_threshold: float = Query(0.6, ge=0.0, le=1.0),
    max_concurrent: int = Query(4, ge=1, le=16),
) -> dict:
    """Re-validate an auto-label result with Gemma 4.

    The agent pulls evidence from every available geo source (raster patch,
    elevation, weather, neighbor polygons) and returns a revised FeatureCollection
    where each polygon's `properties.validation` block contains the agent's verdict.
    """
    info = db.get_dataset(filename)
    if info is None:
        raise HTTPException(404, f"Dataset '{filename}' not found")

    raster_path = UPLOAD_DIR / filename
    if not raster_path.exists():
        raster_path = None  # proceed without patch cropping

    # Pre-flight: verify the LLM is reachable before burning time
    if not await gemma_client.health_check():
        raise HTTPException(
            503,
            f"LLM unreachable at {gemma_client.GEMMA_BASE_URL} "
            f"(runtime={gemma_client.LLM_RUNTIME}). "
            "Click Start in the LLM panel (Ollama), run `docker run vllm/vllm-openai:nightly ...` "
            "in WSL (vLLM), or set GEMMA_BASE_URL + GEMMA_API_KEY for cloud.",
        )

    try:
        validated = await geo_agent.validate_geojson(
            geojson,
            raster_path=raster_path,
            pipeline=pipeline,
            only_low_confidence=only_low_confidence,
            low_conf_threshold=low_conf_threshold,
            max_concurrent=max_concurrent,
        )
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {e}")

    return validated


@router.get("/auto-label/gemma/health")
async def gemma_health() -> dict:
    """Report whether the configured LLM endpoint is reachable.

    The endpoint can be Ollama (:11434), vLLM (:8000), or any cloud provider —
    whichever is selected via LLM_RUNTIME / GEMMA_BASE_URL.

    Response stays ``-> dict`` (no pydantic model) because the ``**status``
    spread varies by runtime: Ollama publishes ``{pulling, pull_progress}``,
    vLLM publishes ``{container_id, up_seconds}``, cloud providers publish
    nothing extra. The audit accepts untyped here on the grounds that the
    frontend's ``LLMHealth`` interface already covers the common subset and
    the optional fields are only displayed in the debug panel.
    """
    reachable = await gemma_client.health_check()
    status = gemma_client.container_status()
    return {
        "reachable": reachable,
        "base_url": gemma_client.GEMMA_BASE_URL,
        "model": gemma_client.GEMMA_MODEL,
        **status,
    }


@router.post("/auto-label/gemma/model")
async def set_gemma_model(payload: dict = Body(...)) -> dict:
    """Update the backend's active model at runtime (UI picker drives this).
    Body: {"model": "unsloth/gemma-4-26b-a4b-it-bnb-4bit"}
    """
    new_model = (payload.get("model") or "").strip()
    if not new_model:
        raise HTTPException(400, "model field is required")
    gemma_client.set_model(new_model)
    return {"ok": True, "model": gemma_client.GEMMA_MODEL}


@router.post("/auto-label/gemma/start")
async def gemma_start(payload: dict = Body(default={})) -> dict:
    """Launch the configured runtime (vllm: docker run, ollama: pull).

    Optional body: {"hf_token": "hf_..."} — passed through to the container's
    HF_TOKEN env var for gated models. Never persisted or logged.
    """
    hf_token = (payload.get("hf_token") or "").strip() or None
    try:
        return gemma_client.start_container(hf_token=hf_token)
    except gemma_client.DockerError as e:
        raise HTTPException(422, str(e))


@router.post("/auto-label/gemma/stop")
async def gemma_stop() -> dict:
    """Stop the vLLM docker container (idempotent)."""
    try:
        return gemma_client.stop_container()
    except gemma_client.DockerError as e:
        raise HTTPException(422, str(e))


@router.get("/auto-label/gemma/logs")
async def gemma_logs(tail: int = Query(200, ge=10, le=2000)) -> dict:
    """Return the last N lines of the container log (useful while weights download)."""
    return {"logs": gemma_client.container_logs(tail=tail)}


@router.post("/auto-label/gemma/chat")
async def gemma_chat(
    payload: dict = Body(
        ...,
        description=(
            "{messages: [{role, content}], scene_context?: "
            "{area, datasets, auto_label_summary, polygon_features}, "
            "tools?: 'auto' | 'none'}"
        ),
    ),
) -> dict:
    """Tool-augmented chat with Gemma 4, anchored on the current scene context.

    Response stays ``-> dict`` (no response_model) because the tool-call
    branch is polymorphic: a turn might return ``{role, content}`` alone,
    or ``{role, content, tool_calls: [...], artifacts: [...]}``, where each
    tool's artifact has a different shape (geojson / png / scalar / table).
    Pinning a single pydantic model would either force ``Any`` on the
    artifact payload (defeats the point) or reject valid-but-new tool
    shapes when a new tool is registered. Frontend carries its own union
    type in ``client.ts::ChatMessage``. Applies to every ``*/chat`` route.

    ``scene_context`` carries:
      - ``area``               — selected bbox, folded into the system prompt
      - ``datasets``           — list of currently-loaded dataset names
      - ``auto_label_summary`` — one-line recap of the last auto-label run
      - ``polygon_features``   — optional GeoJSON feature array; enables the
        ``query_polygon`` tool to look up individual labels

    Tool calls go through ``geo_tools.execute_tool``. Pass ``"tools": "none"``
    in the body to disable tool use for a turn.
    """
    if not await gemma_client.health_check():
        raise HTTPException(
            503,
            f"LLM unreachable at {gemma_client.GEMMA_BASE_URL} (runtime={gemma_client.LLM_RUNTIME}).",
        )

    messages = payload.get("messages", [])
    scene = payload.get("scene_context") or {}
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
        "You are the Gemma 4 geo-assistant embedded in Roger Studio, a "
        "remote-sensing workbench. The user is a geoscientist. Give concise, "
        "technically precise answers. When they ask about a polygon or class, "
        "refer to the scene context provided or call the available tools. "
        "Prefer bulleted answers for multi-part questions. Refuse to invent "
        "satellite data you haven't been shown — if unsure, say so or call a "
        "tool to fetch real data.\n\n"
        "TOOL USE — NON-NEGOTIABLE:\n"
        "1. When the user asks you to run a model, map tiles, compute NDVI, "
        "look up the OlmoEarth catalog, or fetch polygon stats, you MUST emit "
        "a tool_call. Do NOT describe the tool in prose.\n"
        "2. NEVER say 'I have initiated', 'please stand by', 'I will notify "
        "you', 'the process is running', or any variant. There is no async "
        "notification channel. If you did not emit a tool_call in the SAME "
        "turn, the tool DID NOT RUN.\n"
        "3. NEVER invent backend failures ('service integration issue', "
        "'endpoint not found', 'tool execution failed') to explain missing "
        "results. If a tool actually failed, the error is returned to you as "
        "a role=tool message with an 'error' field — cite that verbatim. "
        "Otherwise the tool was not called.\n"
        "4. Plain-text replies are only for: (a) a direct answer that needs "
        "no tool, or (b) a concise clarifying question. Everything else goes "
        "through tool_calls.\n"
        "5. If the user's request is actionable with one of the tools, call "
        "it in this turn — do not announce intent, do not defer.\n"
        "6. If a tool result carries an 'artifacts' array, DO NOT paste the "
        "raw data (CSV rows, full GeoJSON, long tables) into your reply. "
        "Summarize in 2-3 sentences (trend, notable months, units) and tell "
        "the user the file is available for download — the UI renders the "
        "download pill automatically from the artifact metadata.\n"
        + scene_block
    )

    chat_messages = [{"role": "system", "content": system_prompt}, *messages]

    async def _execute(name: str, arguments: str) -> dict:
        # Bounded tool-call ceiling — see ``cloud_chat.py`` for rationale.
        return await geo_tools.execute_tool_with_timeout(name, arguments, scene)

    try:
        resp = await gemma_client.chat_with_tools(
            messages=chat_messages,
            tools=geo_tools.TOOL_SCHEMAS,
            execute_tool_fn=_execute,
            temperature=0.4,
            max_tokens=1500,
            tool_choice=tool_choice,
        )
    except gemma_client.GemmaUnavailable as e:
        raise HTTPException(503, str(e))

    # Collect artifacts across every tool that fired this turn so the UI
    # renders them uniformly as download pills regardless of which tool
    # produced them. Tools that don't emit artifacts just contribute
    # nothing to this list.
    artifacts_this_turn: list[dict] = []
    for call in resp.get("tool_calls_made", []):
        result = call.get("result") or {}
        for a in (result.get("artifacts") or []):
            artifacts_this_turn.append(a)

    return {
        "role": "assistant",
        "content": resp.get("content", ""),
        "reasoning_content": resp.get("reasoning_content", ""),
        "tool_calls_made": resp.get("tool_calls_made", []),
        "iterations": resp.get("iterations", 1),
        "stopped_reason": resp.get("stopped_reason"),
        "usage": resp.get("usage", {}),
        "phantom_tool_call_fixed": resp.get("phantom_tool_call_fixed", False),
        "empty_retry_fixed": resp.get("empty_retry_fixed", False),
        "artifacts": artifacts_this_turn,
    }


@router.get("/auto-label/methods")
async def list_methods() -> dict:
    """List available auto-labeling methods and models."""
    return {
        "methods": {
            "spectral": {
                "available": True,
                "description": "K-means clustering on spectral bands",
                "gpu_required": False,
            },
            "tipsv2": {
                "available": TIPSV2_AVAILABLE,
                "description": "TIPSv2 zero-shot text-prompted classification (CVPR'26)",
                "gpu_required": False,
                "gpu_recommended": True,
                "models": [
                    "google/tipsv2-b14",
                    "google/tipsv2-l14",
                    "google/tipsv2-g14",
                ],
                "default_classes": [{"name": c["name"], "prompt": c["prompt"]} for c in DEFAULT_CLASSES] if TIPSV2_AVAILABLE else [],
            },
            "samgeo": {
                "available": SAMGEO_AVAILABLE,
                "description": "SamGeo (Segment Anything Geospatial) — pixel-accurate boundaries via Meta SAM",
                "gpu_required": False,
                "gpu_recommended": True,
                "models": ["sam_vit_h"],
                "notes": "Downloads ~2.4GB SAM checkpoint on first use. 30-60s per tile on CPU.",
            },
        },
        "validator": {
            "llm": {
                "description": "Multimodal LLM validator — validates/refines any pipeline's polygons using patch + elevation + weather + neighbor context.",
                "model": gemma_client.GEMMA_MODEL,
                "runtime": gemma_client.LLM_RUNTIME,
                "base_url": gemma_client.GEMMA_BASE_URL,
                "endpoint": "POST /auto-label/{filename}/validate",
                "notes": "Works with any OpenAI-compatible endpoint (Ollama, vLLM, or a cloud provider). See docs/llm-setup.md.",
            },
        },
    }
