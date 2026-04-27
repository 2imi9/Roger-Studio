import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, artifacts, auto_label, claude_chat, cloud_chat, env_data, explain_raster, gemini_chat, olmoearth, openai_chat, polygon_stats, projects, stac_imagery, upload
from app.services import olmoearth_inference, olmoearth_model
from app.services.database import init_db

logger = logging.getLogger(__name__)


def _idle_timeout_s() -> float:
    """How long a cached model can sit idle before we evict it from VRAM.

    Default 1800 s (30 min). 0 disables idle eviction entirely (the
    LRU cache still kicks in on memory pressure). Values < 60 s are
    clamped to 60 s — anything tighter would thrash the cold-load
    overhead of 5–15 s on every inference.
    """
    raw = os.environ.get("OE_MODEL_IDLE_TIMEOUT_S")
    if raw is None:
        return 1800.0
    try:
        v = float(raw)
    except ValueError:
        logger.warning(
            "OE_MODEL_IDLE_TIMEOUT_S=%r not parseable; using default 1800 s",
            raw,
        )
        return 1800.0
    if v <= 0:
        return 0.0
    return max(60.0, v)


async def _idle_eviction_loop() -> None:
    """Periodic task: evict models idle longer than the configured
    threshold so the GPU fan calms down between inference bursts.

    Runs every 60 s. Cancelled cleanly when the FastAPI lifespan exits
    (the asyncio.CancelledError propagates up the task tree). Best-
    effort: any exception inside an iteration is logged and the loop
    keeps running.
    """
    timeout_s = _idle_timeout_s()
    if timeout_s == 0:
        logger.info("idle eviction disabled (OE_MODEL_IDLE_TIMEOUT_S=0)")
        return
    logger.info("idle eviction loop started (timeout=%.0fs)", timeout_s)
    while True:
        try:
            await asyncio.sleep(60.0)
            evicted = olmoearth_model.evict_idle_models(timeout_s)
            if evicted:
                logger.info("idle eviction: dropped %d model(s) from VRAM", evicted)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("idle eviction loop iteration failed: %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    # Rehydrate the in-memory ``_jobs`` registry from disk so a backend
    # restart doesn't lose previously-computed real-Pytorch results. The
    # original behavior was: backend restart → empty ``_jobs`` → next
    # /infer for any cached job_id re-runs from scratch even though the
    # disk scene cache might be warm. We observed this exact failure
    # tonight when a typed-error fix triggered uvicorn auto-reload and
    # wiped a 4-minute-old real Mangrove + FL Keys result. With this
    # hook, the same restart now repopulates _jobs on boot and tile
    # serving / cache-hit paths see the prior result transparently.
    olmoearth_inference._load_jobs_from_disk_into_memory()

    # Idle-eviction loop: drops cached models from VRAM after N min
    # without use. Without this the user reported the 5090 fan
    # ramping during sleep because the 20 GB Python process kept
    # holding the model on GPU.
    eviction_task = asyncio.create_task(_idle_eviction_loop())

    try:
        yield
    finally:
        eviction_task.cancel()
        try:
            await eviction_task
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(
    title="GeoEnv Studio API",
    version="0.2.0",
    description="Micro-environment 3D visualization backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Without expose_headers the browser strips everything except a tiny
    # safelist on cross-origin responses — so the frontend's fetch() on
    # /olmoearth/infer-tile couldn't read our X-Inference-Kind /
    # X-Inference-Stub-Reason headers even though the backend set them.
    # Listing them here lets SplitMap's pollReady tell the user "this side
    # is a preview stub, not real inference" rather than silently serving
    # a stubbed tile and looking like the app just glitched.
    expose_headers=[
        "Cache-Control",
        "ETag",
        "X-Inference-Kind",
        "X-Inference-Stub-Reason",
    ],
)

app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(env_data.router, prefix="/api", tags=["env-data"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(auto_label.router, prefix="/api", tags=["auto-label"])
app.include_router(polygon_stats.router, prefix="/api", tags=["polygon-stats"])
app.include_router(stac_imagery.router, prefix="/api", tags=["stac-imagery"])
app.include_router(olmoearth.router, prefix="/api", tags=["olmoearth"])
app.include_router(cloud_chat.router, prefix="/api", tags=["cloud-chat"])
app.include_router(claude_chat.router, prefix="/api", tags=["claude-chat"])
app.include_router(gemini_chat.router, prefix="/api", tags=["gemini-chat"])
app.include_router(openai_chat.router, prefix="/api", tags=["openai-chat"])
app.include_router(artifacts.router, prefix="/api", tags=["artifacts"])
app.include_router(explain_raster.router, prefix="/api", tags=["explain-raster"])
# Projects mount under /api (OE Studio uses /api/v1/projects; the router
# owns the /v1/projects path prefix so the full URL becomes /api/v1/projects).
app.include_router(projects.router, prefix="/api", tags=["projects"])


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}
