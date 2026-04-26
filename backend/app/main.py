from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, artifacts, auto_label, claude_chat, cloud_chat, env_data, explain_raster, gemini_chat, olmoearth, openai_chat, polygon_stats, projects, stac_imagery, upload
from app.services.database import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


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
