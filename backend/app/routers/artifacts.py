"""GET /api/artifacts/{id} — stream a tool-produced file back to the UI.

Paired with ``app.services.artifacts.save_artifact``. Tools that want to
avoid flooding the chat bubble with tabular data instead attach an
``artifacts: [{id, filename, content_type, size_bytes, summary,
download_url}]`` array to their result; this router streams the real
bytes when the user clicks the download pill.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.artifacts import get_artifact

router = APIRouter()


@router.get("/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str) -> FileResponse:
    meta = get_artifact(artifact_id)
    if meta is None:
        raise HTTPException(404, f"artifact {artifact_id} not found (backend restart?)")
    if not meta.path.exists():
        raise HTTPException(410, f"artifact {artifact_id} file missing on disk")
    return FileResponse(
        path=meta.path,
        filename=meta.filename,
        media_type=meta.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{meta.filename}"',
            "Cache-Control": "public, max-age=3600",
        },
    )
