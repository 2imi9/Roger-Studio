"""Projects resource — persistent session container.

Minimal adoption of OlmoEarth Studio's Project concept (see
docs.olmoearth.allenai.org API spec under /api/v1/projects). A Project
bundles the named session state — selected area, imagery layers, label
set, custom tags, datasets — so a researcher can close the tab and
re-open the same working context later.

This is the smallest change that unblocks reproducibility without
requiring a full decomposition into Areas / Labelsets / Predictions as
separate tables. State lives in a single JSON blob per project; it can
be decomposed incrementally later by migrating blob contents into
proper tables.

Endpoints:
  POST   /api/v1/projects                create from ``ProjectWrite``
  GET    /api/v1/projects/{id}           read one
  PUT    /api/v1/projects/{id}           update (save state)
  DELETE /api/v1/projects/{id}           delete
  POST   /api/v1/projects/search         list with filters + pagination

The /search POST pattern matches OE Studio — easier to add filter
operators (``{eq, inc, ne}``) later without breaking existing callers
than stuffing them into GET query params.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException

from app.models.schemas import ProjectRead, ProjectSearchRequest, ProjectWrite
from app.services import database

router = APIRouter()


def _row_to_read(row: dict) -> ProjectRead:
    return ProjectRead(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        state=json.loads(row["state_json"]) if row["state_json"] else {},
    )


@router.post("/v1/projects", response_model=ProjectRead)
async def create_project(payload: ProjectWrite) -> ProjectRead:
    """Create a new project. Server-generated UUID; the client doesn't
    get to pick the id so concurrent create requests from two tabs can
    never collide."""
    project_id = str(uuid.uuid4())
    state_json = json.dumps(payload.state)
    row = database.save_project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        state_json=state_json,
    )
    return _row_to_read(row)


@router.get("/v1/projects/{project_id}", response_model=ProjectRead)
async def read_project(project_id: str) -> ProjectRead:
    row = database.get_project(project_id)
    if row is None:
        raise HTTPException(404, f"project {project_id!r} not found")
    return _row_to_read(row)


@router.put("/v1/projects/{project_id}", response_model=ProjectRead)
async def update_project(project_id: str, payload: ProjectWrite) -> ProjectRead:
    """Update (save) an existing project. If the id doesn't exist yet,
    fall through to create — makes the "Save As" flow single-endpoint
    from the frontend's perspective. Created_at is preserved in the
    database layer."""
    existing = database.get_project(project_id)
    if existing is None:
        raise HTTPException(404, f"project {project_id!r} not found")
    row = database.save_project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        state_json=json.dumps(payload.state),
    )
    return _row_to_read(row)


@router.delete("/v1/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    removed = database.delete_project(project_id)
    if not removed:
        raise HTTPException(404, f"project {project_id!r} not found")
    return {"deleted": project_id}


@router.post("/v1/projects/search", response_model=list[ProjectRead])
async def search_projects(payload: ProjectSearchRequest) -> list[ProjectRead]:
    rows = database.list_projects(
        name_contains=payload.name_contains,
        limit=payload.limit,
        offset=payload.offset,
    )
    return [_row_to_read(r) for r in rows]
