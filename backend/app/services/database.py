"""SQLite persistence for uploaded datasets."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import DatasetInfo

DB_PATH = Path(__file__).parent.parent.parent / "geoenv.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            filename TEXT PRIMARY KEY,
            format TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    # Projects — the root container borrowed from OlmoEarth Studio's design.
    # A project bundles a named session: selected area, imagery layers,
    # label set, custom tags, and arbitrary additional state serialized as
    # JSON. This is the smallest adoption of OE Studio's resource model
    # that unblocks save / load / cite workflows without requiring a full
    # Prediction/PredictionResult/Task/Labelset decomposition up-front.
    # Schema can evolve (e.g., split labels into their own table) by
    # reading state_json and migrating in place — the JSON blob is a
    # forward-compatible escape hatch.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            state_json TEXT NOT NULL
        )
    """)
    # Index for the `search` endpoint's common filters — most listings
    # sort by updated_at desc (recent first); name search is LIKE-based.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Projects CRUD — thin wrapper over sqlite, model-agnostic so ``schemas.py``
# can define the Pydantic types without dragging db logic into them. Router
# passes plain dicts in/out and lets FastAPI + Pydantic validate both sides.
# ---------------------------------------------------------------------------

def save_project(project_id: str, name: str, description: str | None, state_json: str) -> dict:
    """Upsert a project. Returns the stored row as a dict."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    # Preserve created_at on update; coalesce to ``now`` on insert.
    existing = conn.execute(
        "SELECT created_at FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    created_at = existing["created_at"] if existing else now
    conn.execute(
        """INSERT OR REPLACE INTO projects
           (id, name, description, created_at, updated_at, state_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (project_id, name, description, created_at, now, state_json),
    )
    conn.commit()
    conn.close()
    return {
        "id": project_id,
        "name": name,
        "description": description,
        "created_at": created_at,
        "updated_at": now,
        "state_json": state_json,
    }


def get_project(project_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, name, description, created_at, updated_at, state_json "
        "FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_projects(
    name_contains: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Search pattern from OE Studio — POST /search passes filter + pagination.
    Keep it simple: one LIKE filter on name, updated_at-desc order, capped
    at ``limit``. Expand with more filter operators (``eq``, ``inc``, ``ne``)
    once there are enough projects to warrant them."""
    conn = _get_conn()
    if name_contains:
        rows = conn.execute(
            "SELECT id, name, description, created_at, updated_at, state_json "
            "FROM projects WHERE name LIKE ? "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (f"%{name_contains}%", limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, description, created_at, updated_at, state_json "
            "FROM projects "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_project(project_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def save_dataset(info: DatasetInfo, filepath: str) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO datasets
           (filename, format, size_bytes, metadata_json, filepath, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            info.filename,
            info.format.value if hasattr(info.format, "value") else str(info.format),
            info.size_bytes,
            info.model_dump_json(),
            filepath,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_datasets() -> list[DatasetInfo]:
    conn = _get_conn()
    rows = conn.execute("SELECT metadata_json FROM datasets ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    results = []
    for row in rows:
        data = json.loads(row["metadata_json"])
        results.append(DatasetInfo(**data))
    return results


def get_dataset(filename: str) -> DatasetInfo | None:
    conn = _get_conn()
    row = conn.execute("SELECT metadata_json FROM datasets WHERE filename = ?", (filename,)).fetchone()
    conn.close()
    if row is None:
        return None
    return DatasetInfo(**json.loads(row["metadata_json"]))


def delete_dataset(filename: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM datasets WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0
