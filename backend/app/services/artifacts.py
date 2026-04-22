"""Agent-produced file artifacts: write once, serve via /api/artifacts/{id}.

Any tool that would otherwise cram a long table / GeoJSON / raster into
the chat bubble can instead:

    artifact = save_artifact(
        content=csv_bytes,
        filename="ndvi_kenya_2024.csv",
        content_type="text/csv",
        summary="12-month NDVI timeseries over the Kenyan coast bbox",
    )
    return {..., "artifacts": [artifact]}

The LLM is instructed (in the chat system prompt) to summarize in prose
and mention the download, not dump the raw data. The frontend renders
every ``artifacts[]`` entry on an assistant bubble as a clickable pill.

Storage:
  - Files land under ``tempfile.gettempdir() / "roger_artifacts"``.
  - Each artifact gets a short uuid id used in the URL and filename prefix,
    so ``GET /api/artifacts/{id}`` reads from disk deterministically.
  - A tiny ``_index`` dict tracks metadata (original filename,
    content_type, size, summary, created_at) so the /api route can stream
    the file back with correct Content-Disposition.

Retention:
  - In-memory index is process-scoped — backend restart drops pointers.
  - Files on disk are subject to a TTL (``ARTIFACT_TTL_S``, default 24 h)
    enforced by ``_sweep_expired()``. The sweep runs at most once per
    ``ARTIFACT_SWEEP_COOLDOWN_S`` and is triggered lazily on every
    ``save_artifact`` / ``get_artifact`` call — no background task
    needed, and idle servers don't accumulate sweep overhead.
  - Without the sweep, a long-running backend that emits one 50 MB
    artifact per chat turn (e.g. a large NDVI CSV + GeoJSON pair on
    every auto-label run) would accumulate gigabytes in ``/tmp`` over
    a week. The audit flagged this as a disk-fill risk.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "roger_artifacts"
_ARTIFACTS_DIR.mkdir(exist_ok=True)

# In-process metadata index keyed by artifact id.
_index: dict[str, "ArtifactMeta"] = {}

# Time-to-live for artifact files on disk. Default 24 h; override via
# ``ROGER_ARTIFACT_TTL_S`` for longer-running dev sessions or shorter
# CI dry-runs. An artifact older than this is unlinked on the next
# ``_sweep_expired()`` pass + removed from the in-memory index.
ARTIFACT_TTL_S = float(os.environ.get("ROGER_ARTIFACT_TTL_S", str(24 * 3600)))

# How often the sweep actually runs. Preserves the lazy-trigger model
# (no background task, no reaper thread, no startup hook) while avoiding
# an O(N) disk scan on every save. Default 5 min — good balance for
# chat turns that emit 0–3 artifacts each.
ARTIFACT_SWEEP_COOLDOWN_S = float(
    os.environ.get("ROGER_ARTIFACT_SWEEP_COOLDOWN_S", "300")
)

# Last-sweep timestamp. Module-level scalar — process-scoped, resets on
# restart (which is fine because the sweep would just run once more than
# strictly needed; no correctness impact).
_last_sweep_ts: float = 0.0


def _sweep_expired() -> None:
    """Delete artifacts older than ``ARTIFACT_TTL_S`` from disk + index.

    Throttled to run at most once per ``ARTIFACT_SWEEP_COOLDOWN_S`` so
    a rapid burst of ``save_artifact`` calls doesn't pay the scan cost
    repeatedly. Best-effort: individual ``unlink`` failures are logged
    and swallowed so one permission error doesn't prevent the rest of
    the sweep. Missing files (already reaped by the OS temp cleaner)
    are treated as successful expiries — we still drop the index entry.

    Also scans disk for "orphan" files (artifacts the process doesn't
    have in its index because the backend restarted mid-session) older
    than TTL — keeps ``/tmp/roger_artifacts`` bounded even across
    uvicorn reloads during development.
    """
    global _last_sweep_ts
    now = time.time()
    if (now - _last_sweep_ts) < ARTIFACT_SWEEP_COOLDOWN_S:
        return
    _last_sweep_ts = now

    cutoff = now - ARTIFACT_TTL_S
    removed_count = 0
    freed_bytes = 0

    # Pass 1 — expire entries the process still knows about.
    expired_ids = [mid for mid, m in _index.items() if m.created_at < cutoff]
    for mid in expired_ids:
        meta = _index.pop(mid, None)
        if meta is None:
            continue
        try:
            if meta.path.exists():
                freed_bytes += meta.size_bytes
                meta.path.unlink()
            removed_count += 1
        except OSError as e:
            logger.warning("artifact sweep: failed to unlink %s: %s", meta.path, e)

    # Pass 2 — orphan files on disk (restart left them behind, or
    # another process wrote here). Filesystem mtime is the fallback
    # timestamp since we don't have an index entry.
    try:
        for entry in _ARTIFACTS_DIR.iterdir():
            if not entry.is_file():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    freed_bytes += entry.stat().st_size
                    entry.unlink()
                    removed_count += 1
                except OSError as e:
                    logger.warning("artifact sweep: orphan unlink %s: %s", entry, e)
    except OSError as e:
        # Directory disappeared? Recreate so future saves work.
        logger.warning("artifact sweep: dir scan failed: %s", e)
        _ARTIFACTS_DIR.mkdir(exist_ok=True)

    if removed_count:
        logger.info(
            "artifact sweep: removed %d files, freed %.1f KB",
            removed_count, freed_bytes / 1024.0,
        )


@dataclass(frozen=True)
class ArtifactMeta:
    """What we persist per artifact. ``path`` stays server-side; everything
    else is safe to hand back to the LLM / frontend."""
    id: str
    filename: str
    content_type: str
    size_bytes: int
    summary: str
    created_at: float  # unix epoch seconds
    path: Path

    def public_dict(self) -> dict[str, Any]:
        """Trimmed view suitable for chat tool results + UI rendering."""
        d = asdict(self)
        d.pop("path", None)
        d["download_url"] = f"/api/artifacts/{self.id}"
        # Lift created_at into an iso string for the UI's human label.
        from datetime import datetime, timezone  # noqa: PLC0415
        d["created_at_iso"] = (
            datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        )
        return d


def save_artifact(
    content: bytes,
    filename: str,
    content_type: str,
    summary: str,
) -> dict[str, Any]:
    """Persist ``content`` under a new artifact id and return the public dict.

    Call sites should attach the returned dict to an ``artifacts: [...]``
    field in their tool result. The LLM will see the public fields (id,
    filename, content_type, size_bytes, summary, download_url) — and the
    system prompt tells it to cite the download rather than paste the data.
    """
    # Opportunistic TTL sweep — cheap when throttled (early-return),
    # keeps /tmp/roger_artifacts bounded without a background task.
    _sweep_expired()
    artifact_id = uuid.uuid4().hex[:12]
    # Prefix the on-disk filename with the id so a stray ls doesn't confuse
    # multiple CSVs named ``ndvi_kenya.csv`` from different sessions.
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    on_disk = _ARTIFACTS_DIR / f"{artifact_id}__{safe_name}"
    on_disk.write_bytes(content)

    meta = ArtifactMeta(
        id=artifact_id,
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        summary=summary,
        created_at=time.time(),
        path=on_disk,
    )
    _index[artifact_id] = meta
    logger.info(
        "artifact saved: id=%s filename=%s size=%d type=%s",
        artifact_id, filename, len(content), content_type,
    )
    return meta.public_dict()


def get_artifact(artifact_id: str) -> ArtifactMeta | None:
    """Return the metadata + on-disk path for ``artifact_id`` or None."""
    # Opportunistic TTL sweep on read as well, so a backend with heavy
    # download traffic but few writes still clears expired files.
    _sweep_expired()
    return _index.get(artifact_id)


def clear_artifacts() -> None:
    """Drop the in-memory index AND unlink every tracked file.

    Used by tests / shutdown hooks that want a clean slate. Previously
    only cleared the dict and relied on the OS temp reaper, which on
    Windows can take days — meaning dev machines accumulated hundreds
    of stale CSVs across restart cycles. Untracked (orphan) files stay
    because we don't know who owns them; the periodic ``_sweep_expired``
    catches those once they age past the TTL.
    """
    for meta in list(_index.values()):
        try:
            if meta.path.exists():
                meta.path.unlink()
        except OSError:
            pass
    _index.clear()
