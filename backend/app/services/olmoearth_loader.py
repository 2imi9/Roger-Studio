"""Fast loader for OlmoEarth weights + datasets.

Wraps ``huggingface_hub.snapshot_download`` in a background task so the UI
can hit a Load button and see status transitions without blocking the
event loop. A single in-memory ``_tasks`` map tracks start/end/error for
each repo — single-user by design, matches the rest of the dev backend.

Cache introspection uses ``huggingface_hub.scan_cache_dir`` so any repo
already on disk (from a prior session or a manual ``hf download`` run)
reports as cached immediately.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# repo_id -> {status, error, started_ts, finished_ts, size_bytes, path}
# status: "loading" | "cached" | "error"
_tasks: dict[str, dict[str, Any]] = {}

# Singular-connection policy: only one HF download is the "active" one at any
# time. Starting a new load cancels the currently-active task and evicts its
# entry from `_tasks` (so the UI reverts to a Load button for the displaced
# repo). The underlying thread inside `asyncio.to_thread` can't be force-killed
# — it runs to completion in the background and its result is discarded. Any
# bytes that finished downloading stay on disk and will show up as "cached" on
# the next `scan_cache_dir` pass; that's a feature, not a bug.
_active_repo_id: str | None = None
_active_task: asyncio.Task[None] | None = None


def _hf():
    """Import huggingface_hub lazily so backend boot doesn't fail if it's
    somehow missing at import time — we can still return a clean error."""
    import huggingface_hub  # noqa: PLC0415

    return huggingface_hub


# File extensions that count as real model weights. Anything without one of
# these in its snapshot tree is a config-only / tokenizer-only repo that
# can't run inference on its own. The audit caught this: the base repo
# ``allenai/OlmoEarth-v1-Base`` publishes only ``config.json`` — the actual
# encoder tensors ship inside the size-specific siblings (``*-Nano``,
# ``*-Tiny``, ``*-Base``-prefixed FT heads). Showing a 0-byte cache entry
# for the base repo was confusing users into reporting it as "broken".
_WEIGHT_SUFFIXES = (".pth", ".safetensors", ".ckpt", ".bin")


def _has_weights(entry: Path) -> bool:
    """True if ``entry`` (a hub cache folder like ``models--allenai--X``)
    contains at least one file with a known weight-file extension.

    Walks the snapshots tree (HF layout: blobs/ + snapshots/<rev>/) so we
    pick up weights regardless of whether they're symlinked or hydrated.
    Short-circuits on the first match — no need to stat every file.
    """
    snapshots = entry / "snapshots"
    if not snapshots.exists():
        return False
    try:
        for path in snapshots.rglob("*"):
            if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
                return True
    except OSError:
        # Unreadable symlinks on Windows without Dev Mode — treat as "no
        # weights" rather than crashing the whole scan.
        return False
    return False


def _scan_cache() -> dict[str, dict[str, Any]]:
    """Return {repo_id: {size_bytes, path, last_modified, has_weights}} for
    everything currently in the HF hub cache. Iterates the cache dir directly
    so one broken sibling repo (common on Windows without Dev Mode enabled)
    doesn't wipe the whole snapshot.
    """
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    out: dict[str, dict[str, Any]] = {}
    if not hub.exists():
        return out
    for entry in hub.iterdir():
        if not entry.is_dir() or "--" not in entry.name:
            continue
        # entry.name e.g. "models--allenai--OlmoEarth-v1-Base"
        parts = entry.name.split("--")
        if len(parts) < 3:
            continue
        repo_type = {"models": "model", "datasets": "dataset", "spaces": "space"}.get(parts[0])
        if repo_type is None:
            continue
        repo_id = f"{parts[1]}/{'-'.join(parts[2:]) if len(parts) > 3 else parts[2]}"
        # Reconstruct repo_id by splitting only on the FIRST '--' between author
        # and name: models--allenai--OlmoEarth-v1-Base → allenai/OlmoEarth-v1-Base
        name_part = entry.name[len(parts[0]) + 2:]  # strip "models--"
        first_sep = name_part.find("--")
        if first_sep == -1:
            continue
        repo_id = f"{name_part[:first_sep]}/{name_part[first_sep + 2:]}"
        try:
            size_bytes = _dir_size(entry)
            last_modified = entry.stat().st_mtime
        except OSError as e:
            logger.debug("skipping unreadable cache entry %s: %s", entry, e)
            continue
        out[repo_id] = {
            "size_bytes": size_bytes,
            "path": str(entry),
            "last_modified": last_modified,
            "repo_type": repo_type,
            "has_weights": _has_weights(entry) if repo_type == "model" else True,
        }
    return out


def status_snapshot() -> dict[str, Any]:
    """Snapshot of every OlmoEarth-related repo: cached on disk OR in progress.

    Returns ``{repos: {repo_id: {status, size_bytes, error, path}}}``. Repos
    that aren't cached and aren't loading are simply absent — the UI treats
    them as 'missing' by default.

    Config-only model repos (``has_weights == False``) are filtered out —
    these are the ``allenai/OlmoEarth-v1-Base`` entries that ship a
    ``config.json`` only (their weights live inside size-specific siblings
    like ``*-Nano`` or task-specific FT heads). The audit caught a ~0-byte
    entry for the base repo misreading as "broken" in the UI; filtering at
    the snapshot layer keeps the cache-status list clean without requiring
    any frontend change. An in-flight load still wins via the ``_tasks``
    overlay below, so a user-initiated download of a config-only repo is
    still visible while it's running.
    """
    on_disk = _scan_cache()
    out: dict[str, dict[str, Any]] = {}
    for repo_id, info in on_disk.items():
        if not ("olmoearth" in repo_id.lower() or "OlmoEarth" in repo_id):
            continue
        if info.get("repo_type") == "model" and not info.get("has_weights", True):
            logger.debug(
                "hiding config-only repo %s from cache-status (no weights)",
                repo_id,
            )
            continue
        out[repo_id] = {"status": "cached", **info}
    # Overlay in-memory tasks on top so loading/error states win over cached
    for repo_id, task in _tasks.items():
        out[repo_id] = {**out.get(repo_id, {}), **task}
    return {"repos": out}


_RETRY_DELAYS_SEC = (1.0, 3.0, 7.0)

# Per-attempt wall-clock budgets. snapshot_download has no built-in
# "hung socket" timeout — a single TLS stall can hang a thread forever.
# We enforce a ceiling via `asyncio.wait_for` around `asyncio.to_thread`.
# Datasets vary wildly in size (AWF is <100 MB, pretrain is hundreds of GB),
# so the budget for datasets is a per-attempt quota, not a whole-download
# one. If you've cached most of it already, resume will clear quickly on
# the next retry. If nothing is progressing, the timeout fires and the
# error surfaces instead of a silent infinite Loading…
_ATTEMPT_TIMEOUT_SEC = {
    "model": 20 * 60,     # 20 min — encoders + FT heads are typically <5 GB
    "dataset": 10 * 60,   # 10 min per attempt; heavy datasets resume across retries
}


def _is_transient(err: BaseException) -> bool:
    """WinError 10054 / connection resets / timeouts are retriable; auth and
    404s are not. Keep the classifier conservative: retry anything that looks
    network-shaped, give up on everything else so real bugs surface quickly.
    """
    msg = str(err).lower()
    transient_markers = (
        "10054",            # Windows reset by peer
        "10053",            # Windows connection aborted
        "10060",            # Windows connection timed out
        "connectionreset",
        "connection reset",
        "connection aborted",
        "connection forcibly closed",
        "forcibly closed by the remote host",
        "connecterror",
        "connection error",
        "connectionerror",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "remote end closed",
        "remote disconnected",
        "eof occurred in violation of protocol",
        "server disconnected",      # httpx RemoteProtocolError
        "remoteprotocolerror",
        "readtimeout",
        "readerror",
        "writeerror",
        "protocolerror",
        "incomplete",
        "broken pipe",
        "429 ",             # HF rate limit — retriable
        "500 ",
        "502 ",
        "503 ",
        "504 ",
    )
    return any(m in msg for m in transient_markers)


class NeedsDevModeError(RuntimeError):
    """Raised when snapshot_download hits WinError 1314 — Windows requires
    Developer Mode or admin privileges to create the symlinks HuggingFace's
    cache layout relies on."""


def _translate_error(err: BaseException) -> BaseException:
    """Turn cryptic platform errors into actionable ones before surfacing
    them to the UI."""
    msg = str(err)
    if "1314" in msg or "required privilege" in msg.lower():
        return NeedsDevModeError(
            "Windows needs symlink privileges to lay out the HF cache. "
            "Fix once by enabling Windows Developer Mode "
            "(Settings → System → For developers → Developer Mode) OR by "
            "running the backend as Administrator. No other model changes "
            "needed; existing downloads on disk are still usable."
        )
    return err


def _snapshot_download_sync(repo_id: str, repo_type: str, hf_token: str | None) -> str:
    """One blocking snapshot_download call — no retries, no timeout. The
    retry + timeout orchestration lives in ``_download_with_retry_async`` so
    we can use ``asyncio.wait_for`` around each attempt.
    """
    hf = _hf()
    return hf.snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token,
    )


def _finalize_download(path: str) -> dict[str, Any]:
    """Compute post-download size on disk."""
    import os  # noqa: PLC0415

    size = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return {"path": path, "size_bytes": size}


async def _download_with_retry_async(
    repo_id: str,
    repo_type: str,
    hf_token: str | None,
) -> dict[str, Any]:
    """snapshot_download with exponential-backoff retries + per-attempt
    timeout. Timeouts are treated as transient — the download resumes on the
    next attempt since HF stores partials in the cache.
    """
    timeout_sec = _ATTEMPT_TIMEOUT_SEC.get(repo_type, 10 * 60)
    last_err: BaseException | None = None
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_SEC)):
        if delay:
            logger.info("retrying %s (attempt %d) after %ss", repo_id, attempt + 1, delay)
            await asyncio.sleep(delay)
        try:
            path = await asyncio.wait_for(
                asyncio.to_thread(_snapshot_download_sync, repo_id, repo_type, hf_token),
                timeout=timeout_sec,
            )
            return await asyncio.to_thread(_finalize_download, path)
        except asyncio.TimeoutError as e:
            last_err = e
            logger.warning(
                "%s snapshot_download exceeded %ds on attempt %d",
                repo_id, timeout_sec, attempt + 1,
            )
            if attempt == len(_RETRY_DELAYS_SEC):
                raise TimeoutError(
                    f"Download of {repo_id} exceeded {timeout_sec}s per attempt "
                    f"across {attempt + 1} tries. Partial shards remain on disk "
                    f"and will resume on the next Load click."
                ) from e
            # Otherwise fall through to next retry — snapshot_download resumes
            # from where it left off on the next call.
        except Exception as e:
            last_err = e
            if not _is_transient(e) or attempt == len(_RETRY_DELAYS_SEC):
                raise _translate_error(e) from e
            logger.warning("%s load hit transient error: %s", repo_id, str(e)[:120])
    assert last_err is not None  # pragma: no cover
    raise _translate_error(last_err) from last_err


def _expected_cache_paths(repo_id: str) -> list[Path]:
    """Where HF puts a given repo on disk. Returns both model + dataset
    variants so callers don't need to know the type up-front — we just
    delete whichever one exists.
    """
    slug = repo_id.replace("/", "--")
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return [
        hub / f"models--{slug}",
        hub / f"datasets--{slug}",
        hub / f"spaces--{slug}",
    ]


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def unload_repo(repo_id: str) -> dict[str, Any]:
    """Delete a repo from the HF cache (reverts it to 'missing').

    Bypasses ``scan_cache_dir`` — that scan iterates every cached repo and
    will throw if ANY entry on disk has a broken symlink (common on Windows
    without Developer Mode). We just construct the expected path from the
    HF cache convention (``models--author--name`` / ``datasets--author--name``)
    and ``rmtree`` whichever variant exists.
    """
    global _active_repo_id, _active_task

    if _active_repo_id == repo_id and _active_task and not _active_task.done():
        _active_task.cancel()
        _active_repo_id = None

    _tasks.pop(repo_id, None)

    import shutil  # noqa: PLC0415

    for path in _expected_cache_paths(repo_id):
        if not path.exists():
            continue
        size = _dir_size(path)
        try:
            shutil.rmtree(path, ignore_errors=False)
            logger.info("unloaded %s (%s MB freed)", repo_id, round(size / 1_000_000, 1))
            return {
                "removed": True,
                "repo_id": repo_id,
                "bytes_freed": size,
                "path": str(path),
            }
        except Exception as e:
            logger.exception("unload failed for %s", repo_id)
            return {"removed": False, "repo_id": repo_id, "error": str(e)}

    return {"removed": False, "repo_id": repo_id, "error": "not in cache"}


async def start_load(
    repo_id: str,
    repo_type: str = "model",
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Kick off a background download under a **singular-connection** policy.

    Only one load may be 'active' at a time. Calling ``start_load`` for a
    different repo cancels the current active task and clears its entry from
    ``_tasks`` — the UI reverts to a Load button for the displaced repo.

    If the same ``repo_id`` is already loading, this is a no-op.
    """
    global _active_repo_id, _active_task

    task = _tasks.get(repo_id)
    if (
        task
        and task.get("status") == "loading"
        and _active_repo_id == repo_id
        and _active_task
        and not _active_task.done()
    ):
        return task

    # Displace the current active load, if any
    if _active_task and not _active_task.done():
        displaced = _active_repo_id
        _active_task.cancel()
        if displaced and displaced != repo_id and displaced in _tasks:
            # Revert displaced entry to missing so the UI renders Load again.
            # Bytes that already hit disk stay in the HF cache and will
            # surface as "cached" on the next scan_cache_dir().
            del _tasks[displaced]
            logger.info("superseded active load of %s by %s", displaced, repo_id)

    _tasks[repo_id] = {
        "status": "loading",
        "error": None,
        "started_ts": time.time(),
        "finished_ts": None,
        "size_bytes": None,
        "path": None,
        "repo_type": repo_type,
    }

    async def _run() -> None:
        try:
            result = await _download_with_retry_async(repo_id, repo_type, hf_token)
            # Only commit success if we weren't displaced mid-flight
            if _active_repo_id == repo_id:
                _tasks[repo_id] = {
                    **_tasks[repo_id],
                    "status": "cached",
                    "finished_ts": time.time(),
                    "size_bytes": result["size_bytes"],
                    "path": result["path"],
                }
                logger.info(
                    "loaded %s (%s MB) to %s",
                    repo_id,
                    round(result["size_bytes"] / 1_000_000, 1),
                    result["path"],
                )
        except asyncio.CancelledError:
            # Superseded — already cleaned up above, nothing to do
            raise
        except Exception as e:
            logger.exception("load failed for %s", repo_id)
            if _active_repo_id == repo_id:
                _tasks[repo_id] = {
                    **_tasks[repo_id],
                    "status": "error",
                    "finished_ts": time.time(),
                    "error": str(e),
                }

    _active_repo_id = repo_id
    _active_task = asyncio.create_task(_run())
    return _tasks[repo_id]
