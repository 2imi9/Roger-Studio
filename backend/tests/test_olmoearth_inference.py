"""End-to-end tests for :mod:`app.services.olmoearth_inference`.

Covers both fallback paths (stub on failure) and the full real pipeline.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services import olmoearth_inference as OI  # noqa: E402
from app.services.sentinel2_fetch import SentinelFetchError  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Drop in-memory job state between tests to keep them independent."""
    OI.clear_jobs()
    yield
    OI.clear_jobs()


def _latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int(
        (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    )
    return x, y


def test_make_job_id_is_deterministic() -> None:
    spec1 = {"bbox": {"w": 1}, "model": "foo"}
    spec2 = {"model": "foo", "bbox": {"w": 1}}  # same content, different key order
    assert OI._make_job_id(spec1) == OI._make_job_id(spec2)


def test_tile_bounds_known_example() -> None:
    # Zoom 0 tile covers the whole world.
    w, s, e, n = OI._tile_to_lonlat_bounds(0, 0, 0)
    assert w == pytest.approx(-180.0)
    assert e == pytest.approx(180.0)
    assert n == pytest.approx(85.05112878, abs=1e-6)
    assert s == pytest.approx(-85.05112878, abs=1e-6)


def test_interp_color_linear() -> None:
    stops = [("#000000", 0.0), ("#ffffff", 1.0)]
    assert OI._interp_color(stops, 0.0) == (0, 0, 0)
    assert OI._interp_color(stops, 1.0) == (255, 255, 255)
    mid = OI._interp_color(stops, 0.5)
    assert all(abs(c - 127) <= 1 for c in mid)


@pytest.mark.asyncio
async def test_start_inference_falls_back_to_stub_on_fetch_failure() -> None:
    """Real-path pre-failure -> the stub path is still served, with a reason."""
    bbox = BBox(west=-122.35, south=47.60, east=-122.32, north=47.63)

    async def boom(**_kwargs):
        raise SentinelFetchError("synthetic network outage")

    with patch.object(OI, "fetch_s2_composite", side_effect=boom):
        resp = await OI.start_inference(bbox, "allenai/OlmoEarth-v1-Nano")

    assert resp["kind"] == "stub"
    assert resp["status"] == "ready"
    assert "synthetic network outage" in resp["stub_reason"]

    # Stub tile render still works and carries the watermark signature.
    x, y = _latlon_to_tile(47.615, -122.335, 14)
    png = OI.render_tile(resp["job_id"], 14, x, y)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_tile_for_unknown_job_returns_none() -> None:
    assert OI.render_tile("doesnotexist", 0, 0, 0) is None


def test_render_tile_for_pending_job_returns_transparent_png() -> None:
    OI._jobs["pending_job"] = {
        "job_id": "pending_job",
        "spec": {"bbox": {"west": 0, "south": 0, "east": 1, "north": 1}},
        "status": "running",
        "kind": "pending",
        "colormap": "embedding",
    }
    png = OI.render_tile("pending_job", 0, 0, 0)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    from PIL import Image  # local import avoids unused-dep warnings when tests skip.
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    assert im.size == (256, 256)


def test_colorize_classes_uses_legend_palette() -> None:
    """Each class index maps to its legend hex color, with alpha=210."""
    import numpy as np

    class_raster = np.array([[0, 1], [1, 2]], dtype=np.int64)
    outside = np.zeros_like(class_raster, dtype=bool)
    colors = ["#ff0000", "#00ff00", "#0000ff"]
    rgba = OI._colorize_classes(class_raster, outside, colors)
    assert rgba.shape == (2, 2, 4)
    # (0,0) = class 0 = red
    assert tuple(rgba[0, 0]) == (255, 0, 0, 210)
    # (0,1) = class 1 = green
    assert tuple(rgba[0, 1]) == (0, 255, 0, 210)
    # (1,1) = class 2 = blue
    assert tuple(rgba[1, 1]) == (0, 0, 255, 210)


def test_colorize_classes_marks_outside_pixels_transparent() -> None:
    import numpy as np

    class_raster = np.array([[0, 1]], dtype=np.int64)
    outside = np.array([[False, True]], dtype=bool)
    rgba = OI._colorize_classes(class_raster, outside, ["#ff0000", "#00ff00"])
    assert tuple(rgba[0, 0]) == (255, 0, 0, 210)
    assert tuple(rgba[0, 1]) == (0, 0, 0, 0)


def test_colorize_classes_clips_unknown_indices_to_last_color() -> None:
    """Any class index >= n_classes gets clamped rather than crashing."""
    import numpy as np

    class_raster = np.array([[5]], dtype=np.int64)  # out of range (only 2 colors)
    outside = np.zeros_like(class_raster, dtype=bool)
    rgba = OI._colorize_classes(class_raster, outside, ["#ff0000", "#00ff00"])
    assert tuple(rgba[0, 0]) == (0, 255, 0, 210)


@pytest.mark.asyncio
async def test_render_pytorch_tile_uses_class_raster_for_segmentation() -> None:
    """A segmentation job with a 2-class raster should paint two legend colors."""
    import numpy as np
    from rasterio.transform import from_bounds

    from PIL import Image  # noqa: PLC0415

    # A tiny 4×4 class raster split diagonally into classes 0 and 1, tiled over
    # a bbox that covers exactly the WGS-84 tile z=0 x=0 y=0.
    class_raster = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    transform = from_bounds(-180.0, -85.05, 180.0, 85.05, 4, 4)

    OI._jobs["seg_job"] = {
        "job_id": "seg_job",
        "spec": {
            "bbox": {"west": -180.0, "south": -85.05, "east": 180.0, "north": 85.05},
            "model_repo_id": "allenai/OlmoEarth-v1-FT-AWF-Base",
        },
        "status": "ready",
        "kind": "pytorch",
        "task_type": "segmentation",
        "colormap": "landuse",
        "class_raster": class_raster,
        "scalar_raster": np.full((4, 4), 0.5, dtype=np.float32),
        "raster_transform": transform,
        "raster_crs": "EPSG:4326",
        "legend": {
            "kind": "segmentation",
            "classes": [
                {"index": 0, "name": "forest", "color": "#228b22"},
                {"index": 1, "name": "water", "color": "#1e90ff"},
            ],
        },
    }

    png = OI.render_tile("seg_job", 0, 0, 0)
    assert png is not None
    im = Image.open(io.BytesIO(png))
    arr = np.array(im)

    # Left half should be forest-green; right half should be water-blue.
    left_mean = arr[128, 64, :3]
    right_mean = arr[128, 192, :3]
    assert tuple(left_mean) == (34, 139, 34)    # #228b22
    assert tuple(right_mean) == (30, 144, 255)  # #1e90ff
    # Full tile opaque at the 210 alpha we paint for class pixels.
    assert arr[128, 64, 3] == 210


@pytest.mark.asyncio
async def test_render_pytorch_tile_uses_scalar_path_for_embedding() -> None:
    """Embedding / regression jobs fall back to the scalar gradient."""
    import numpy as np
    from rasterio.transform import from_bounds

    from PIL import Image  # noqa: PLC0415

    OI._jobs["emb_job"] = {
        "job_id": "emb_job",
        "spec": {
            "bbox": {"west": -180.0, "south": -85.05, "east": 180.0, "north": 85.05},
            "model_repo_id": "allenai/OlmoEarth-v1-Nano",
        },
        "status": "ready",
        "kind": "pytorch",
        "task_type": "embedding",
        "colormap": "embedding",
        "scalar_raster": np.full((4, 4), 0.0, dtype=np.float32),
        "raster_transform": from_bounds(-180.0, -85.05, 180.0, 85.05, 4, 4),
        "raster_crs": "EPSG:4326",
    }

    png = OI.render_tile("emb_job", 0, 0, 0)
    im = Image.open(io.BytesIO(png))
    arr = np.array(im)
    # scalar=0 → first gradient stop for "embedding" = #0f172a = (15, 23, 42)
    assert tuple(arr[128, 128, :3]) == (15, 23, 42)


# ---------------------------------------------------------------------------
# Network-gated: the real PC + torch pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_start_inference_real_path_returns_pytorch_kind() -> None:
    bbox = BBox(west=-122.35, south=47.60, east=-122.32, north=47.63)
    resp = await OI.start_inference(
        bbox=bbox,
        model_repo_id="allenai/OlmoEarth-v1-Nano",
        date_range="2024-06-01/2024-09-30",
        max_size_px=64,
    )
    assert resp["kind"] == "pytorch"
    assert resp["status"] == "ready"
    assert resp["scene_id"].startswith("S2")
    assert resp["embedding_dim"] == 128

    x, y = _latlon_to_tile(47.615, -122.335, 14)
    png = OI.render_tile(resp["job_id"], 14, x, y)
    assert png is not None
    # Tile is inside the bbox, so it should be non-empty + valid PNG.
    from PIL import Image  # local import avoids unused-dep warnings when tests skip.
    im = Image.open(io.BytesIO(png))
    assert im.size == (256, 256)
    # Colormapped pixels are opaque at alpha=210; transparent (=0) is the
    # out-of-bbox case.
    import numpy as np
    arr = np.array(im)
    assert arr[..., 3].max() > 0


# ---------------------------------------------------------------------------
# Global concurrency cap on chunked jobs (P0 host-safety guard).
#
# These tests verify the module-scope semaphore added to prevent the
# multi-request RAM pile-up that crashed the dev host (35 GB resident from
# overlapping PCA requests, breaker tripped per-request but couldn't see
# the global picture). The mechanism:
#   * ``_max_concurrent_jobs()`` reads OE_MAX_CONCURRENT_JOBS, default 1.
#   * ``_global_job_sem()`` lazy-inits an asyncio.Semaphore(_max).
#   * ``_with_global_job_lock`` decorator wraps both chunked orchestrators
#     so only N can be running at once across the whole process.
# ---------------------------------------------------------------------------


def test_max_concurrent_jobs_default(monkeypatch):
    monkeypatch.delenv("OE_MAX_CONCURRENT_JOBS", raising=False)
    assert OI._max_concurrent_jobs() == 1


def test_max_concurrent_jobs_env_override(monkeypatch):
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "4")
    assert OI._max_concurrent_jobs() == 4


def test_max_concurrent_jobs_garbage_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "not-a-number")
    assert OI._max_concurrent_jobs() == 1


def test_max_concurrent_jobs_zero_or_negative_clamped_to_one(monkeypatch):
    """``0`` and negatives clamp to 1 — disabling the safety guard via
    this env var would defeat its purpose, so we refuse rather than
    silently drop the protection."""
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "0")
    assert OI._max_concurrent_jobs() == 1
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "-3")
    assert OI._max_concurrent_jobs() == 1


@pytest.mark.asyncio
async def test_with_global_job_lock_serializes_when_max_is_one(monkeypatch):
    """Default config (max=1): three concurrent jobs must run one at a time.

    Without the lock, all three would be in flight simultaneously; the
    counter would peak at 3. With the lock, peak is 1.
    """
    import asyncio as _asyncio
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "1")
    OI._reset_global_job_sem_for_tests()

    in_flight = 0
    peak = 0

    @OI._with_global_job_lock
    async def fake_job(tag: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Yield the loop so the next coroutine *would* run if the sem
        # weren't holding it back. 5 ms is enough to expose any race.
        await _asyncio.sleep(0.005)
        in_flight -= 1
        return tag

    results = await _asyncio.gather(fake_job("a"), fake_job("b"), fake_job("c"))
    assert results == ["a", "b", "c"]
    assert peak == 1, f"sem should serialize, but {peak} ran in parallel"
    OI._reset_global_job_sem_for_tests()


@pytest.mark.asyncio
async def test_with_global_job_lock_allows_parallelism_when_max_is_higher(monkeypatch):
    """OE_MAX_CONCURRENT_JOBS=3: three jobs may all run concurrently."""
    import asyncio as _asyncio
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "3")
    OI._reset_global_job_sem_for_tests()

    in_flight = 0
    peak = 0

    @OI._with_global_job_lock
    async def fake_job() -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await _asyncio.sleep(0.01)
        in_flight -= 1

    await _asyncio.gather(fake_job(), fake_job(), fake_job())
    assert peak == 3, f"max=3 should allow 3-wide parallelism, got peak={peak}"
    OI._reset_global_job_sem_for_tests()


@pytest.mark.asyncio
async def test_with_global_job_lock_releases_on_exception(monkeypatch):
    """If a wrapped coroutine raises, the semaphore must still release —
    otherwise a single failed job permanently wedges the backend."""
    import asyncio as _asyncio
    monkeypatch.setenv("OE_MAX_CONCURRENT_JOBS", "1")
    OI._reset_global_job_sem_for_tests()

    @OI._with_global_job_lock
    async def fail_job() -> None:
        raise RuntimeError("simulated chunk failure")

    @OI._with_global_job_lock
    async def ok_job() -> str:
        return "second-job-ran"

    with pytest.raises(RuntimeError, match="simulated"):
        await fail_job()
    # Sem must be released; the next job must acquire and complete.
    result = await _asyncio.wait_for(ok_job(), timeout=1.0)
    assert result == "second-job-ran"
    OI._reset_global_job_sem_for_tests()


# ---------------------------------------------------------------------------
# Client-disconnect polling (P0 host-safety guard).
#
# Background: with chunk_sem(4) per request and 300 s per-chunk timeout, a
# single abandoned PCA tab on a 25-chunk AOI keeps the worker grinding for
# ~30 minutes after the user gave up. Across a session this stacks into a
# resource leak. _watch_for_disconnect polls a caller-supplied async check
# every 5 s and cancels the in-flight gather task on disconnect, so the
# orchestrator surfaces ClientDisconnectedError and the route returns 499.
# These tests target the helper directly because exercising the full
# orchestrator would require mocking the entire S2 fetch + encoder stack.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watch_for_disconnect_cancels_target_when_check_returns_true(monkeypatch):
    """Disconnect → target task is cancelled."""
    import asyncio as _asyncio
    # Use a tiny poll interval so the test runs quickly.
    monkeypatch.setattr(OI, "_DISCONNECT_POLL_INTERVAL_S", 0.01)

    async def long_work():
        await _asyncio.sleep(2.0)
        return "should-be-cancelled"

    fired = {"n": 0}

    async def disconnect_check():
        fired["n"] += 1
        # Simulate "client still here" for 2 polls, then "client gone".
        return fired["n"] >= 2

    target = _asyncio.create_task(long_work())
    watcher = _asyncio.create_task(OI._watch_for_disconnect(disconnect_check, target))

    with pytest.raises(_asyncio.CancelledError):
        await target
    await watcher  # watcher exits cleanly after cancelling
    assert fired["n"] >= 2  # check was actually polled


@pytest.mark.asyncio
async def test_watch_for_disconnect_exits_quietly_when_target_finishes(monkeypatch):
    """Happy path: target completes normally → watcher exits without cancelling."""
    import asyncio as _asyncio
    monkeypatch.setattr(OI, "_DISCONNECT_POLL_INTERVAL_S", 0.01)

    async def quick_work():
        await _asyncio.sleep(0.02)
        return "done"

    async def disconnect_check():
        return False  # client never disconnects

    target = _asyncio.create_task(quick_work())
    watcher = _asyncio.create_task(OI._watch_for_disconnect(disconnect_check, target))

    result = await target
    assert result == "done"
    # Wait for the watcher to notice target finished. With 10 ms poll, this
    # should be < 50 ms — but the watcher is also sleeping so we await it.
    await _asyncio.wait_for(watcher, timeout=1.0)
    assert not target.cancelled()


@pytest.mark.asyncio
async def test_watch_for_disconnect_survives_check_raising(monkeypatch, caplog):
    """A broken disconnect_check must not wedge the job — log + exit poll."""
    import asyncio as _asyncio
    import logging as _logging
    monkeypatch.setattr(OI, "_DISCONNECT_POLL_INTERVAL_S", 0.01)
    caplog.set_level(_logging.WARNING)

    async def long_work():
        await _asyncio.sleep(0.05)
        return "ok"

    async def broken_check():
        raise RuntimeError("disconnect probe failed")

    target = _asyncio.create_task(long_work())
    watcher = _asyncio.create_task(OI._watch_for_disconnect(broken_check, target))

    # Watcher should give up on the first exception and exit.
    await _asyncio.wait_for(watcher, timeout=1.0)
    # Target still finishes normally — watcher's failure didn't poison the job.
    result = await target
    assert result == "ok"
    assert any("disconnect_check raised" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_watch_for_disconnect_can_be_cancelled_by_caller(monkeypatch):
    """The orchestrator's finally cancels the watcher when gather completes —
    the watcher must accept cancellation cleanly."""
    import asyncio as _asyncio
    monkeypatch.setattr(OI, "_DISCONNECT_POLL_INTERVAL_S", 5.0)  # long sleep

    async def long_work():
        await _asyncio.sleep(5.0)

    async def never_disconnects():
        return False

    target = _asyncio.create_task(long_work())
    watcher = _asyncio.create_task(OI._watch_for_disconnect(never_disconnects, target))

    # Give the watcher a moment to enter its sleep, then cancel it.
    await _asyncio.sleep(0.02)
    watcher.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await watcher
    target.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await target
