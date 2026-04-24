"""Unit tests for the RAM precheck that gates chunked inference jobs.

Coverage:
  1. ``MemoryStatus.ok`` threshold logic (including the 0 = disabled case)
  2. ``check_memory_or_raise`` raises the right error type + message
  3. The router endpoint ``GET /olmoearth/system-health`` returns the
     snapshot shape the frontend needs to render a pre-click warning
  4. The router endpoints ``POST /olmoearth/infer`` and
     ``POST /olmoearth/export-embedding`` both translate
     ``InsufficientMemoryError`` to HTTP 503 + Retry-After
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services import system_health  # noqa: E402
from app.services.system_health import (  # noqa: E402
    AOISizeExceededError,
    CircuitBreakerTrippedError,
    InsufficientMemoryError,
    MemoryStatus,
    check_aoi_size_or_raise,
    check_memory_or_raise,
    circuit_breaker_threshold,
    max_chunks_threshold,
    measure_memory,
)


# ---------------------------------------------------------------------------
# Pure-function: MemoryStatus.ok()
# ---------------------------------------------------------------------------


def test_memory_status_ok_when_available_above_threshold() -> None:
    s = MemoryStatus(
        total_gb=16.0, available_gb=8.0, used_gb=8.0,
        percent=50.0, threshold_gb=3.0,
    )
    assert s.ok() is True


def test_memory_status_not_ok_when_available_below_threshold() -> None:
    s = MemoryStatus(
        total_gb=16.0, available_gb=2.5, used_gb=13.5,
        percent=84.0, threshold_gb=3.0,
    )
    assert s.ok() is False


def test_memory_status_threshold_zero_disables_check() -> None:
    """Threshold of 0 means the operator has opted out — always allow."""
    s = MemoryStatus(
        total_gb=16.0, available_gb=0.1, used_gb=15.9,
        percent=99.0, threshold_gb=0.0,
    )
    assert s.ok() is True


def test_memory_status_describe_includes_all_numbers() -> None:
    """The UI / error body surfaces this string to the user — every
    number the operator needs to make a decision must be in it."""
    s = MemoryStatus(
        total_gb=16.0, available_gb=2.5, used_gb=13.5,
        percent=84.0, threshold_gb=3.0,
    )
    desc = s.describe()
    assert "available=2.50" in desc
    assert "used=13.50" in desc
    assert "total=16.00" in desc
    assert "required=3.00" in desc


# ---------------------------------------------------------------------------
# measure_memory uses psutil — lightly mock to keep tests deterministic.
# ---------------------------------------------------------------------------


def test_measure_memory_returns_snapshot_with_psutil_values() -> None:
    class _FakeVm:
        total = 16 * 1024 ** 3
        available = 5 * 1024 ** 3
        used = 11 * 1024 ** 3
        percent = 68.75

    with patch.object(system_health.psutil, "virtual_memory", return_value=_FakeVm):
        status = measure_memory()

    assert status.total_gb == pytest.approx(16.0, rel=1e-6)
    assert status.available_gb == pytest.approx(5.0, rel=1e-6)
    assert status.used_gb == pytest.approx(11.0, rel=1e-6)
    assert status.percent == pytest.approx(68.75, rel=1e-6)
    # Threshold defaults to module-level constant unless caller overrides.
    assert status.threshold_gb == system_health.MIN_FREE_RAM_GB


def test_measure_memory_honors_caller_threshold_override() -> None:
    class _FakeVm:
        total = 16 * 1024 ** 3
        available = 5 * 1024 ** 3
        used = 11 * 1024 ** 3
        percent = 68.75

    with patch.object(system_health.psutil, "virtual_memory", return_value=_FakeVm):
        status = measure_memory(threshold_gb=1.5)
    assert status.threshold_gb == 1.5


# ---------------------------------------------------------------------------
# check_memory_or_raise — the guard wired into the chunked orchestrator.
# ---------------------------------------------------------------------------


def test_check_memory_or_raise_allows_when_ample() -> None:
    class _FakeVm:
        total = 16 * 1024 ** 3
        available = 10 * 1024 ** 3
        used = 6 * 1024 ** 3
        percent = 37.5

    with patch.object(system_health.psutil, "virtual_memory", return_value=_FakeVm):
        # Must return the status (caller can log) and NOT raise.
        status = check_memory_or_raise(threshold_gb=3.0)
    assert isinstance(status, MemoryStatus)
    assert status.ok() is True


def test_check_memory_or_raise_raises_with_actionable_message() -> None:
    class _FakeVm:
        total = 8 * 1024 ** 3
        available = 1 * 1024 ** 3     # Only 1 GB free — below 3 GB threshold
        used = 7 * 1024 ** 3
        percent = 87.5

    with patch.object(system_health.psutil, "virtual_memory", return_value=_FakeVm):
        with pytest.raises(InsufficientMemoryError) as exc_info:
            check_memory_or_raise(threshold_gb=3.0)

    msg = str(exc_info.value)
    # The error body tells the user exactly what to do — keep this
    # contract because the frontend surfaces this text verbatim.
    assert "available=1.00 GB" in msg
    assert "required=3.00 GB" in msg
    assert "Free up memory" in msg
    assert "OE_MIN_FREE_RAM_GB" in msg  # escape hatch documented in error


def test_check_memory_or_raise_threshold_zero_always_passes() -> None:
    class _FakeVm:
        total = 8 * 1024 ** 3
        available = 0      # Zero free RAM
        used = 8 * 1024 ** 3
        percent = 100.0

    with patch.object(system_health.psutil, "virtual_memory", return_value=_FakeVm):
        # With threshold=0 the check is opted out — must NOT raise even
        # when available_gb is absurdly low.
        check_memory_or_raise(threshold_gb=0.0)


# ---------------------------------------------------------------------------
# Circuit breaker — threshold parsing + error construction.
# ---------------------------------------------------------------------------


def test_breaker_threshold_default_is_three(monkeypatch) -> None:
    monkeypatch.delenv("OE_CIRCUIT_BREAKER_FAILS", raising=False)
    assert circuit_breaker_threshold() == 3


def test_breaker_threshold_from_valid_env(monkeypatch) -> None:
    monkeypatch.setenv("OE_CIRCUIT_BREAKER_FAILS", "5")
    assert circuit_breaker_threshold() == 5


def test_breaker_threshold_zero_disables(monkeypatch) -> None:
    """Operators can disable the breaker entirely for legacy behaviour."""
    monkeypatch.setenv("OE_CIRCUIT_BREAKER_FAILS", "0")
    assert circuit_breaker_threshold() == 0


def test_breaker_threshold_invalid_falls_back(monkeypatch) -> None:
    """Garbage env var shouldn't crash the backend — fall back to 3."""
    monkeypatch.setenv("OE_CIRCUIT_BREAKER_FAILS", "not-a-number")
    assert circuit_breaker_threshold() == 3


def test_breaker_threshold_negative_treated_as_disabled(monkeypatch) -> None:
    """Negative values don't make sense — treat as 0."""
    monkeypatch.setenv("OE_CIRCUIT_BREAKER_FAILS", "-5")
    assert circuit_breaker_threshold() == 0


def test_circuit_breaker_error_carries_stats() -> None:
    """The error body must include the stats the UI + retry layer need."""
    err = CircuitBreakerTrippedError(
        processed=7, failed=3, total=25, threshold=3,
    )
    assert err.processed == 7
    assert err.failed == 3
    assert err.total == 25
    assert err.threshold == 3
    msg = str(err)
    # Operators / UI render this message directly.
    assert "3 chunks failed in a row" in msg
    assert "out of 25 total" in msg
    assert "7 succeeded" in msg
    assert "3 failed" in msg
    assert "Partial results are cached" in msg  # reassures users about retry


# ---------------------------------------------------------------------------
# AOI size guardrail — threshold parsing + error construction + check helper.
# ---------------------------------------------------------------------------


def test_max_chunks_threshold_default_is_twenty(monkeypatch) -> None:
    monkeypatch.delenv("OE_MAX_CHUNKS", raising=False)
    assert max_chunks_threshold() == 20


def test_max_chunks_threshold_from_valid_env(monkeypatch) -> None:
    monkeypatch.setenv("OE_MAX_CHUNKS", "100")
    assert max_chunks_threshold() == 100


def test_max_chunks_threshold_zero_disables(monkeypatch) -> None:
    """Azure deployments with dedicated RAM can opt out of the guard."""
    monkeypatch.setenv("OE_MAX_CHUNKS", "0")
    assert max_chunks_threshold() == 0


def test_max_chunks_threshold_invalid_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("OE_MAX_CHUNKS", "garbage")
    assert max_chunks_threshold() == 20


def test_max_chunks_threshold_negative_treated_as_disabled(monkeypatch) -> None:
    monkeypatch.setenv("OE_MAX_CHUNKS", "-5")
    assert max_chunks_threshold() == 0


def test_aoi_size_error_carries_stats() -> None:
    """Error body must include chunks/area/thresholds so the UI can
    render a concrete ``your 1200 km² AOI > max 500 km²`` message."""
    err = AOISizeExceededError(
        chunks=48, max_chunks=20,
        chunk_size_m=5000,
        aoi_area_km2=1200.0,
        max_aoi_area_km2=500.0,
    )
    assert err.chunks == 48
    assert err.max_chunks == 20
    assert err.chunk_size_m == 5000
    assert err.aoi_area_km2 == 1200.0
    assert err.max_aoi_area_km2 == 500.0
    msg = str(err)
    assert "48 chunks" in msg
    assert "max 20" in msg
    assert "1200 km²" in msg
    assert "500 km²" in msg
    assert "OE_MAX_CHUNKS" in msg       # escape hatch documented
    assert "Azure" in msg               # production recommendation included


def test_check_aoi_size_or_raise_allows_small_aoi(monkeypatch) -> None:
    """A small AOI (few chunks) must NOT raise."""
    monkeypatch.setenv("OE_MAX_CHUNKS", "20")
    check_aoi_size_or_raise(
        chunks=3, chunk_size_m=5000, aoi_area_km2=75.0,
    )  # returns None silently


def test_check_aoi_size_or_raise_rejects_oversized(monkeypatch) -> None:
    """A big AOI (more chunks than threshold) must raise the specific
    error type with stats populated."""
    monkeypatch.setenv("OE_MAX_CHUNKS", "20")
    with pytest.raises(AOISizeExceededError) as exc_info:
        check_aoi_size_or_raise(
            chunks=48, chunk_size_m=5000, aoi_area_km2=1200.0,
        )
    err = exc_info.value
    assert err.chunks == 48
    assert err.max_chunks == 20
    assert err.max_aoi_area_km2 == pytest.approx(500.0)  # 20 × (5000² / 1e6)


def test_check_aoi_size_or_raise_disabled_when_threshold_zero(monkeypatch) -> None:
    """With OE_MAX_CHUNKS=0 the guard is disabled — even absurd chunk
    counts must pass. Covers the Azure-deployment path."""
    monkeypatch.setenv("OE_MAX_CHUNKS", "0")
    # 1000 chunks — way over any sane default — should still pass.
    check_aoi_size_or_raise(
        chunks=1000, chunk_size_m=5000, aoi_area_km2=25_000.0,
    )


# ---------------------------------------------------------------------------
# Router integration — ensures the precheck becomes a real HTTP 503 + that
# the /system-health endpoint surfaces the snapshot to the frontend.
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_system_health_endpoint_returns_memory_snapshot(test_client) -> None:
    """GET /olmoearth/system-health returns the psutil snapshot in the
    exact shape the frontend will consume."""
    r = test_client.get("/api/olmoearth/system-health")
    assert r.status_code == 200
    body = r.json()
    # Contract fields the frontend + docs depend on.
    for key in ("total_gb", "available_gb", "used_gb", "percent_used",
                "threshold_gb", "ok"):
        assert key in body, f"missing key {key!r} in system-health response"
    assert isinstance(body["ok"], bool)
    assert body["total_gb"] > 0


def test_infer_endpoint_returns_503_on_insufficient_memory(test_client) -> None:
    """POST /olmoearth/infer surfaces InsufficientMemoryError as 503
    with Retry-After set, not as a generic 500."""
    async def _raise_mem(**_kw):
        raise InsufficientMemoryError(
            "Insufficient free RAM to launch inference safely. "
            "host memory: available=1.50 GB, used=14.50 GB (90%), "
            "total=16.00 GB, required=3.00 GB."
        )
    # Patch start_inference (the service-layer entry the router calls).
    from app.services import olmoearth_inference as oi
    with patch.object(oi, "start_inference", side_effect=_raise_mem):
        r = test_client.post(
            "/api/olmoearth/infer",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "30"
    assert "available=1.50 GB" in r.text


def test_infer_endpoint_returns_503_on_circuit_breaker_trip(test_client) -> None:
    """POST /olmoearth/infer surfaces CircuitBreakerTrippedError as
    503 with Retry-After=60 (longer than the memory precheck's 30 —
    network stabilization takes longer than RAM freeing)."""
    async def _raise_breaker(**_kw):
        raise CircuitBreakerTrippedError(
            processed=2, failed=3, total=25, threshold=3,
        )
    from app.services import olmoearth_inference as oi
    with patch.object(oi, "start_inference", side_effect=_raise_breaker):
        r = test_client.post(
            "/api/olmoearth/infer",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "60"
    # Users see the full breaker message — proves retry is useful ("cached").
    assert "Partial results are cached" in r.text
    assert "3 chunks failed in a row" in r.text


def test_export_embedding_endpoint_returns_503_on_circuit_breaker_trip(test_client) -> None:
    """POST /olmoearth/export-embedding same as /infer — breaker trip
    maps to 503 + Retry-After=60."""
    async def _raise_breaker(**_kw):
        raise CircuitBreakerTrippedError(
            processed=1, failed=3, total=12, threshold=3,
        )

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    from app.services import olmoearth_inference as oi
    with patch.object(
        oi, "_run_chunked_embedding_export", side_effect=_raise_breaker,
    ), patch(
        "app.routers.olmoearth.olmoearth_model.load_encoder",
        return_value=(_FakeModel(), "cpu"),
    ):
        r = test_client.post(
            "/api/olmoearth/export-embedding",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "60"
    assert "Partial results are cached" in r.text


def test_infer_endpoint_returns_413_on_oversized_aoi(test_client) -> None:
    """POST /olmoearth/infer converts AOISizeExceededError to HTTP 413
    Payload Too Large. NO Retry-After (retrying same AOI won't help —
    user must shrink it)."""
    async def _raise_aoi(**_kw):
        raise AOISizeExceededError(
            chunks=48, max_chunks=20,
            chunk_size_m=5000,
            aoi_area_km2=1200.0,
            max_aoi_area_km2=500.0,
        )
    from app.services import olmoearth_inference as oi
    with patch.object(oi, "start_inference", side_effect=_raise_aoi):
        r = test_client.post(
            "/api/olmoearth/infer",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 2.0, "north": 46.0},
                "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
            },
        )
    assert r.status_code == 413
    # Retry-After must NOT be present — this is a user-shrink-your-AOI
    # issue, not a wait-a-minute-and-retry issue.
    assert "retry-after" not in {k.lower() for k in r.headers.keys()}
    # Body carries the concrete sizes so UI can render actionable message.
    assert "48 chunks" in r.text
    assert "1200 km²" in r.text
    assert "500 km²" in r.text


def test_export_embedding_endpoint_returns_413_on_oversized_aoi(test_client) -> None:
    """POST /olmoearth/export-embedding — same 413 pattern as /infer."""
    async def _raise_aoi(**_kw):
        raise AOISizeExceededError(
            chunks=36, max_chunks=20,
            chunk_size_m=5000,
            aoi_area_km2=900.0,
            max_aoi_area_km2=500.0,
        )

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    from app.services import olmoearth_inference as oi
    with patch.object(
        oi, "_run_chunked_embedding_export", side_effect=_raise_aoi,
    ), patch(
        "app.routers.olmoearth.olmoearth_model.load_encoder",
        return_value=(_FakeModel(), "cpu"),
    ):
        r = test_client.post(
            "/api/olmoearth/export-embedding",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 2.0, "north": 46.0},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )
    assert r.status_code == 413
    assert "retry-after" not in {k.lower() for k in r.headers.keys()}
    assert "36 chunks" in r.text
    assert "900 km²" in r.text


def test_export_embedding_endpoint_returns_503_on_insufficient_memory(test_client) -> None:
    """POST /olmoearth/export-embedding surfaces InsufficientMemoryError
    as 503 + Retry-After just like /infer does."""
    async def _raise_mem(**_kw):
        raise InsufficientMemoryError(
            "Insufficient free RAM to launch inference safely. "
            "host memory: available=1.00 GB, used=15.00 GB (94%), "
            "total=16.00 GB, required=3.00 GB."
        )

    class _FakeModel:
        repo_id = "allenai/OlmoEarth-v1-Tiny"

    from app.services import olmoearth_inference as oi
    with patch.object(
        oi, "_run_chunked_embedding_export", side_effect=_raise_mem,
    ), patch(
        "app.routers.olmoearth.olmoearth_model.load_encoder",
        return_value=(_FakeModel(), "cpu"),
    ):
        r = test_client.post(
            "/api/olmoearth/export-embedding",
            json={
                "bbox": {"west": 0.0, "south": 45.0, "east": 0.01, "north": 45.01},
                "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
            },
        )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "30"
    assert "available=1.00 GB" in r.text


# ---------------------------------------------------------------------------
# Per-chunk RAM recheck (P1 host-safety guard).
#
# Background: the submit-time RAM precheck is a one-shot at job entry. A
# 4-chunk job that passes the precheck can still balloon to 6+ GB resident
# mid-run if the user's other apps grow concurrently. Failing fast at the
# *next* chunk is much cheaper than letting the OS swap-thrash.
# ---------------------------------------------------------------------------


def test_per_chunk_min_free_ram_default(monkeypatch):
    monkeypatch.delenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", raising=False)
    assert system_health.per_chunk_min_free_ram_gb() == 1.0


def test_per_chunk_min_free_ram_env_override(monkeypatch):
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "0.5")
    assert system_health.per_chunk_min_free_ram_gb() == 0.5


def test_per_chunk_min_free_ram_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "not-a-float")
    assert system_health.per_chunk_min_free_ram_gb() == 1.0


def test_per_chunk_min_free_ram_negative_disables(monkeypatch):
    """Negative threshold is treated as ``0`` (disabled), matching the
    pattern used by ``MIN_FREE_RAM_GB`` and ``OE_MAX_CHUNKS``."""
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "-1.5")
    assert system_health.per_chunk_min_free_ram_gb() == 0.0


def test_chunk_ram_ok_returns_true_when_disabled(monkeypatch):
    """Threshold = 0 disables the gate — ok is True regardless of available."""
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "0")
    ok, status = system_health.chunk_ram_ok()
    assert ok is True
    assert status.threshold_gb == 0.0


def test_chunk_ram_ok_passes_when_available_above_threshold(monkeypatch):
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "0.001")  # 1 MB — must pass
    ok, status = system_health.chunk_ram_ok()
    assert ok is True
    assert status.available_gb >= 0.001


def test_chunk_ram_ok_fails_when_available_below_threshold(monkeypatch):
    """Setting the threshold above total host RAM forces a fail. We pick
    1 PB (10⁶ GB) so this can't accidentally pass on any real machine."""
    monkeypatch.setenv("OE_PER_CHUNK_MIN_FREE_RAM_GB", "1000000")
    ok, status = system_health.chunk_ram_ok()
    assert ok is False
    assert status.threshold_gb == 1000000.0
    assert status.available_gb < 1000000.0


# ---------------------------------------------------------------------------
# Fractional-failure-rate breaker (P1).
#
# The original consecutive-fails breaker misses the slow-burn case: a 50 %
# flake rate keeps resetting the consecutive counter (success-fail-success-
# fail) while still wasting half the budget. The new rule trips when total
# failures cross a floor *and* the failure rate exceeds a threshold.
# ---------------------------------------------------------------------------


def test_breaker_min_total_fails_default(monkeypatch):
    monkeypatch.delenv("OE_BREAKER_FAIL_RATE_MIN_FAILS", raising=False)
    assert system_health.circuit_breaker_min_total_fails() == 5


def test_breaker_min_total_fails_env_override(monkeypatch):
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_MIN_FAILS", "10")
    assert system_health.circuit_breaker_min_total_fails() == 10


def test_breaker_min_total_fails_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_MIN_FAILS", "n/a")
    assert system_health.circuit_breaker_min_total_fails() == 5


def test_breaker_min_total_fails_zero_disables(monkeypatch):
    """``0`` is the explicit "disable the fractional rule" knob."""
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_MIN_FAILS", "0")
    assert system_health.circuit_breaker_min_total_fails() == 0


def test_breaker_min_total_fails_negative_disables(monkeypatch):
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_MIN_FAILS", "-7")
    assert system_health.circuit_breaker_min_total_fails() == 0


def test_breaker_fail_rate_threshold_default(monkeypatch):
    monkeypatch.delenv("OE_BREAKER_FAIL_RATE_THRESHOLD", raising=False)
    assert system_health.circuit_breaker_fail_rate_threshold() == 0.5


def test_breaker_fail_rate_threshold_env_override(monkeypatch):
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_THRESHOLD", "0.75")
    assert system_health.circuit_breaker_fail_rate_threshold() == 0.75


def test_breaker_fail_rate_threshold_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("OE_BREAKER_FAIL_RATE_THRESHOLD", "fifty-percent")
    assert system_health.circuit_breaker_fail_rate_threshold() == 0.5


def test_breaker_fail_rate_threshold_out_of_range_disables(monkeypatch):
    """Values outside (0, 1] are nonsensical for a fractional rate; refuse
    them rather than silently letting the breaker behave oddly."""
    for bad in ["0", "-0.2", "1.5", "10"]:
        monkeypatch.setenv("OE_BREAKER_FAIL_RATE_THRESHOLD", bad)
        assert system_health.circuit_breaker_fail_rate_threshold() == 0.0, bad


# Behavioral tests for the fractional-rate breaker decision (pure helper).


def test_should_trip_fractional_under_min_total_does_not_trip():
    """4 failures, threshold=5 — must NOT trip even at 100% fail rate."""
    assert system_health.should_trip_fractional(
        failures=4, successes=0, min_total_fails=5, rate_threshold=0.5,
    ) is False


def test_should_trip_fractional_at_min_total_with_high_rate_trips():
    """5 fails, 0 success, default 50% threshold → 100% fail rate, trip."""
    assert system_health.should_trip_fractional(
        failures=5, successes=0, min_total_fails=5, rate_threshold=0.5,
    ) is True


def test_should_trip_fractional_50_50_does_not_trip_at_50_threshold():
    """5 fails, 5 success → exactly 50%, threshold strict ``>`` so no trip."""
    assert system_health.should_trip_fractional(
        failures=5, successes=5, min_total_fails=5, rate_threshold=0.5,
    ) is False


def test_should_trip_fractional_just_over_50_trips():
    """6 fails, 5 success → 6/11 ≈ 0.545, trips at 0.5 threshold."""
    assert system_health.should_trip_fractional(
        failures=6, successes=5, min_total_fails=5, rate_threshold=0.5,
    ) is True


def test_should_trip_fractional_high_success_count_does_not_trip():
    """5 fails, 50 success → 5/55 ≈ 9% — happy network with isolated drops."""
    assert system_health.should_trip_fractional(
        failures=5, successes=50, min_total_fails=5, rate_threshold=0.5,
    ) is False


def test_should_trip_fractional_disabled_via_min_total():
    """min_total_fails=0 disables the rule entirely."""
    assert system_health.should_trip_fractional(
        failures=100, successes=0, min_total_fails=0, rate_threshold=0.5,
    ) is False


def test_should_trip_fractional_disabled_via_rate():
    """rate_threshold=0 also disables the rule entirely (treated as 'off')."""
    assert system_health.should_trip_fractional(
        failures=100, successes=0, min_total_fails=5, rate_threshold=0.0,
    ) is False


def test_should_trip_fractional_zero_total_does_not_divide_by_zero():
    """Defensive — failures=0 successes=0 is the no-data case."""
    assert system_health.should_trip_fractional(
        failures=0, successes=0, min_total_fails=5, rate_threshold=0.5,
    ) is False
