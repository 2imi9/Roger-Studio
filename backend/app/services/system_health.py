"""System-health prechecks gating resource-heavy inference jobs.

The backend refuses to launch a chunked inference when free RAM is below
a safety threshold. Rationale: during this project's development, running
a large-AOI inference while Chrome + a local vLLM container were already
resident twice produced an OOM-triggered **force shutdown** on the host
(Windows killed userland when the working set spilled into swap past the
point the OS could recover). A cheap ``psutil.virtual_memory()`` check at
job submission turns that into a visible ``503`` response instead.

The threshold is intentionally conservative. One chunked inference over a
~500 km² AOI peaks at ~1.5 GB of resident state (rasterio/GDAL buffers +
torch allocator + numpy scratch), so leaving 3 GB of headroom absorbs
that plus typical browser/IDE overhead without false positives.

Override:
  * ``OE_MIN_FREE_RAM_GB=<float>`` sets a custom threshold (e.g. ``1.5``
    on a machine with 8 GB total where even 3 GB feels too strict).
  * ``OE_MIN_FREE_RAM_GB=0`` disables the precheck entirely (tests,
    containerised environments with dedicated RAM).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

# Default: 3 GB. Rationale documented above. Overridable per-deployment
# via the OE_MIN_FREE_RAM_GB env var so operators don't have to fork the
# code for different hardware profiles.
DEFAULT_MIN_FREE_RAM_GB = 3.0

_env_override = os.environ.get("OE_MIN_FREE_RAM_GB")
if _env_override is not None:
    try:
        MIN_FREE_RAM_GB = float(_env_override)
    except ValueError:
        logger.warning(
            "OE_MIN_FREE_RAM_GB=%r not parseable as float — falling back to %s GB",
            _env_override, DEFAULT_MIN_FREE_RAM_GB,
        )
        MIN_FREE_RAM_GB = DEFAULT_MIN_FREE_RAM_GB
    else:
        logger.info(
            "OE_MIN_FREE_RAM_GB=%.2f — RAM precheck threshold overridden",
            MIN_FREE_RAM_GB,
        )
else:
    MIN_FREE_RAM_GB = DEFAULT_MIN_FREE_RAM_GB


class InsufficientMemoryError(RuntimeError):
    """Raised when a job's RAM precheck fails.

    The router layer catches this and returns HTTP 503 with the Error's
    message so the frontend / curl user sees a concrete "free up N GB"
    recommendation rather than a silent force-shutdown later.
    """


class ClientDisconnectedError(RuntimeError):
    """Raised when an in-flight chunked inference is aborted because the
    HTTP client closed its connection.

    Rationale: with ``chunk_sem(4)`` running per request and 300 s per-chunk
    timeouts, a single abandoned PCA tab on a 25-chunk AOI keeps the worker
    grinding for ~30 minutes after the user already gave up. Across a few
    hours of UI exploration this stacks up into the 35 GB / 100% CPU / 37 hr
    runaway state observed in dev. Polling ``request.is_disconnected()``
    every few seconds and cancelling the in-flight ``asyncio.gather`` cuts
    the leak at the source.

    The router catches this and returns HTTP 499 (nginx convention for
    "client closed request") so log scraping can distinguish abandonment
    from real failures.
    """


class CircuitBreakerTrippedError(RuntimeError):
    """Raised when an in-flight chunked inference aborts after too many
    consecutive chunk failures.

    Rationale: when the user's ISP is dropping TLS handshakes at 30 %+,
    every chunk of a 25-chunk job fails after burning the per-chunk
    timeout (300 s × 25 = 125 min). Detecting "3 fails in a row" and
    bailing converts that 2-hour waste into a ~15-minute failure with a
    clear "network unstable — partial results cached, retry later"
    response. Partial scene-cache writes survive, so a retry resumes
    with some chunks already on disk.

    Carries ``processed``, ``failed``, ``total`` so the caller / UI can
    surface "completed N of M chunks before aborting" and users know
    how close a retry is to success.
    """

    def __init__(self, *, processed: int, failed: int, total: int, threshold: int) -> None:
        self.processed = processed
        self.failed = failed
        self.total = total
        self.threshold = threshold
        super().__init__(
            f"Circuit breaker tripped: {threshold} chunks failed in a row "
            f"(out of {total} total, {processed} succeeded, {failed} failed). "
            f"Partial results are cached on disk — retry when network "
            f"stabilizes and cached chunks will skip re-fetch."
        )


class AOISizeExceededError(RuntimeError):
    """Raised when a submitted AOI would produce more chunks than the
    deployment is configured to accept.

    Every 5 km × 5 km chunk costs ~1.5 GB RAM peak + ~300 MB network + up
    to 300 s wall time. Letting users submit 70-chunk AOIs from a laptop
    was the root cause of two force-shutdowns earlier in this project.
    Refusing oversized AOIs at submit time (before any fetch) turns a
    2-hour machine-crashing grind into an instant HTTP 413 with a
    concrete "draw smaller or deploy to Azure" message.

    Carries ``chunks``, ``max_chunks``, ``chunk_size_m``, and the approx.
    AOI area in km² so the caller / UI can render "your 1200 km² AOI
    would produce 48 chunks — max is 20 — halve the bbox".
    """

    def __init__(
        self,
        *,
        chunks: int,
        max_chunks: int,
        chunk_size_m: int,
        aoi_area_km2: float,
        max_aoi_area_km2: float,
    ) -> None:
        self.chunks = chunks
        self.max_chunks = max_chunks
        self.chunk_size_m = chunk_size_m
        self.aoi_area_km2 = aoi_area_km2
        self.max_aoi_area_km2 = max_aoi_area_km2
        super().__init__(
            f"AOI too large for this deployment: would produce "
            f"{chunks} chunks (max {max_chunks}). "
            f"Approx {aoi_area_km2:.0f} km² vs max {max_aoi_area_km2:.0f} km² "
            f"at {chunk_size_m} m chunks. "
            f"Draw a smaller rectangle, or raise the ceiling via "
            f"OE_MAX_CHUNKS=<N> (only on a machine with dedicated RAM). "
            f"For production-grade demos at country scale, deploy the "
            f"backend to an Azure VM in the same region as Microsoft's "
            f"Planetary Computer data."
        )


def max_chunks_threshold() -> int:
    """Max chunks a single inference job is allowed to produce.

    Rationale: each chunk is a native-resolution fetch + forward pass
    that peaks at ~1.5 GB RAM + 72 HTTP reads. 20 chunks ≈ 500 km² at
    the default 5 km chunk size — comfortably within a 16 GB laptop's
    headroom AND completes inside the 5 min HTTP client timeout on a
    working network.

    Override via ``OE_MAX_CHUNKS`` env var. ``0`` disables the guard —
    appropriate on Azure VMs with dedicated RAM + intra-datacenter S2
    fetch speeds, inappropriate on laptops.
    """
    env = os.environ.get("OE_MAX_CHUNKS")
    if env is None:
        return 20
    try:
        value = int(env)
    except ValueError:
        logger.warning(
            "OE_MAX_CHUNKS=%r not parseable as int — falling back to 20", env,
        )
        return 20
    if value < 0:
        logger.warning(
            "OE_MAX_CHUNKS=%d < 0 — treating as disabled (0)", value,
        )
        return 0
    return value


def check_aoi_size_or_raise(
    chunks: int, chunk_size_m: int, aoi_area_km2: float,
) -> None:
    """Raise :class:`AOISizeExceededError` when the AOI would produce
    more chunks than ``max_chunks_threshold()`` allows.

    Called AFTER ``plan_chunks`` in both chunked orchestrators. Cheap —
    pure int comparison, no network / IO.

    ``aoi_area_km2`` is a caller-computed approximation surfaced in the
    error message so the user sees the size they actually drew (not just
    the chunk count). Use the lat-corrected cosine projection the
    orchestrator already has handy.
    """
    max_chunks = max_chunks_threshold()
    if max_chunks <= 0:
        return  # guard disabled
    if chunks > max_chunks:
        # Area per chunk ≈ (chunk_size_m ** 2) / 1e6 km². Max area = max_chunks × that.
        max_aoi_area_km2 = max_chunks * (chunk_size_m ** 2) / 1e6
        raise AOISizeExceededError(
            chunks=chunks,
            max_chunks=max_chunks,
            chunk_size_m=chunk_size_m,
            aoi_area_km2=aoi_area_km2,
            max_aoi_area_km2=max_aoi_area_km2,
        )


def per_chunk_min_free_ram_gb() -> float:
    """Minimum free RAM required to start a single chunk's fetch + forward.

    The submit-time precheck (``MIN_FREE_RAM_GB``, default 3 GB) only
    measures *once* before dispatch. A 4-chunk job that passes the precheck
    can still balloon to 6 GB resident mid-run if the user's other apps
    grow concurrently. Re-checking before each chunk lets us bail fast —
    failing one chunk is much cheaper than letting the OS swap-thrash or
    invoke OOM.

    Default 1 GB is the OS's "danger zone" floor on a 64 GB host: below
    1 GB available, Windows starts paging and the desktop becomes
    unresponsive within seconds. Override via
    ``OE_PER_CHUNK_MIN_FREE_RAM_GB``. ``0`` disables the per-chunk recheck.
    """
    env = os.environ.get("OE_PER_CHUNK_MIN_FREE_RAM_GB")
    if env is None:
        return 1.0
    try:
        value = float(env)
    except ValueError:
        logger.warning(
            "OE_PER_CHUNK_MIN_FREE_RAM_GB=%r not parseable as float — "
            "falling back to 1.0",
            env,
        )
        return 1.0
    if value < 0:
        logger.warning(
            "OE_PER_CHUNK_MIN_FREE_RAM_GB=%.2f < 0 — treating as disabled (0)",
            value,
        )
        return 0.0
    return value


def chunk_ram_ok() -> tuple[bool, MemoryStatus]:
    """Cheap per-chunk readout: returns ``(ok, status)``.

    Used inside the chunked orchestrators right before the heavy fetch +
    forward kicks off — so a job that started with healthy RAM but is now
    near the OS's swap-thrash threshold aborts the *next* chunk instead of
    the entire host.

    The threshold lives in ``per_chunk_min_free_ram_gb()`` (default 1 GB).
    Threshold ``0`` makes this always return ``True`` — appropriate when
    operators want the legacy "trust the submit-time check" behaviour.
    """
    threshold = per_chunk_min_free_ram_gb()
    status = measure_memory(threshold_gb=threshold)
    if threshold <= 0:
        return True, status
    return status.available_gb >= threshold, status


def circuit_breaker_threshold() -> int:
    """Max consecutive chunk failures before aborting a job.

    Overridable via ``OE_CIRCUIT_BREAKER_FAILS`` for operators who want
    more / fewer retries before giving up. ``0`` disables the breaker
    (legacy behaviour — exhaust every chunk).
    """
    env = os.environ.get("OE_CIRCUIT_BREAKER_FAILS")
    if env is None:
        return 3
    try:
        value = int(env)
    except ValueError:
        logger.warning(
            "OE_CIRCUIT_BREAKER_FAILS=%r not parseable as int — falling back to 3",
            env,
        )
        return 3
    if value < 0:
        logger.warning(
            "OE_CIRCUIT_BREAKER_FAILS=%d < 0 — treating as disabled (0)", value,
        )
        return 0
    return value


def circuit_breaker_min_total_fails() -> int:
    """Minimum total failures before the fractional-rate breaker trips.

    Companion to ``circuit_breaker_fail_rate_threshold``. The consecutive
    breaker (``circuit_breaker_threshold``) catches bursty drops — e.g.
    network goes down for 30 s. The fractional rule catches the slow-burn
    case: 50 % flake rate where every other chunk succeeds, so the
    consecutive counter never reaches its threshold but half the work is
    being wasted. Without this rule, a 25-chunk job on a flaky link
    grinds through all 25 (12 wasted) before completing.

    Default 5 — small enough to bail before too many cycles wasted, large
    enough to avoid false positives on a job that just had bad luck early.
    Override via ``OE_BREAKER_FAIL_RATE_MIN_FAILS``. ``0`` disables the
    fractional rule entirely (only consecutive trip remains).
    """
    env = os.environ.get("OE_BREAKER_FAIL_RATE_MIN_FAILS")
    if env is None:
        return 5
    try:
        value = int(env)
    except ValueError:
        logger.warning(
            "OE_BREAKER_FAIL_RATE_MIN_FAILS=%r not parseable as int — "
            "falling back to 5", env,
        )
        return 5
    if value < 0:
        logger.warning(
            "OE_BREAKER_FAIL_RATE_MIN_FAILS=%d < 0 — treating as disabled (0)",
            value,
        )
        return 0
    return value


def should_trip_fractional(
    *,
    failures: int,
    successes: int,
    min_total_fails: int,
    rate_threshold: float,
) -> bool:
    """Pure decision function for the fractional-rate breaker rule.

    Returns ``True`` when:
      * ``min_total_fails`` is enabled (> 0), AND
      * ``rate_threshold`` is enabled (> 0), AND
      * we've seen at least ``min_total_fails`` failures so far, AND
      * the failure rate exceeds ``rate_threshold``.

    Lifted into a module-level helper (instead of inlined in the
    orchestrator) so the trip logic is unit-testable without spinning up
    the whole chunked pipeline. Both ``_run_chunked_aoi_inference`` and
    ``_run_chunked_embedding_export`` call this from their respective
    ``_record_chunk_outcome`` closures.
    """
    if min_total_fails <= 0 or rate_threshold <= 0:
        return False
    if failures < min_total_fails:
        return False
    total = failures + successes
    if total <= 0:
        return False
    return (failures / total) > rate_threshold


def circuit_breaker_fail_rate_threshold() -> float:
    """Fractional failure rate (0.0–1.0) above which the breaker trips,
    once the minimum-total-fails floor is met.

    Default 0.5 — half the chunks failing means the network is genuinely
    unreliable for this AOI's geography / time window, and continuing
    just burns resources for an unusable result. Override via
    ``OE_BREAKER_FAIL_RATE_THRESHOLD``. Values <= 0 or > 1 disable the
    rule.
    """
    env = os.environ.get("OE_BREAKER_FAIL_RATE_THRESHOLD")
    if env is None:
        return 0.5
    try:
        value = float(env)
    except ValueError:
        logger.warning(
            "OE_BREAKER_FAIL_RATE_THRESHOLD=%r not parseable as float — "
            "falling back to 0.5", env,
        )
        return 0.5
    if value <= 0 or value > 1:
        logger.warning(
            "OE_BREAKER_FAIL_RATE_THRESHOLD=%.2f outside (0, 1] — "
            "treating as disabled (0)", value,
        )
        return 0.0
    return value


@dataclass(frozen=True)
class MemoryStatus:
    """Snapshot of host memory at precheck time. Surfaced in error bodies
    + debug logs so users can see exactly what was measured."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent: float
    threshold_gb: float

    def ok(self) -> bool:
        """True if available >= threshold. Threshold <= 0 disables the
        check entirely (useful for tests / trusted container
        environments)."""
        if self.threshold_gb <= 0:
            return True
        return self.available_gb >= self.threshold_gb

    def describe(self) -> str:
        return (
            f"host memory: available={self.available_gb:.2f} GB, "
            f"used={self.used_gb:.2f} GB ({self.percent:.0f}%), "
            f"total={self.total_gb:.2f} GB, "
            f"required={self.threshold_gb:.2f} GB"
        )


def measure_memory(threshold_gb: float | None = None) -> MemoryStatus:
    """Snapshot host memory + the threshold we're comparing against.

    Separated from ``check_memory`` so tests + diagnostic endpoints can
    inspect the reading without raising.
    """
    t = MIN_FREE_RAM_GB if threshold_gb is None else threshold_gb
    vm = psutil.virtual_memory()
    return MemoryStatus(
        total_gb=vm.total / (1024 ** 3),
        available_gb=vm.available / (1024 ** 3),
        used_gb=vm.used / (1024 ** 3),
        percent=vm.percent,
        threshold_gb=t,
    )


def check_memory_or_raise(threshold_gb: float | None = None) -> MemoryStatus:
    """Raise :class:`InsufficientMemoryError` if free RAM is below the
    threshold. Returns the snapshot on success so callers can log it.

    Call this at the **very start** of any resource-heavy inference entry
    point (chunked AOI inference, embedding export). Cheap — one syscall
    via psutil.
    """
    status = measure_memory(threshold_gb)
    if not status.ok():
        raise InsufficientMemoryError(
            f"Insufficient free RAM to launch inference safely. "
            f"{status.describe()}. "
            f"Free up memory (close browser tabs, stop Docker containers, "
            f"unload other ML models) and retry. To lower the threshold, "
            f"set OE_MIN_FREE_RAM_GB in the backend env. "
            f"To disable the precheck, set OE_MIN_FREE_RAM_GB=0."
        )
    logger.info("RAM precheck ok — %s", status.describe())
    return status
