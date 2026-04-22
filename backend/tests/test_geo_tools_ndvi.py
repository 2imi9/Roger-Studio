"""Unit tests for the real ``query_ndvi_timeseries`` tool in geo_tools."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.geo_tools import _subtract_months  # noqa: E402


def test_subtract_months_within_year() -> None:
    from datetime import date

    assert _subtract_months(date(2026, 4, 19), 1) == date(2026, 3, 19)
    assert _subtract_months(date(2026, 4, 19), 3) == date(2026, 1, 19)


def test_subtract_months_wraps_year() -> None:
    from datetime import date

    assert _subtract_months(date(2026, 2, 15), 3) == date(2025, 11, 15)


def test_subtract_months_clamps_day_to_shorter_month() -> None:
    from datetime import date

    # Jan 31 → Feb (only 28 / 29 days).
    result = _subtract_months(date(2026, 3, 31), 1)
    assert result.year == 2026 and result.month == 2
    assert result.day in (28, 29)


# ---------------------------------------------------------------------------
# Network-gated: hits PC for monthly S2 scenes.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_ndvi_timeseries_returns_monthly_stats() -> None:
    from app.services.geo_tools import _tool_query_ndvi_timeseries

    # Vegetated bbox over Knoxville forest — should produce non-trivial NDVI.
    args = {
        "bbox": {"west": -83.95, "south": 35.95, "east": -83.93, "north": 35.97},
        "months": 2,
        "max_size_px": 48,
    }
    res = await _tool_query_ndvi_timeseries(args, scene_context={})
    assert res["status"] in {"ok", "empty"}
    assert res["months_requested"] == 2
    assert len(res["timeseries"]) == 2
    assert res["formula"] == "NDVI = (B08 - B04) / (B08 + B04)"

    for entry in res["timeseries"]:
        assert "month" in entry
        if entry.get("status") == "ok":
            assert -1.0 <= entry["ndvi_mean"] <= 1.0
            assert 0 < entry["n_pixels"]
            assert entry["scene_id"].startswith("S2")
