"""Unit tests for :mod:`app.services.worldcover`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.schemas import BBox  # noqa: E402
from app.services.worldcover import (  # noqa: E402
    WORLDCOVER_CLASSES,
    WorldCoverResult,
)


def test_worldcover_classes_are_complete_and_unique() -> None:
    codes = [c["code"] for c in WORLDCOVER_CLASSES]
    # ESA WorldCover publishes 11 classes.
    assert len(codes) == 11
    assert len(set(codes)) == 11
    # All required keys present.
    for c in WORLDCOVER_CLASSES:
        assert {"code", "name", "color"} <= c.keys()
        assert c["color"].startswith("#") and len(c["color"]) == 7


def test_as_percentages_normalizes_and_sorts_descending() -> None:
    result = WorldCoverResult(
        total_pixels=100,
        counts={10: 40, 50: 35, 80: 25},
        year=2021,
    )
    pct = result.as_percentages()
    assert [p["code"] for p in pct] == [10, 50, 80]
    assert [p["percentage"] for p in pct] == [40.0, 35.0, 25.0]


def test_as_percentages_drops_unknown_codes() -> None:
    result = WorldCoverResult(
        total_pixels=100,
        counts={10: 50, 999: 50},  # 999 isn't a real WorldCover code.
        year=2021,
    )
    pct = result.as_percentages()
    assert len(pct) == 1
    assert pct[0]["code"] == 10


def test_as_percentages_handles_zero_pixels() -> None:
    result = WorldCoverResult(total_pixels=0, counts={}, year=2021)
    assert result.as_percentages() == []


# ---------------------------------------------------------------------------
# Network-gated: reads ESA WorldCover COG from Planetary Computer.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_classify_land_cover_seattle_is_mostly_urban() -> None:
    from app.services.worldcover import classify_land_cover

    bbox = BBox(west=-122.35, south=47.60, east=-122.32, north=47.63)
    result = await classify_land_cover(bbox, year=2021)
    assert result.total_pixels > 0

    pct = result.as_percentages()
    by_name = {p["name"]: p["percentage"] for p in pct}
    # Downtown Seattle is dominated by Built-up.
    assert by_name.get("Built-up", 0) > 50.0
    # Elliott Bay falls inside this bbox, so Water should register.
    assert by_name.get("Permanent water bodies", 0) > 1.0
