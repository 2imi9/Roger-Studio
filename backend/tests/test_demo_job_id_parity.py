"""Parity test: `olmoearth_demos._make_job_id` must match
`olmoearth_inference._make_job_id` byte-for-byte.

The audit caught this: demos duplicates the hash function (to stay a
pure-data module without pulling torch/rasterio at import time). If either
copy drifts — e.g. someone tweaks the JSON serialization in one file but
forgets the other — demo tile URLs silently point at job_ids that real
inference never produces, so the compare-mode tiles come back empty with
no error surface. This test fails fast when the two diverge.
"""
from __future__ import annotations

from app.services.olmoearth_demos import _make_job_id as demo_make_job_id
from app.services.olmoearth_demos import DEMO_PAIRS
from app.services.olmoearth_inference import _make_job_id as infer_make_job_id


def test_hashers_agree_on_a_demo_spec() -> None:
    pair = DEMO_PAIRS[0]
    for side in (pair.a, pair.b):
        spec = side.to_inference_spec()
        assert demo_make_job_id(spec) == infer_make_job_id(spec), (
            f"job_id drift: demo={demo_make_job_id(spec)!r} "
            f"infer={infer_make_job_id(spec)!r} for spec={spec!r}"
        )


def test_hashers_agree_across_all_demo_pairs() -> None:
    for pair in DEMO_PAIRS:
        for label, side in (("a", pair.a), ("b", pair.b)):
            spec = side.to_inference_spec()
            assert demo_make_job_id(spec) == infer_make_job_id(spec), (
                f"pair={pair.id} side={label} job_id drift"
            )


def test_hashers_agree_on_edge_case_specs() -> None:
    # Empty, nested, int/float mix — the kinds of shapes that might behave
    # differently across json.dumps invocations if someone changes flags.
    cases = [
        {},
        {"a": 1, "b": 2.0, "c": "x"},
        {"bbox": {"west": -1.0, "south": 2.0, "east": 3.0, "north": 4.0}},
        {"nested": {"a": [1, 2, 3], "b": {"c": None}}},
    ]
    for spec in cases:
        assert demo_make_job_id(spec) == infer_make_job_id(spec), (
            f"edge-case drift for spec={spec!r}"
        )
