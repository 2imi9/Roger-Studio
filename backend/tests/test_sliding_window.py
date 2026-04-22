"""Tests for ``run_ft_tiled_inference`` — FT forward pass over a grid of
non-overlapping training-size windows."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services import olmoearth_model as M  # noqa: E402
from app.services.olmoearth_ft import FTHeadSpec, FTModel  # noqa: E402


def _fake_ft_model(
    task_type: str,
    num_classes: int = 4,
    patch_size: int = 4,
    repo_id: str = "allenai/OlmoEarth-v1-FT-Fake",
    metadata: dict | None = None,
) -> MagicMock:
    """A minimal FTModel-shaped mock that returns scripted forward output.

    We bypass ``FTModel.forward`` by patching the wrapper; callers only
    interact with the public ``run_ft_*_inference`` functions so this is
    enough to exercise the tiling + stitching logic.
    """
    m = MagicMock(spec=FTModel)
    m.repo_id = repo_id
    m.metadata = metadata or {}
    m.spec = FTHeadSpec(
        task_type=task_type,
        num_classes=num_classes,
        decoder_key="fake",
        weight_shape=(num_classes, 768),
        # FTHeadSpec.__post_init__ now asserts this is non-empty and starts
        # with "model." — the fixture never exercises weight-copy (the
        # forward is mocked), so any placeholder that satisfies the guard
        # is fine. Matches the convention ``_infer_head_spec`` produces
        # for the ``model.decoders.<key>`` pattern.
        head_prefix="model.decoders.fake",
    )

    # Default forward: segmentation → per-pixel argmax raster that depends on
    # the input's mean value (so we can distinguish windows).
    def _forward(sample, patch_size):
        h, w = sample.sentinel2_l2a.shape[1], sample.sentinel2_l2a.shape[2]
        out_h = h // patch_size
        out_w = w // patch_size
        # Use the first sentinel2 pixel to pick a class — lets the test
        # inject per-window content and check stitching.
        val = int(sample.sentinel2_l2a[0, 0, 0, 0, 0].item()) % num_classes
        if task_type == "segmentation":
            logits = torch.zeros(1, num_classes, out_h, out_w)
            logits[0, val] = 1.0
            return {
                "task_type": "segmentation",
                "logits": logits,
                "probs": torch.softmax(logits, dim=1),
                "prediction": logits.argmax(dim=1),
            }
        if task_type == "classification":
            logits = torch.zeros(1, num_classes)
            logits[0, val] = 10.0  # very confident
            return {
                "task_type": "classification",
                "logits": logits,
                "probs": torch.softmax(logits, dim=-1),
                "prediction": logits.argmax(dim=-1),
            }
        # regression — use the raw input value, not modulo (num_classes=1
        # for regression would collapse it to 0).
        raw = float(sample.sentinel2_l2a[0, 0, 0, 0, 0].item())
        return {
            "task_type": "regression",
            "logits": torch.tensor([raw]),
            "prediction": torch.tensor([raw]),
        }

    m.forward.side_effect = _forward
    m.device = torch.device("cpu")
    return m


def _image_with_per_window_values(
    H: int, W: int, window_size: int, values_grid: np.ndarray
) -> np.ndarray:
    """Build a (1, H, W, 1, 12) image where each window_size×window_size
    block is filled with a constant picked from ``values_grid[i,j]``."""
    img = np.zeros((1, H, W, 1, 12), dtype=np.float32)
    n_rows, n_cols = values_grid.shape
    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * window_size
            x0 = j * window_size
            img[:, y0 : y0 + window_size, x0 : x0 + window_size, :, :] = float(values_grid[i, j])
    return img


def test_tiled_segmentation_stitches_per_window_class_rasters() -> None:
    """3×3 grid of 32-pixel windows → stitched class raster picks up each
    window's argmax class."""
    model = _fake_ft_model("segmentation", num_classes=4, patch_size=4)
    # Each window gets a value in {0..3} → maps to that argmax class.
    grid = np.array([[0, 1, 2], [3, 0, 1], [2, 3, 0]])
    img = _image_with_per_window_values(H=96, W=96, window_size=32, values_grid=grid)

    result = M.run_ft_tiled_inference(
        model, img, timestamp_dmy=(15, 6, 2024), window_size=32, patch_size=4, normalize=False,
    )
    assert result.task_type == "segmentation"
    # Output grid at patch resolution: 96/4 = 24 per side.
    assert result.class_raster.shape == (24, 24)
    # Each window contributes a (32/4)=8 per-side block at its argmax class.
    for i, row in enumerate(grid):
        for j, expected in enumerate(row):
            block = result.class_raster[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            assert int(block.min()) == int(block.max()) == int(expected)


def test_tiled_classification_is_upgraded_to_segmentation() -> None:
    """Scene-level classification + sliding_window → spatially-varying class
    map, and task_type is reported as segmentation to the caller."""
    model = _fake_ft_model("classification", num_classes=4, patch_size=4)
    grid = np.array([[0, 2], [1, 3]])
    img = _image_with_per_window_values(H=64, W=64, window_size=32, values_grid=grid)

    result = M.run_ft_tiled_inference(
        model, img, timestamp_dmy=(15, 6, 2024), window_size=32, patch_size=4, normalize=False,
    )
    assert result.task_type == "segmentation"
    assert result.class_raster.shape == (16, 16)
    # Top-left quadrant should be class 0, etc.
    assert int(result.class_raster[:8, :8].mean()) == 0
    assert int(result.class_raster[:8, 8:].mean()) == 2
    assert int(result.class_raster[8:, :8].mean()) == 1
    assert int(result.class_raster[8:, 8:].mean()) == 3


def test_tiled_regression_preserves_task_type_and_varies_spatially() -> None:
    model = _fake_ft_model(
        "regression", num_classes=1, patch_size=4,
        metadata={"value_range": [0.0, 4.0]},
    )
    grid = np.array([[0, 2], [3, 4]])
    img = _image_with_per_window_values(H=64, W=64, window_size=32, values_grid=grid)

    result = M.run_ft_tiled_inference(
        model, img, timestamp_dmy=(15, 6, 2024), window_size=32, patch_size=4, normalize=False,
    )
    assert result.task_type == "regression"
    # class_raster nulled for pure-regression output.
    assert result.class_raster is None
    # Scalar raster varies spatially.
    assert not np.allclose(result.scalar, result.scalar[0, 0])


def test_tiled_inference_falls_back_to_single_forward_when_bbox_is_smaller_than_window() -> None:
    """Image smaller than window_size → single forward pass, same as
    run_ft_inference."""
    model = _fake_ft_model("segmentation", num_classes=4, patch_size=4)
    img = np.zeros((1, 16, 16, 1, 12), dtype=np.float32)
    result = M.run_ft_tiled_inference(
        model, img, timestamp_dmy=(15, 6, 2024), window_size=32, patch_size=4, normalize=False,
    )
    # 16/4 = 4 patches per side from the single-window fallback.
    assert result.class_raster.shape == (4, 4)


def test_tiled_inference_rejects_incompatible_window_size() -> None:
    model = _fake_ft_model("segmentation", num_classes=4, patch_size=4)
    img = np.zeros((1, 64, 64, 1, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        M.run_ft_tiled_inference(
            model, img, timestamp_dmy=(15, 6, 2024),
            window_size=30, patch_size=4, normalize=False,
        )


def test_start_inference_schema_accepts_sliding_window_params() -> None:
    """The new kwargs are plumbed through to start_inference without
    breaking the signature."""
    import inspect
    from app.services.olmoearth_inference import start_inference

    sig = inspect.signature(start_inference)
    assert "sliding_window" in sig.parameters
    assert "window_size" in sig.parameters
    assert sig.parameters["sliding_window"].default is False
    assert sig.parameters["window_size"].default == 32


def test_tool_schema_exposes_sliding_window_params() -> None:
    from app.services import geo_tools

    schema = next(
        t["function"] for t in geo_tools.TOOL_SCHEMAS
        if t["function"]["name"] == "run_olmoearth_inference"
    )
    props = schema["parameters"]["properties"]
    assert "sliding_window" in props
    assert "window_size" in props
    assert props["sliding_window"]["type"] == "boolean"
    assert props["window_size"]["minimum"] == 16
    assert props["window_size"]["maximum"] == 128
