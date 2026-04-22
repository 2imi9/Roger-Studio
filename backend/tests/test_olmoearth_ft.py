"""Tests for :mod:`app.services.olmoearth_ft` — FT checkpoint loading and
shape-based head reconstruction.

Offline tests build synthetic state_dicts to exercise the shape-inference
logic without network. The network-gated test loads the cached FT-Mangrove
checkpoint and verifies the full encoder + head pipeline works end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.olmoearth_ft import (  # noqa: E402
    FT_TASK_METADATA,
    FTHeadSpec,
    _ConvSegmentationHead,
    _infer_embed_dim,
    _infer_head_spec,
    _LinearClassificationHead,
    _LinearRegressionHead,
    _PerPatchLinearHead,
    class_colors_for,
    class_names_for,
    is_ft_repo,
)


# ---------------------------------------------------------------------------
# Identity / naming
# ---------------------------------------------------------------------------


def test_is_ft_repo_matches_published_naming_convention() -> None:
    assert is_ft_repo("allenai/OlmoEarth-v1-FT-Mangrove-Base")
    assert is_ft_repo("allenai/OlmoEarth-v1-FT-LFMC-Base")
    assert is_ft_repo("allenai/OlmoEarth-v1-FT-AWF-Base")
    # Base models are not FT.
    assert not is_ft_repo("allenai/OlmoEarth-v1-Base")
    assert not is_ft_repo("allenai/OlmoEarth-v1-Nano")


# ---------------------------------------------------------------------------
# Shape-based head inference
# ---------------------------------------------------------------------------


def test_infer_head_spec_classification_linear() -> None:
    """4-class classification head: Linear(768 -> 4). Mirrors FT-Mangrove."""
    sd = {
        "model.decoders.mangrove_classification.0.output_layer.weight": torch.zeros(4, 768),
        "model.decoders.mangrove_classification.0.output_layer.bias":   torch.zeros(4),
        "model.encoder.0.some.encoder.weight":                          torch.zeros(10),
    }
    spec = _infer_head_spec(sd)
    assert spec.task_type == "classification"
    assert spec.num_classes == 4
    assert spec.decoder_key == "mangrove_classification"
    assert spec.weight_shape == (4, 768)
    assert _infer_embed_dim(spec) == 768


def test_infer_head_spec_regression_linear_single_output() -> None:
    """Regression heads have num_classes==1, e.g. LFMC."""
    sd = {
        "model.decoders.lfmc.0.output_layer.weight": torch.zeros(1, 768),
        "model.decoders.lfmc.0.output_layer.bias":   torch.zeros(1),
    }
    spec = _infer_head_spec(sd)
    assert spec.task_type == "regression"
    assert spec.num_classes == 1
    assert spec.decoder_key == "lfmc"


def test_infer_head_spec_segmentation_conv() -> None:
    """10-class segmentation head: Conv2d(768 -> 10, 1x1). Mirrors FT-AWF."""
    sd = {
        "model.decoders.segment.1.layer.weight": torch.zeros(10, 768, 1, 1),
        "model.decoders.segment.1.layer.bias":   torch.zeros(10),
    }
    spec = _infer_head_spec(sd)
    assert spec.task_type == "segmentation"
    assert spec.num_classes == 10
    assert spec.decoder_key == "segment"
    assert spec.weight_shape == (10, 768, 1, 1)


def test_infer_head_spec_raises_on_unknown_decoder_shape() -> None:
    sd = {
        "model.decoders.weird.0.some_other_layer.weight": torch.zeros(5, 768),
    }
    with pytest.raises(ValueError, match="no recognized FT head"):
        _infer_head_spec(sd)


# ---------------------------------------------------------------------------
# Head modules — forward pass shapes
# ---------------------------------------------------------------------------


def test_linear_classification_head_mean_pools_and_projects() -> None:
    # (B=2, H'=4, W'=4, T=1, S=3, D=64)
    tokens = torch.randn(2, 4, 4, 1, 3, 64)
    head = _LinearClassificationHead(embed_dim=64, num_classes=7)
    out = head(tokens)
    assert out.shape == (2, 7)


def test_linear_regression_head_squeezes_last_dim() -> None:
    tokens = torch.randn(2, 4, 4, 1, 3, 64)
    head = _LinearRegressionHead(embed_dim=64, num_classes=1)
    out = head(tokens)
    # Regression returns (B,), not (B, 1).
    assert out.shape == (2,)


def test_conv_segmentation_head_produces_bchw_logits() -> None:
    # (B=2, H'=8, W'=8, T=1, S=3, D=64)
    tokens = torch.randn(2, 8, 8, 1, 3, 64)
    head = _ConvSegmentationHead(embed_dim=64, num_classes=5)
    out = head(tokens)
    assert out.shape == (2, 5, 8, 8)


# ---------------------------------------------------------------------------
# Class-name lookup
# ---------------------------------------------------------------------------


def test_class_names_for_known_mangrove_published_from_rslearn_config() -> None:
    """Mangrove class names come from olmoearth_projects olmoearth_run.yaml."""
    names, tentative = class_names_for("allenai/OlmoEarth-v1-FT-Mangrove-Base", 4)
    # Published legend order: nodata, mangrove, water, other.
    assert names == ["nodata", "mangrove", "water", "other"]
    assert tentative is False


def test_class_names_for_forest_loss_driver_full_10_class_list() -> None:
    names, tentative = class_names_for(
        "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base", 10
    )
    assert tentative is False
    assert names == [
        "agriculture", "mining", "airstrip", "road", "logging",
        "burned", "landslide", "hurricane", "river", "none",
    ]


def test_class_names_for_awf_has_nodata_as_last_class() -> None:
    names, tentative = class_names_for("allenai/OlmoEarth-v1-FT-AWF-Base", 10)
    assert tentative is False
    assert len(names) == 10
    assert names[0] == "woodland_forest"
    assert names[-1] == "nodata"


def test_class_names_for_ecosystem_has_full_110_slot_list() -> None:
    names, tentative = class_names_for(
        "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base", 110
    )
    assert tentative is False
    assert len(names) == 110
    # First slot is a tropical rainforest class.
    assert "TROPICAL" in names[0]
    # Unnamed slots fall back to ``class_N``.
    assert any(n.startswith("class_") for n in names)


def test_class_colors_for_returns_published_palette() -> None:
    """Published colors round-trip through ``class_colors_for``."""
    colors = class_colors_for("allenai/OlmoEarth-v1-FT-Mangrove-Base", 4)
    assert colors == ["#6b7280", "#94eb63", "#63d8eb", "#eba963"]


def test_class_colors_for_unknown_repo_returns_none() -> None:
    assert class_colors_for("allenai/not-a-real-ft", 3) is None


def test_class_names_for_unknown_repo_returns_placeholders() -> None:
    names, tentative = class_names_for("allenai/totally-made-up", 3)
    assert names == ["class_0", "class_1", "class_2"]
    assert tentative is True


def test_class_names_for_wrong_count_falls_back() -> None:
    # Mangrove metadata lists 4 names; requesting 5 should fall back.
    names, tentative = class_names_for("allenai/OlmoEarth-v1-FT-Mangrove-Base", 5)
    assert names == ["class_0", "class_1", "class_2", "class_3", "class_4"]
    assert tentative is True


# ---------------------------------------------------------------------------
# Network-gated: real FT-Mangrove round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_and_forward_ft_mangrove_on_synthetic_input() -> None:
    """Mangrove loads as per-patch segmentation (not scene-level
    classification) per olmoearth_projects/mangrove/model.yaml."""
    from app.services.olmoearth_ft import load_ft_model
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    model = load_ft_model("allenai/OlmoEarth-v1-FT-Mangrove-Base")
    assert model.spec.task_type == "segmentation"
    assert model.spec.num_classes == 4
    assert model.spec.decoder_key == "mangrove_classification"
    # Base model embedding dim = 768.
    assert model.spec.weight_shape == (4, 768)
    assert type(model.head).__name__ == "_PerPatchLinearHead"

    img = torch.randn(1, 32, 32, 1, 12)
    mask = torch.ones(1, 32, 32, 1, 3) * MaskValue.ONLINE_ENCODER.value
    ts = torch.tensor([[[15, 6, 2024]]])
    sample = MaskedOlmoEarthSample(sentinel2_l2a=img, sentinel2_l2a_mask=mask, timestamps=ts)

    out = model.forward(sample, patch_size=4)
    assert out["task_type"] == "segmentation"
    # 32 / patch_size=4 = 8 → (B, C, 8, 8).
    assert out["logits"].shape == (1, 4, 8, 8)
    assert out["probs"].shape == (1, 4, 8, 8)
    # Per-pixel softmax sums to 1 along the class dim.
    per_pixel_sum = out["probs"].sum(dim=1)  # (B, H, W)
    assert torch.allclose(per_pixel_sum, torch.ones_like(per_pixel_sum), atol=1e-4)
    # Prediction is a (B, H, W) class raster.
    assert out["prediction"].shape == (1, 8, 8)
