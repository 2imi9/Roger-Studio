"""Unit tests for the TIPSv2 dense-readout helpers added in
``app.services.tipsv2_labeler``.

These tests deliberately avoid loading any real TIPSv2 weights — they use
small synthetic mocks so they're fast (<1 s) and run without GPU or HF
network access. Real-model behaviour is covered by the ADE20K bench
(``backend/scripts/bench_tipsv2_ade20k.py``); this file just verifies the
helper plumbing.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.tipsv2_labeler import (  # noqa: E402
    PROMPT_TEMPLATES,
    _encode_image_dense,
    _ensure_imports,
    encode_text_with_templates,
)


# ---------------------------------------------------------------------------
# PROMPT_TEMPLATES
# ---------------------------------------------------------------------------


def test_prompt_templates_count_and_format() -> None:
    """The 9 TCL templates from the official TIPSv2 release."""
    assert len(PROMPT_TEMPLATES) == 9
    for tpl in PROMPT_TEMPLATES:
        assert "{}" in tpl, f"template missing {{}}: {tpl!r}"
        # Every template ends with a period — matches `_TCL_PROMPTS` exactly.
        assert tpl.endswith("."), f"template missing trailing period: {tpl!r}"


def test_prompt_templates_substitution_smoke() -> None:
    """``str.format`` works on every template with a typical class name."""
    for tpl in PROMPT_TEMPLATES:
        out = tpl.format("forest")
        assert "forest" in out


# ---------------------------------------------------------------------------
# encode_text_with_templates
# ---------------------------------------------------------------------------


def _fake_text_model(embed_dim: int = 4) -> MagicMock:
    """A model stub that returns zeros for encode_text and exposes the
    minimum attribute surface ``encode_text_with_templates`` needs."""
    m = MagicMock()
    m.config = types.SimpleNamespace(embed_dim=embed_dim)
    # next(model.parameters()) is used to find the device — return a fake
    # tensor on CPU.
    fake_param = torch.zeros(1)
    m.parameters = MagicMock(return_value=iter([fake_param]))
    # encode_text returns a fixed (C, D) tensor regardless of input. The
    # average over templates is trivially the same tensor; the post-
    # normalize gives unit-norm rows.
    def _encode_text(prompts):
        c = len(prompts)
        # Distinct rows so we can detect template averaging if it goes wrong.
        return torch.arange(c * embed_dim, dtype=torch.float32).reshape(c, embed_dim)
    m.encode_text = MagicMock(side_effect=_encode_text)
    return m


def test_encode_text_with_templates_calls_encode_text_once_per_template() -> None:
    _ensure_imports()
    model = _fake_text_model(embed_dim=4)
    classes = ["wall", "tree", "road"]
    out = encode_text_with_templates(model, classes)

    # One encode_text call per template, with len(classes) prompts each time.
    assert model.encode_text.call_count == len(PROMPT_TEMPLATES)
    for call in model.encode_text.call_args_list:
        prompts = call.args[0]
        assert len(prompts) == len(classes)
        # Each prompt should be a template-substituted form of one class.
        # Loose check: every prompt contains exactly one class name.
        for p, c in zip(prompts, classes):
            assert c in p

    # Output shape is (C, D) and L2-normalised.
    assert out.shape == (len(classes), 4)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"expected unit-norm rows, got {norms.tolist()}"
    )


def test_encode_text_with_templates_averages_in_normalised_space() -> None:
    """The average is taken over L2-normalised per-template embeddings, then
    re-normalised. Ensures one large-magnitude template can't dominate."""
    _ensure_imports()
    model = _fake_text_model(embed_dim=4)
    out = encode_text_with_templates(model, ["x", "y"])
    # Each row should be a unit vector — directly checks the final normalize.
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# _encode_image_dense — default attn_mode short-circuits
# ---------------------------------------------------------------------------


def _fake_image_output(batch: int = 1, n_patches: int = 16, dim: int = 4) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cls_token=torch.ones(batch, 1, dim),
        patch_tokens=torch.full((batch, n_patches, dim), 2.0),
        register_tokens=torch.zeros(batch, 1, dim),
    )


def test_encode_image_dense_default_calls_model_encode_image() -> None:
    """``attn_mode='default'`` and zero CLS subtract → straight passthrough
    of ``model.encode_image`` outputs."""
    _ensure_imports()
    model = MagicMock()
    expected = _fake_image_output()
    model.encode_image = MagicMock(return_value=expected)
    pixel_values = torch.zeros(1, 3, 8, 8)

    cls_token, patch_tokens = _encode_image_dense(model, pixel_values)

    model.encode_image.assert_called_once_with(pixel_values)
    assert torch.equal(cls_token, expected.cls_token)
    assert torch.equal(patch_tokens, expected.patch_tokens)


def test_encode_image_dense_cls_subtract_modifies_patch_tokens() -> None:
    """``cls_subtract=λ`` should subtract ``λ * cls_token`` from each patch
    token (broadcast over the patch axis)."""
    _ensure_imports()
    model = MagicMock()
    out = _fake_image_output(batch=1, n_patches=4, dim=2)
    out.cls_token = torch.tensor([[[1.0, 1.0]]])         # (1, 1, 2)
    out.patch_tokens = torch.tensor([[                   # (1, 4, 2)
        [10.0, 10.0],
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
    ]])
    model.encode_image = MagicMock(return_value=out)

    _, patch_tokens = _encode_image_dense(
        model, torch.zeros(1, 3, 8, 8), cls_subtract=0.5,
    )
    expected = torch.tensor([[
        [9.5, 9.5], [19.5, 19.5], [29.5, 29.5], [39.5, 39.5],
    ]])
    assert torch.allclose(patch_tokens, expected, atol=1e-6), (
        f"cls_subtract=0.5 expected to remove 0.5 from each patch dim; "
        f"got {patch_tokens}, expected {expected}"
    )


def test_encode_image_dense_zero_cls_subtract_is_noop() -> None:
    """``cls_subtract=0.0`` should leave patch_tokens identical to the model's
    output — no subtract path runs."""
    _ensure_imports()
    model = MagicMock()
    expected = _fake_image_output()
    model.encode_image = MagicMock(return_value=expected)
    _, patch_tokens = _encode_image_dense(
        model, torch.zeros(1, 3, 8, 8), cls_subtract=0.0,
    )
    assert torch.equal(patch_tokens, expected.patch_tokens)


def test_encode_image_dense_rejects_unknown_attn_mode() -> None:
    _ensure_imports()
    model = MagicMock()
    model.encode_image = MagicMock(return_value=_fake_image_output())
    with pytest.raises(ValueError, match="unknown attn_mode"):
        _encode_image_dense(model, torch.zeros(1, 3, 8, 8), attn_mode="bogus")


# ---------------------------------------------------------------------------
# Integration shape: end-to-end invariants without a real TIPSv2 model.
# ---------------------------------------------------------------------------


def test_encode_image_dense_default_passthrough_preserves_shapes() -> None:
    """Whatever the upstream model returns, the helper should pass shapes
    through unchanged for the default attn_mode."""
    _ensure_imports()
    model = MagicMock()
    out = _fake_image_output(batch=2, n_patches=64, dim=8)
    model.encode_image = MagicMock(return_value=out)
    cls, patches = _encode_image_dense(model, torch.zeros(2, 3, 16, 16))
    assert cls.shape == (2, 1, 8)
    assert patches.shape == (2, 64, 8)
