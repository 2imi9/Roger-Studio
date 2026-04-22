"""Unit tests for :mod:`app.services.olmoearth_model`.

These exercise the synchronous wrapper around ``olmoearth_pretrain`` that
loads a pretrained encoder and runs a forward pass. The network-dependent
tests are marked ``network`` and hit Hugging Face for the Nano checkpoint
(~4 MB) — run with ``pytest -m network`` to include them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``app.`` importable when pytest is invoked from backend/.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services import olmoearth_model as M  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure tests don't leak cached models between cases."""
    yield
    M.clear_cache()


def test_preferred_device_returns_torch_device() -> None:
    import torch

    d = M.preferred_device()
    assert isinstance(d, torch.device)
    assert d.type in {"cpu", "cuda"}


def test_model_summary_misses_before_load() -> None:
    summary = M.model_summary("allenai/OlmoEarth-v1-Nano")
    assert summary == {"cached": False, "repo_id": "allenai/OlmoEarth-v1-Nano"}


def test_pca_to_scalar_on_trivial_embedding() -> None:
    # Constant embedding -> zero PCA -> zero scalar (special-cased).
    emb = np.ones((4, 4, 8), dtype=np.float32)
    scalar = M._pca_to_scalar(emb)
    assert scalar.shape == (4, 4)
    assert float(scalar.min()) == 0.0 and float(scalar.max()) == 0.0


def test_pca_to_scalar_rescales_to_unit_interval() -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((6, 8, 12)).astype(np.float32)
    scalar = M._pca_to_scalar(emb)
    assert scalar.shape == (6, 8)
    assert 0.0 <= float(scalar.min())
    assert float(scalar.max()) <= 1.0
    # Not collapsed to a constant for random input.
    assert float(scalar.max() - scalar.min()) > 0.1


def test_run_s2_inference_rejects_wrong_shape() -> None:
    import torch.nn as nn

    dummy = nn.Identity()
    bad = np.zeros((32, 32, 12), dtype=np.float32)  # missing B, T
    with pytest.raises(ValueError, match="BHWTC"):
        M.run_s2_inference(dummy, bad, timestamp_dmy=(15, 6, 2024))


def test_run_s2_inference_rejects_wrong_band_count() -> None:
    import torch.nn as nn

    dummy = nn.Identity()
    bad = np.zeros((1, 32, 32, 1, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="12 bands"):
        M.run_s2_inference(dummy, bad, timestamp_dmy=(15, 6, 2024))


def test_run_s2_inference_rejects_non_divisible_hw() -> None:
    import torch.nn as nn

    dummy = nn.Identity()
    bad = np.zeros((1, 30, 32, 1, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        M.run_s2_inference(dummy, bad, timestamp_dmy=(15, 6, 2024), patch_size=4)


# ---------------------------------------------------------------------------
# Network-gated: hits Hugging Face for Nano weights (~4 MB) + runs a forward.
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_nano_and_forward_pass_returns_expected_shapes() -> None:
    model, device = M.load_encoder("allenai/OlmoEarth-v1-Nano")
    summary = M.model_summary("allenai/OlmoEarth-v1-Nano")
    assert summary["cached"] is True
    assert summary["class_name"] == "LatentMIM"
    # Nano's encoder has ~1.4M params + 800K decoder = ~2-4M total.
    assert summary["num_parameters"] > 1_000_000

    rng = np.random.default_rng(42)
    image = (rng.random((1, 32, 32, 1, 12)) * 5000).astype(np.float32)
    result = M.run_s2_inference(model, image, timestamp_dmy=(15, 6, 2024), patch_size=4)
    # 32 / 4 = 8 patches per spatial side; Nano embedding dim = 128.
    assert result.embedding.shape == (8, 8, 128)
    assert result.embedding_dim == 128
    assert result.patch_size == 4
    assert result.scalar.shape == (8, 8)
    assert 0.0 <= float(result.scalar.min()) <= float(result.scalar.max()) <= 1.0


@pytest.mark.network
def test_load_encoder_caches_on_second_call() -> None:
    m1, _ = M.load_encoder("allenai/OlmoEarth-v1-Nano")
    m2, _ = M.load_encoder("allenai/OlmoEarth-v1-Nano")
    assert m1 is m2
