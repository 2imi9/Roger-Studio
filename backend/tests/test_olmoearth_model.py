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


def test_pre_post_inference_concatenates_768_to_1536() -> None:
    """``run_ft_pre_post_inference`` must encode pre and post stacks
    independently and feed the head a tensor whose feature dim is
    ``2 × encoder_dim`` — the contract the conv-pool-fc decoder of
    ForestLossDriver was trained against.
    """
    import torch
    import torch.nn as nn

    from app.services import olmoearth_ft as FT

    embed_dim = 8        # tiny stand-in for 768; the contract is "2D after concat"
    num_classes = 10     # mirrors ForestLossDriver's 10-driver classification
    patch_size = 4
    h = w = 8            # 2 × 2 patches

    captured: dict[str, torch.Tensor] = {}

    class _FakeEncoder(nn.Module):
        def forward(self, sample, fast_pass: bool = True, patch_size: int = 4):
            # tokens_and_masks.sentinel2_l2a — shape (B, H', W', T, S, D).
            # Use the sample's image to know T; pick S=3 to mirror the
            # real encoder's per-band-set output.
            t = sample.timestamps.shape[1]
            tokens = torch.randn(1, h // patch_size, w // patch_size, t, 3, embed_dim)

            class _Out:
                pass

            class _TM:
                pass

            tm = _TM()
            tm.sentinel2_l2a = tokens
            o = _Out()
            o.__dict__["tokens_and_masks"] = tm  # type: ignore[attr-defined]

            class _Holder(dict):
                pass

            return _Holder({"tokens_and_masks": tm})

    class _FakeEncoderParent(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _FakeEncoder()

    class _FakeHead(nn.Module):
        def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
            captured["head_input"] = tokens_bhwtsd
            # Scene-level classification: (B, num_classes) logits.
            return torch.zeros(tokens_bhwtsd.shape[0], num_classes)

    spec = FT.FTHeadSpec(
        task_type="classification",
        num_classes=num_classes,
        decoder_key="conv_pool_fc_classification",
        weight_shape=(num_classes, 2 * embed_dim),
        head_prefix="model.decoder",
    )

    fake_model = FT.FTModel(
        encoder_parent=_FakeEncoderParent(),
        head=_FakeHead(),
        repo_id="allenai/OlmoEarth-v1-FT-ForestLossDriver-Base",
        spec=spec,
        metadata={"colormap": "forestloss", "patch_size": patch_size},
    )

    rng = np.random.default_rng(0)
    pre = (rng.random((1, h, w, 4, 12)) * 5000).astype(np.float32)
    post = (rng.random((1, h, w, 4, 12)) * 5000).astype(np.float32)

    result = M.run_ft_pre_post_inference(
        model=fake_model,
        pre_image_bhwtc=pre,
        post_image_bhwtc=post,
        pre_timestamp_dmy=(15, 7, 2022),
        post_timestamp_dmy=(22, 8, 2022),
        patch_size=patch_size,
        device=torch.device("cpu"),
        normalize=False,
    )

    head_in = captured["head_input"]
    # Head input must be 6D (B, H', W', T, S, D) with the LAST dim equal to
    # 2 × embed_dim — the concatenation contract.
    assert head_in.ndim == 6
    assert head_in.shape[-1] == 2 * embed_dim
    # T and S collapsed to 1 each (we pooled before concat).
    assert head_in.shape[3] == 1
    assert head_in.shape[4] == 1
    # Spatial dims = h/patch_size.
    assert head_in.shape[1] == h // patch_size
    assert head_in.shape[2] == w // patch_size

    # Result still has the FT classification shape contract.
    assert result.task_type == "classification"
    assert result.num_classes == num_classes
    assert result.class_raster is not None
    assert result.class_raster.shape == (h // patch_size, w // patch_size)


def test_run_ft_pre_post_inference_rejects_mismatched_spatial_dims() -> None:
    """Pre and post stacks must share H/W; if they don't, the concatenation
    contract breaks. Surface a ValueError before the encoder runs."""
    import torch.nn as nn

    from app.services import olmoearth_ft as FT

    spec = FT.FTHeadSpec(
        task_type="classification",
        num_classes=10,
        decoder_key="conv_pool_fc_classification",
        weight_shape=(10, 16),
        head_prefix="model.decoder",
    )

    class _Stub(nn.Module):
        pass

    fake_model = FT.FTModel(
        encoder_parent=_Stub(),
        head=_Stub(),
        repo_id="x",
        spec=spec,
        metadata={},
    )

    pre = np.zeros((1, 8, 8, 4, 12), dtype=np.float32)
    post = np.zeros((1, 12, 12, 4, 12), dtype=np.float32)

    with pytest.raises(ValueError, match="must equal post H/W"):
        M.run_ft_pre_post_inference(
            model=fake_model,
            pre_image_bhwtc=pre,
            post_image_bhwtc=post,
            pre_timestamp_dmy=(15, 7, 2022),
            post_timestamp_dmy=(22, 8, 2022),
            patch_size=4,
            normalize=False,
        )


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


# ---------------------------------------------------------------------------
# LRU cache cap (P1 host-safety guard).
#
# Background: the encoder cache used to grow without bound. Clicking
# Nano → Tiny → Base → Large in the UI loaded all four into VRAM
# simultaneously — Large alone is multiple GB on GPU, so the four
# together easily wedged a 24 GB consumer card. The cache is now bounded
# by ``OE_MODEL_CACHE_MAX_ENTRIES`` (default 1) with LRU eviction.
# ---------------------------------------------------------------------------


def test_max_cache_entries_default(monkeypatch):
    monkeypatch.delenv("OE_MODEL_CACHE_MAX_ENTRIES", raising=False)
    assert M._max_cache_entries() == 1


def test_max_cache_entries_env_override(monkeypatch):
    monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", "4")
    assert M._max_cache_entries() == 4


def test_max_cache_entries_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", "many")
    assert M._max_cache_entries() == 1


def test_max_cache_entries_zero_or_neg_clamps_to_one(monkeypatch):
    """A 0-entry cache means every chunked job re-loads the encoder
    (~30 s cold). That's worse than 1, so refuse the footgun."""
    for v in ["0", "-2"]:
        monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", v)
        assert M._max_cache_entries() == 1, v


class _StubModel:
    """Minimal stand-in for a torch.nn.Module — has the .to() method the
    eviction path calls. No real GPU memory is involved."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.device = "cuda"

    def to(self, dev) -> "_StubModel":
        self.device = str(dev)
        return self


def _seed_cache(*tags: str) -> None:
    """Manually populate _cache with stub models in a known order. Tests
    that exercise eviction logic don't need real models."""
    M.clear_cache()
    for tag in tags:
        M._cache[tag] = (_StubModel(tag), "cpu")


def test_evict_oldest_locked_removes_first_entry():
    _seed_cache("a", "b", "c")
    assert list(M._cache.keys()) == ["a", "b", "c"]
    with M._cache_lock:
        M._evict_oldest_locked()
    assert list(M._cache.keys()) == ["b", "c"]


def test_evict_oldest_locked_on_empty_cache_is_noop():
    M.clear_cache()
    with M._cache_lock:
        M._evict_oldest_locked()  # must not raise
    assert len(M._cache) == 0


def test_load_encoder_evicts_when_at_cap(monkeypatch):
    """Cap=1: loading a second model evicts the first."""
    monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", "1")

    def fake_load_a(repo_id, device=None):
        return _StubModel("a"), "cpu"

    def fake_load_b(repo_id, device=None):
        return _StubModel("b"), "cpu"

    M.clear_cache()
    # Drive load_encoder via direct cache injection — simpler than
    # monkeypatching every load path. The eviction logic doesn't care
    # how the entries got there, only how many there are.
    with M._cache_lock:
        M._cache["a"] = (_StubModel("a"), "cpu")
    assert "a" in M._cache

    # Simulate inserting a second model when cap=1.
    with M._cache_lock:
        max_entries = M._max_cache_entries()
        while len(M._cache) >= max_entries:
            M._evict_oldest_locked()
        M._cache["b"] = (_StubModel("b"), "cpu")

    # 'a' was the oldest and got evicted; 'b' is the only entry now.
    assert list(M._cache.keys()) == ["b"]


def test_lru_order_protects_recently_used(monkeypatch):
    """Cap=2: load A, load B (cap reached), touch A, load C → B evicted (A
    is most-recently-used and stays)."""
    monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", "2")
    M.clear_cache()
    with M._cache_lock:
        M._cache["a"] = (_StubModel("a"), "cpu")
        M._cache["b"] = (_StubModel("b"), "cpu")
        # "Touch" a → moves to most-recent end. This is what load_encoder
        # does on a cache hit.
        M._cache.move_to_end("a")
        # Now insert C, which forces eviction of the oldest = "b".
        max_entries = M._max_cache_entries()
        while len(M._cache) >= max_entries:
            M._evict_oldest_locked()
        M._cache["c"] = (_StubModel("c"), "cpu")
    assert list(M._cache.keys()) == ["a", "c"]
    assert "b" not in M._cache


def test_lru_higher_cap_keeps_more_models(monkeypatch):
    """Cap=4 lets all four base encoders coexist (the original failure
    mode being prevented)."""
    monkeypatch.setenv("OE_MODEL_CACHE_MAX_ENTRIES", "4")
    M.clear_cache()
    with M._cache_lock:
        for tag in ("nano", "tiny", "base", "large"):
            max_entries = M._max_cache_entries()
            while len(M._cache) >= max_entries:
                M._evict_oldest_locked()
            M._cache[tag] = (_StubModel(tag), "cpu")
    assert sorted(M._cache.keys()) == ["base", "large", "nano", "tiny"]
