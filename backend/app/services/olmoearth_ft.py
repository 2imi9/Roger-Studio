"""Load fine-tuned OlmoEarth models from ``allenai/OlmoEarth-v1-FT-*`` repos.

The FT repos on Hugging Face ship a single ``model.ckpt`` (PyTorch Lightning
checkpoint from ``olmoearth_projects``) rather than the ``config.json`` +
``weights.pth`` pair base encoders use. The olmoearth-runner package can load
them, but it pulls ~130 transitive deps (rslearn, torchgeo, terratorch, …)
and would downgrade our torch pin — not a reasonable cost for a thin FastAPI
backend that only needs inference.

Instead, this module:
  1. Loads the Lightning state_dict via ``torch.load``.
  2. Extracts the encoder weights (prefix ``model.encoder.0.model.``) and
     loads them into a stock ``LatentMIM`` built from the corresponding base
     config (Nano / Tiny / Base / Large, picked by the FT repo's name).
  3. Inspects the decoder prefix (``model.decoders.<task>.*``) to figure out
     the head architecture from weight shapes:
       - ``Linear(D, C)`` ``(C, D)``       → scene-level classification
       - ``Conv2d(D, C, 1, 1)`` ``(C, D, 1, 1)`` → per-patch segmentation
       - ``Linear(D, 1)`` ``(1, D)``       → scene-level regression
     and rebuilds the head as a plain ``nn.Module`` with the checkpoint's
     trained weights.
  4. Wraps (encoder, head, task_type, class_count) into a ``FTModel`` wrapper
     that exposes a single ``forward(sample, patch_size)`` callable returning
     per-class / per-pixel logits + the task metadata needed downstream.

Known FT repos and task hints (as of April 2026):
  - ``allenai/OlmoEarth-v1-FT-Mangrove-Base``              — mangrove extent
  - ``allenai/OlmoEarth-v1-FT-LFMC-Base``                  — live fuel moisture
  - ``allenai/OlmoEarth-v1-FT-AWF-Base``                   — Kenya LULC (10 class)
  - ``allenai/OlmoEarth-v1-FT-ForestLossDriver-Base``      — forest-loss driver
  - ``allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base``  — ecosystem type

Class labels for these tasks live in ``olmoearth_projects``'s rslearn dataset
configs, which aren't published in the HF checkpoint. Until we vendor those,
we expose numeric class indices (``class_0``, ``class_1``, …) alongside the
head's raw logits — the service layer maps known tasks to human-readable tag
names via :data:`FT_TASK_METADATA`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import (
    CONFIG_FILENAME,
    ModelID,
    load_model_from_id,
)

logger = logging.getLogger(__name__)

# Map FT repo id → the matching base encoder so we can rebuild the backbone
# from the same config.json ``olmoearth_pretrain`` ships for the base model.
_FT_REPO_TO_BASE: dict[str, ModelID] = {
    "allenai/OlmoEarth-v1-FT-LFMC-Base": ModelID.OLMOEARTH_V1_BASE,
    "allenai/OlmoEarth-v1-FT-Mangrove-Base": ModelID.OLMOEARTH_V1_BASE,
    "allenai/OlmoEarth-v1-FT-AWF-Base": ModelID.OLMOEARTH_V1_BASE,
    "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base": ModelID.OLMOEARTH_V1_BASE,
    "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": ModelID.OLMOEARTH_V1_BASE,
}

# Task-specific metadata the service layer renders as legends. Class names +
# colors sourced directly from the olmoearth_projects rslearn configs at
# https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data
# (olmoearth_run.yaml → inference_results_config → classification_fields /
# regression_fields). Where a task exposes a ``task_type_override`` we bypass
# shape-based inference — some decoders look like classification in the
# state_dict (Linear(C, D)) but are actually per-patch segmentation (wrapped
# in ``SegmentationPoolingDecoder``).

import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ECOSYSTEM_JSON = _Path(__file__).parent / "_ft_ecosystem_classes.json"


def _ecosystem_class_names_and_colors() -> tuple[list[str], list[str]]:
    with _ECOSYSTEM_JSON.open() as f:
        entries = _json.load(f)
    # Expand to full 110 slots; any missing index gets a placeholder
    # (ecosystem head emits 110 logits but only ~60 classes are named).
    size = max(e["index"] for e in entries) + 1
    names = [f"class_{i}" for i in range(110)]
    colors = ["#6b7280"] * 110
    for e in entries:
        i = e["index"]
        if 0 <= i < 110:
            names[i] = e["name"]
            colors[i] = e["color"]
    del size
    return names, colors


_ECOSYSTEM_NAMES, _ECOSYSTEM_COLORS = _ecosystem_class_names_and_colors()


# ``input_spec`` keys (read by olmoearth_inference._run_real_inference and
# sentinel2_fetch.fetch_s2_temporal_stack) describe what the FT head was
# trained on. Sourced one-for-one from the ``model.yaml`` + ``dataset.json``
# pairs at https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data
# Each layer name like ``sentinel2.5`` is one PER_PERIOD_MOSAIC chunk:
#   * ``n_periods``        = len(layers)        — how many T slots to stack
#   * ``period_days``      = query_config.period_duration
#   * ``total_days``       = data_source.duration
#   * ``time_offset_days`` = data_source.time_offset (relative to label date;
#       at inference we anchor on the user's date_range end)
#   * ``s1_required``      = True if the head's training inputs include
#       sentinel1 (LFMC); the temporal-stack fetcher refuses to take this
#       path until the S1 fetcher exists, so we keep the head on the legacy
#       single-scene fallback and surface a known-broken warning.
#   * ``pre_post_split``   = True for ForestLossDriver — needs two non-
#       contiguous windows (pre at -300d, post at +7d) and rslearn's
#       SimpleTimeSeries(groups=[[0],[1]]) wrapper. Same fallback as above.
#   * ``predict_window_px`` / ``predict_overlap`` = the head's recommended
#       sliding-window inference parameters; not yet auto-applied (see
#       olmoearth_inference.start_inference's ``sliding_window`` kwarg) but
#       captured so we can flip the default per-head later.
FT_TASK_METADATA: dict[str, dict[str, Any]] = {
    "allenai/OlmoEarth-v1-FT-Mangrove-Base": {
        "task_family": "mangrove extent",
        # Actual task is per-pixel segmentation (SegmentationPoolingDecoder),
        # even though the decoder weight shape looks like scene-level
        # classification. Source: mangrove/model.yaml in olmoearth_projects.
        "task_type_override": "segmentation",
        "head_kind": "per_patch_linear",
        "patch_size": 2,
        "colormap": "mangrove",
        # Head emits 4 logits; published legend uses values 1/2/3 with 0 as
        # implicit nodata. See mangrove/olmoearth_run.yaml.
        "class_names": ["nodata", "mangrove", "water", "other"],
        "class_colors": ["#6b7280", "#94eb63", "#63d8eb", "#eba963"],
        "class_names_tentative": False,
        "input_spec": {
            "n_periods": 12,
            "period_days": 30,
            "total_days": 366,
            "time_offset_days": -180,
            "s1_required": False,
            "pre_post_split": False,
            "predict_window_px": 2,
            "predict_overlap": None,
        },
    },
    "allenai/OlmoEarth-v1-FT-LFMC-Base": {
        "task_family": "live fuel moisture",
        "task_type_override": "regression",
        # LFMC ships a 6-layer Conv3×3 stack (768→256→256→128→128→1) —
        # state-dict keys at ``model.decoder.0.layers.0.{0,3,5,8,10,12}``
        # — that the old scene_regression path (single Linear) couldn't
        # load; it raised ``no recognized FT head found`` and fell back
        # to the preview stub. ``conv_stack_regression`` dispatches to
        # ``_ConvStackRegressionHead``, which rebuilds the stack and
        # copies conv weights by shape-matched index. Output is per-
        # patch regression raster (same resolution + downstream
        # treatment as Mangrove / AWF class rasters).
        "head_kind": "conv_stack_regression",
        "patch_size": 4,
        "colormap": "flammability",
        "class_names": None,
        "class_names_tentative": False,
        "units": "% live fuel moisture",
        "value_range": [30.0, 200.0],      # matches lfmc/olmoearth_run.yaml
        "input_spec": {
            # LFMC is the only multi-modal head — needs S1 (VV/VH in dB)
            # alongside S2. Until the S1 fetch path lands the dispatcher
            # leaves this on the legacy single-scene S2-only fallback and
            # logs a known-broken warning.
            "n_periods": 12,
            "period_days": 14,
            "total_days": 168,
            "time_offset_days": -168,
            "s1_required": True,
            "pre_post_split": False,
            "predict_window_px": 32,
            "predict_overlap": 0.125,
        },
    },
    "allenai/OlmoEarth-v1-FT-AWF-Base": {
        "task_family": "southern-Kenya LULC",
        "task_type_override": "segmentation",
        "head_kind": "conv_segmentation",
        "patch_size": 4,
        "colormap": "landuse",
        # 10 classes (values 0-9), with index 9 published as implicit nodata.
        # See awf/olmoearth_run.yaml.
        "class_names": [
            "woodland_forest", "open_water", "shrubland_savanna",
            "herbaceous_wetland", "grassland_barren", "agriculture_settlement",
            "montane_forest", "lava_forest", "urban_dense_development", "nodata",
        ],
        "class_colors": [
            "#26734d", "#4682b4", "#8fbc8f", "#2e8b57", "#bdb76b",
            "#66cdaa", "#228b22", "#ff8c00", "#b22222", "#6b7280",
        ],
        "class_names_tentative": False,
        "input_spec": {
            # AWF dataset.json declares no ``duration`` / ``time_offset`` —
            # rslearn defaults to a 12-period × 30 d window centered on the
            # label, so we mirror that with a 360 d span and zero offset.
            "n_periods": 12,
            "period_days": 30,
            "total_days": 360,
            "time_offset_days": 0,
            "s1_required": False,
            "pre_post_split": False,
            "predict_window_px": 16,
            "predict_overlap": 0.25,
        },
    },
    "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base": {
        "task_family": "forest-loss driver",
        # Scene-level classification — pre/post Sentinel-2 pair → 10-way
        # softmax. Source: forest_loss_driver/model.yaml + olmoearth_run.yaml.
        # Decoder shape is rslearn's SegmentationPoolingDecoder with 1
        # conv bottleneck + 2 FC layers + output — NOT a flat Linear
        # classification head. Wired via ``conv_pool_fc_classification``
        # which matches the ``model.decoder.0.{conv_layers, fc_layers,
        # output_layer}`` state-dict keys.
        "task_type_override": "classification",
        "head_kind": "conv_pool_fc_classification",
        "patch_size": 4,
        "colormap": "forestloss",
        "class_names": [
            "agriculture", "mining", "airstrip", "road", "logging",
            "burned", "landslide", "hurricane", "river", "none",
        ],
        "class_colors": [
            "#89f336", "#ffde21", "#ffc0cb", "#ffa500", "#800080",
            "#ff8c00", "#ff0000", "#f5f5dc", "#00ffff", "#ffffff",
        ],
        "class_names_tentative": False,
        "input_spec": {
            # ForestLossDriver is structurally different — 4 ``pre`` scenes
            # at -300 d + 4 ``post`` scenes at +7 d, both with the
            # ``CONTAINS`` (atomic-scene, not mosaic) space_mode, fed
            # through rslearn's SimpleTimeSeries(groups=[[0],[1]]) wrapper
            # so the decoder receives concatenated pre/post features
            # (in_channels=1536 = 2 × 768). We capture the spec for
            # documentation but the dispatcher routes it back to the
            # legacy single-scene path until the pre/post fetch lands.
            "n_periods": 8,
            "period_days": 0,            # CONTAINS, not mosaic
            "total_days": 180,           # per group; 360 d total spread
            "time_offset_days": -300,    # pre group; post is +7 d
            "s1_required": False,
            "pre_post_split": True,
            "predict_window_px": 64,
            "predict_overlap": None,
        },
    },
    "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": {
        "task_family": "IUCN ecosystem type (level 3)",
        "task_type_override": "segmentation",
        "head_kind": "conv_segmentation",
        "patch_size": 4,
        "colormap": "ecosystem",
        # 110-class segmentation; ~60 named in the public run config, the
        # rest fall back to placeholders. See ecosystem_type_mapping/*.yaml.
        "class_names": _ECOSYSTEM_NAMES,
        "class_colors": _ECOSYSTEM_COLORS,
        "class_names_tentative": False,
        "input_spec": {
            # Trained ONLY on ``groups: ["north_africa"]`` — an AOI outside
            # north Africa is OOD by definition. The classes themselves
            # are global (IUCN level 3) so the labels still render, but
            # users running this on, e.g. Bay Area should be warned.
            "n_periods": 6,
            "period_days": 30,
            "total_days": 270,
            "time_offset_days": -90,
            "s1_required": False,
            "pre_post_split": False,
            "predict_window_px": 32,
            "predict_overlap": None,
        },
    },
}


TaskType = Literal["classification", "segmentation", "regression"]


@dataclass(frozen=True)
class FTHeadSpec:
    """Shape-inferred description of an FT decoder head."""

    task_type: TaskType
    num_classes: int
    decoder_key: str            # e.g. "mangrove_classification" / "segment" / "" (Ecosystem)
    weight_shape: tuple[int, ...]
    # Full state_dict prefix up to but not including the head-kind suffix.
    # Covers three observed patterns:
    #   * "model.decoders.mangrove_classification" + ".0.output_layer.{w,b}"  (Mangrove)
    #   * "model.decoders.segment"                 + ".1.layer.{w,b}"          (AWF)
    #   * "model.decoder"                          + ".1.layer.{w,b}"          (Ecosystem)
    # The last one doesn't have a task-name segment at all, which the old
    # ``model.decoders.{decoder_key}.*`` template can't express — hence
    # storing the resolved prefix here.
    head_prefix: str = ""

    def __post_init__(self) -> None:
        # Defense-in-depth. ``head_prefix=""`` has bitten us before:
        # when ``load_ft_model`` rebuilt ``spec`` after ``head_kind``
        # resolution, it briefly dropped ``head_prefix`` from the copy.
        # The downstream ``_copy_linear_weights`` + conv-seg branches
        # then looked up state-dict keys like ``.1.layer.weight`` (with
        # a leading dot, no prefix) and raised ``KeyError`` — which the
        # broad ``except Exception`` in ``start_inference`` caught and
        # turned into a "PREVIEW stub" render, silently masking the
        # bug. This assertion fails loudly at spec construction so any
        # future code path that forgets to thread the prefix breaks
        # visibly instead of producing synthetic output.
        #
        # Allow the bare empty string only at the class-level dataclass
        # default (i.e. nobody actually passed a value) — but once the
        # spec is constructed via ``_infer_head_spec`` or the rebuild
        # site, this MUST resolve to a real ``model.*`` prefix.
        if not self.head_prefix:
            raise ValueError(
                "FTHeadSpec.head_prefix is empty — the shape inferrer "
                "or the head-kind rebuild site forgot to thread it "
                "through. Without it, weight-copy keys look like "
                "`.0.output_layer.weight` (leading dot) and the load "
                "falls back to the stub renderer. See olmoearth_ft.py "
                "`__post_init__` for the full rationale."
            )
        if not self.head_prefix.startswith("model."):
            raise ValueError(
                f"FTHeadSpec.head_prefix must start with 'model.' "
                f"(checkpoint convention) — got {self.head_prefix!r}. "
                f"Check _infer_head_spec for a naming-convention drift."
            )


class _LinearClassificationHead(nn.Module):
    """Scene-level classifier: mean-pool tokens over spatial/temporal/band-set
    dims, then apply a single ``nn.Linear(D, num_classes)``."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        # tokens is (B, H', W', T, S, D). Mean over everything but B and D.
        pooled = tokens_bhwtsd.mean(dim=(1, 2, 3, 4))  # (B, D)
        return self.output_layer(pooled)               # (B, num_classes)


class _LinearRegressionHead(_LinearClassificationHead):
    """Same shape as classification but with ``num_classes=1`` and no softmax."""

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        return super().forward(tokens_bhwtsd).squeeze(-1)  # (B,)


class _ConvSegmentationHead(nn.Module):
    """Per-patch segmentation head: ``nn.Conv2d(D, num_classes, 1)`` applied
    to spatial tokens after collapsing the temporal + band-set dims."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        # (B, H, W, T, S, D) -> (B, D, H, W) via mean over T and S.
        pooled = tokens_bhwtsd.mean(dim=(3, 4))         # (B, H, W, D)
        x = pooled.permute(0, 3, 1, 2).contiguous()     # (B, D, H, W)
        return self.layer(x)                            # (B, C, H, W)


class _ConvPoolFCClassificationHead(nn.Module):
    """Scene-level classifier with a conv-bottleneck + FC-stack decoder.

    Matches ForestLossDriver's checkpoint shape — rslearn's
    ``SegmentationPoolingDecoder`` pattern with ``num_conv_layers=1``
    + ``num_fc_layers=2``:

      Conv3×3(D, conv_out) → AdaptiveAvgPool2d(1) → flatten →
      [Linear(in, fc_hidden) + ReLU] × num_fc_layers →
      Linear(fc_hidden, num_classes)

    The audit caught this as a loader gap: ForestLossDriver's
    state_dict has ``model.decoder.0.output_layer.weight`` with shape
    (C, fc_hidden) instead of (C, embed_dim). The old loader copied it
    straight into ``_LinearClassificationHead(embed_dim, C)`` which
    runtime-exploded because Linear(1536, 10) can't accept weights
    shaped (10, 512). This head builds the full pipeline so the
    weight copies land in the right places.
    """

    def __init__(
        self,
        embed_dim: int,
        conv_out: int,
        fc_hidden: int,
        num_classes: int,
        num_fc_layers: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, conv_out, kernel_size=3, padding=1)
        # Global spatial pool collapses H'×W' → 1×1 before the FC stack.
        # Matches rslearn's pooling behavior; without it the FC input
        # wouldn't match the (B, conv_out) shape the weights expect.
        self.pool = nn.AdaptiveAvgPool2d(1)
        fc: list[nn.Module] = []
        in_dim = conv_out
        for _ in range(num_fc_layers):
            fc.append(nn.Linear(in_dim, fc_hidden))
            fc.append(nn.ReLU(inplace=True))
            in_dim = fc_hidden
        self.fc = nn.Sequential(*fc)
        self.output_layer = nn.Linear(fc_hidden, num_classes)

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        # (B, H, W, T, S, D) → pool T/S → (B, H, W, D) → (B, D, H, W)
        pooled = tokens_bhwtsd.mean(dim=(3, 4))
        x = pooled.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)                  # (B, conv_out, H, W)
        x = self.pool(x).flatten(1)        # (B, conv_out)
        x = self.fc(x)                     # (B, fc_hidden)
        return self.output_layer(x)        # (B, num_classes)


class _ConvStackRegressionHead(nn.Module):
    """UNet-style Conv3×3 stack for per-patch regression (LFMC).

    The LFMC FT checkpoint ships a richer decoder than the other FT heads —
    a six-layer Conv3×3 stack that progressively reduces channels
    ``768 → 256 → 256 → 128 → 128 → 1``. State-dict keys live at
    ``{head_prefix}.layers.0.{0,3,5,8,10,12}.weight`` — the non-sequential
    indices reveal activations / (likely) upsamples sit between the conv
    blocks in the original module, but those are parameter-free and don't
    affect weight loading.

    We rebuild a Sequential of [Conv3×3 + ReLU] × 5 + Conv3×3 and copy the
    checkpoint's conv weights into matching positions by shape. The output
    is a single-channel regression raster at encoder-patch resolution,
    which the inference pipeline handles identically to the simpler
    ``_LinearRegressionHead`` path (both feed into
    ``olmoearth_model.run_ft_inference`` as regression).

    ``channels`` lists the ``(C_in, C_out)`` pairs of each conv layer, in
    order. Drives both module construction and the state-dict weight
    copy: each ``conv_keys[i]`` key in the checkpoint maps onto the i-th
    conv we build, in order of appearance in the sorted key list.
    """

    def __init__(self, channels: list[tuple[int, int]]) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(
                f"conv-stack regression needs ≥2 conv layers, got {len(channels)}"
            )
        last_out = channels[-1][1]
        if last_out != 1:
            raise ValueError(
                f"conv-stack regression head expects final Conv output channel = 1 "
                f"(regression scalar), got {last_out}"
            )
        layers: list[nn.Module] = []
        for i, (c_in, c_out) in enumerate(channels):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            # ReLU between every conv EXCEPT after the final regression
            # output — we want the raw scalar, not a non-negative-clamped
            # view. The original checkpoint structure appears to match:
            # after index 12 the Sequential ends.
            if i < len(channels) - 1:
                layers.append(nn.ReLU(inplace=True))
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        # Pool T and S dims so the input matches the checkpoint's expected
        # (B, D, H', W') conv input. Same reshape convention as
        # ``_ConvSegmentationHead``.
        pooled = tokens_bhwtsd.mean(dim=(3, 4))          # (B, H, W, D)
        x = pooled.permute(0, 3, 1, 2).contiguous()       # (B, D, H, W)
        return self.conv_stack(x)                         # (B, 1, H, W)


class _PerPatchLinearHead(nn.Module):
    """Apply ``nn.Linear(D, num_classes)`` per patch — mirrors rslearn's
    ``SegmentationPoolingDecoder`` with ``num_conv_layers=0, num_fc_layers=0``.

    The state_dict shape looks identical to a scene-level classification head
    (``(C, D)``), but the decoder actually keeps the spatial dims, so we
    produce ``(B, C, H, W)`` logits suitable for an argmax class raster. Used
    for Mangrove, whose published legend labels 4 per-pixel classes.
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens_bhwtsd: torch.Tensor) -> torch.Tensor:
        # (B, H, W, T, S, D) -> (B, H, W, D) by mean over T and S.
        pooled = tokens_bhwtsd.mean(dim=(3, 4))
        logits = self.output_layer(pooled)  # (B, H, W, C)
        return logits.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)


@dataclass
class FTModel:
    """The assembled fine-tuned model: base encoder + reconstructed head.

    ``repo_id`` is the HF repo it was loaded from. ``task_type``,
    ``num_classes``, and ``decoder_key`` come from :class:`FTHeadSpec`.
    ``metadata`` is the :data:`FT_TASK_METADATA` entry (may be ``{}`` for
    unknown repos).
    """

    encoder_parent: nn.Module           # the whole LatentMIM — we call .encoder
    head: nn.Module
    repo_id: str
    spec: FTHeadSpec
    metadata: dict[str, Any]

    def parameters(self) -> Any:
        """Chain encoder + head params for move / dtype calls."""
        import itertools  # noqa: PLC0415
        return itertools.chain(self.encoder_parent.parameters(), self.head.parameters())

    def eval(self) -> "FTModel":
        self.encoder_parent.eval()
        self.head.eval()
        return self

    def to(self, device: torch.device | str) -> "FTModel":
        self.encoder_parent.to(device)
        self.head.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.encoder_parent.parameters()).device

    @torch.no_grad()
    def forward(
        self, sample: MaskedOlmoEarthSample, patch_size: int = 4
    ) -> dict[str, torch.Tensor | str | int]:
        """Run encoder + head. Returns task-dependent output plus metadata.

        Returned dict:
          ``logits``     — raw head output (shape depends on task_type)
          ``probs``      — softmax/sigmoid view for classification/segmentation
          ``prediction`` — regression value or argmax class id
          ``task_type``  — classification / segmentation / regression
        """
        out = self.encoder_parent.encoder(sample, fast_pass=True, patch_size=patch_size)
        tokens = out["tokens_and_masks"].sentinel2_l2a  # (B, H', W', T, S, D)
        logits = self.head(tokens)
        if self.spec.task_type == "regression":
            return {
                "logits": logits,
                "prediction": logits,
                "task_type": self.spec.task_type,
            }
        if self.spec.task_type == "classification":
            probs = F.softmax(logits, dim=-1)
            return {
                "logits": logits,
                "probs": probs,
                "prediction": probs.argmax(dim=-1),
                "task_type": self.spec.task_type,
            }
        # segmentation
        probs = F.softmax(logits, dim=1)  # per-class over channel dim
        return {
            "logits": logits,
            "probs": probs,
            "prediction": probs.argmax(dim=1),  # (B, H, W) class indices
            "task_type": self.spec.task_type,
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _copy_linear_weights(
    layer: nn.Linear, state_dict: dict[str, torch.Tensor], head_prefix: str
) -> None:
    """Copy weights from ``{head_prefix}.0.output_layer.{weight,bias}``.

    ``head_prefix`` is e.g. ``model.decoders.mangrove_classification`` for
    the Mangrove FT head, or ``model.decoder`` for the singular-decoder
    Ecosystem checkpoint.
    """
    w_key = f"{head_prefix}.0.output_layer.weight"
    b_key = f"{head_prefix}.0.output_layer.bias"
    with torch.no_grad():
        layer.weight.copy_(state_dict[w_key])
        layer.bias.copy_(state_dict[b_key])


def is_ft_repo(repo_id: str) -> bool:
    """Best-effort "does this repo look like a fine-tuned OlmoEarth model?"
    check. Matches the ``-FT-`` naming convention the published artifacts use.
    """
    return "-FT-" in repo_id


def load_ft_model(repo_id: str, device: torch.device | None = None) -> FTModel:
    """Load an FT checkpoint from HF and assemble encoder + head.

    The encoder config is pulled from the base model's HF repo (downloaded
    lazily via ``olmoearth_pretrain.model_loader.load_model_from_id``) so we
    don't have to vendor the config here. Encoder weights are then overwritten
    with the FT-tuned state. The decoder state lives in ``model.decoders.*``
    and is reconstructed by shape.
    """
    base_model_id = _FT_REPO_TO_BASE.get(repo_id)
    if base_model_id is None:
        raise ValueError(
            f"Unknown FT repo {repo_id!r}. Add an entry to _FT_REPO_TO_BASE."
        )

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{repo_id}: model.ckpt has no state_dict")
    state_dict = ckpt["state_dict"]

    # 1. Build a fresh base encoder (downloads its config.json from HF if needed).
    encoder_parent = load_model_from_id(base_model_id, load_weights=False)
    encoder_parent.eval()

    # 2. Strip the Lightning ``model.encoder.0.model.`` prefix and load into
    #    the encoder submodule. Any keys that don't match are ignored —
    #    olmoearth_projects checkpoints carry the full encoder even for
    #    S2-only tasks, so strict=False guards against minor config drift.
    enc_prefix = "model.encoder.0.model."
    enc_state = {
        k[len(enc_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(enc_prefix)
    }
    # ``encoder_parent.encoder`` is the MultiModalEncoder; it's the object
    # whose forward takes MaskedOlmoEarthSample.
    missing, unexpected = encoder_parent.encoder.load_state_dict(enc_state, strict=False)
    logger.info(
        "FT encoder load for %s: %d keys loaded, %d missing, %d unexpected",
        repo_id, len(enc_state), len(missing), len(unexpected),
    )

    # 3. Identify the decoder prefix + infer task type from weight shape,
    #    then let repo-specific metadata override the task type when the
    #    shape is ambiguous (e.g. SegmentationPoolingDecoder looks like a
    #    Linear classification head but is actually per-patch segmentation).
    spec = _infer_head_spec(state_dict)
    embed_dim = _infer_embed_dim(spec)
    metadata = FT_TASK_METADATA.get(repo_id, {})
    override = metadata.get("task_type_override")
    head_kind = metadata.get("head_kind")
    effective_task: TaskType = override or spec.task_type  # type: ignore[assignment]

    # Resolve head_kind from (override + spec) if metadata didn't pin one.
    if head_kind is None:
        if effective_task == "regression":
            head_kind = "scene_regression"
        elif effective_task == "classification":
            head_kind = "scene_classification"
        else:
            # segmentation: 2D Linear → per-patch Linear; 4D Conv → Conv1x1
            head_kind = "per_patch_linear" if len(spec.weight_shape) == 2 else "conv_segmentation"

    spec = FTHeadSpec(
        task_type=effective_task,
        num_classes=spec.num_classes,
        decoder_key=spec.decoder_key,
        weight_shape=spec.weight_shape,
        # CRITICAL: thread head_prefix through the rebuild — otherwise the
        # dataclass default ("") kicks in and _copy_linear_weights / the
        # conv-seg path look up ``.1.layer.weight`` (leading dot, no
        # prefix) instead of ``model.decoder.1.layer.weight``, blowing up
        # with a KeyError that falls back to the stub renderer.
        head_prefix=spec.head_prefix,
    )

    if head_kind == "scene_classification":
        head: nn.Module = _LinearClassificationHead(embed_dim, spec.num_classes)
        _copy_linear_weights(head.output_layer, state_dict, spec.head_prefix)
    elif head_kind == "scene_regression":
        head = _LinearRegressionHead(embed_dim, 1)
        _copy_linear_weights(head.output_layer, state_dict, spec.head_prefix)
    elif head_kind == "per_patch_linear":
        head = _PerPatchLinearHead(embed_dim, spec.num_classes)
        _copy_linear_weights(head.output_layer, state_dict, spec.head_prefix)
    elif head_kind == "conv_segmentation":
        head = _ConvSegmentationHead(embed_dim, spec.num_classes)
        w_key = f"{spec.head_prefix}.1.layer.weight"
        b_key = f"{spec.head_prefix}.1.layer.bias"
        with torch.no_grad():
            head.layer.weight.copy_(state_dict[w_key])
            head.layer.bias.copy_(state_dict[b_key])
    elif head_kind == "conv_pool_fc_classification":
        # ForestLossDriver-style decoder. All sub-modules live under
        # ``{head_prefix}.0.`` — the ".0" is the ModuleList index of the
        # first decoder (same convention as the scene_classification and
        # conv_segmentation paths above).
        inner = f"{spec.head_prefix}.0"
        conv_w_key = f"{inner}.conv_layers.0.0.weight"
        conv_b_key = f"{inner}.conv_layers.0.0.bias"
        if conv_w_key not in state_dict:
            raise ValueError(
                f"conv_pool_fc_classification: expected {conv_w_key} in "
                f"state_dict — ForestLossDriver-style head missing its "
                "convolutional bottleneck."
            )
        conv_w = state_dict[conv_w_key]
        conv_out = int(conv_w.shape[0])
        # True encoder embedding dim (≠ spec.weight_shape[1] which is the
        # output_layer's input dim = fc_hidden).
        real_embed_dim = int(conv_w.shape[1])
        # Discover FC layers by scanning for fc_layers.<N>.0.weight keys.
        prefix_fc = f"{inner}.fc_layers."
        fc_w_keys: list[str] = []
        for k, v in state_dict.items():
            if (
                k.startswith(prefix_fc)
                and k.endswith(".0.weight")
                and hasattr(v, "ndim")
                and v.ndim == 2
            ):
                fc_w_keys.append(k)
        # Sort by numeric index in "fc_layers.<N>.0.weight"
        fc_w_keys.sort(key=lambda k: int(k[len(prefix_fc):].split(".")[0]))
        if not fc_w_keys:
            raise ValueError(
                f"conv_pool_fc_classification: no fc_layers under "
                f"{prefix_fc!r}. Head architecture doesn't match."
            )
        fc_hidden = int(state_dict[fc_w_keys[0]].shape[0])
        num_fc_layers = len(fc_w_keys)
        head = _ConvPoolFCClassificationHead(
            embed_dim=real_embed_dim,
            conv_out=conv_out,
            fc_hidden=fc_hidden,
            num_classes=spec.num_classes,
            num_fc_layers=num_fc_layers,
        )
        with torch.no_grad():
            head.conv.weight.copy_(state_dict[conv_w_key])
            if conv_b_key in state_dict:
                head.conv.bias.copy_(state_dict[conv_b_key])
            fc_linears = [m for m in head.fc.children() if isinstance(m, nn.Linear)]
            assert len(fc_linears) == len(fc_w_keys), (
                f"built {len(fc_linears)} FC layers but checkpoint has {len(fc_w_keys)}"
            )
            for w_key, linear in zip(fc_w_keys, fc_linears):
                b_key = w_key[:-len(".weight")] + ".bias"
                linear.weight.copy_(state_dict[w_key])
                if b_key in state_dict:
                    linear.bias.copy_(state_dict[b_key])
            out_w_key = f"{inner}.output_layer.weight"
            out_b_key = f"{inner}.output_layer.bias"
            head.output_layer.weight.copy_(state_dict[out_w_key])
            if out_b_key in state_dict:
                head.output_layer.bias.copy_(state_dict[out_b_key])
        logger.info(
            "loaded conv_pool_fc classification head for %s: "
            "encoder_D=%d → conv_out=%d → %d × fc_hidden=%d → %d classes",
            repo_id, real_embed_dim, conv_out, num_fc_layers, fc_hidden, spec.num_classes,
        )
    elif head_kind == "conv_stack_regression":
        # Scan the state_dict for all ``{head_prefix}.layers.0.<N>.weight``
        # keys (4D conv weights) and build the head from the sorted
        # sequence. The checkpoint's non-sequential indices (0, 3, 5, 8,
        # 10, 12 for LFMC) map onto contiguous positions in our Sequential
        # because our reconstruction drops the parameter-free gap layers.
        conv_keys: list[tuple[int, str]] = []
        prefix_layers = f"{spec.head_prefix}.layers.0."
        for k, v in state_dict.items():
            if (
                k.startswith(prefix_layers)
                and k.endswith(".weight")
                and hasattr(v, "ndim")
                and v.ndim == 4
            ):
                idx_str = k[len(prefix_layers):-len(".weight")]
                try:
                    conv_keys.append((int(idx_str), k))
                except ValueError:
                    # Sub-modules deeper than one-level don't match the
                    # LFMC pattern — skip defensively.
                    continue
        conv_keys.sort()
        if not conv_keys:
            raise ValueError(
                f"conv_stack_regression: no conv weights found under "
                f"{prefix_layers!r}"
            )
        # Derive (C_in, C_out) pairs and sanity-check they chain. The first
        # conv's C_in should match the encoder embed_dim; each successive
        # conv's C_in must equal the prior conv's C_out. Final C_out = 1.
        channels: list[tuple[int, int]] = []
        for _, key in conv_keys:
            w = state_dict[key]
            c_out, c_in = int(w.shape[0]), int(w.shape[1])
            channels.append((c_in, c_out))
        # No embed_dim sanity check here — ``_infer_embed_dim`` is keyed
        # off ``spec.weight_shape`` which for conv-stack points at the
        # FINAL conv (C_out=1), not the first. The encoder's real output
        # width (768 for Base) matches the first conv's ``C_in`` which we
        # derive directly from the state_dict below, so the comparison
        # would only produce a noisy false-positive warning.
        for i in range(1, len(channels)):
            prev_out = channels[i - 1][1]
            cur_in = channels[i][0]
            if prev_out != cur_in:
                raise ValueError(
                    f"conv_stack_regression: channel chain broken between "
                    f"layers {i - 1} (C_out={prev_out}) and {i} (C_in={cur_in}). "
                    f"Checkpoint may include upsample/downsample in a way "
                    f"our reconstruction doesn't model."
                )
        head = _ConvStackRegressionHead(channels)
        # Copy weights: conv_keys[i] maps onto the i-th Conv2d in the
        # Sequential (ReLU layers between them have no state_dict keys, so
        # positional indexing into .conv_stack by multiplying i*2 matches).
        with torch.no_grad():
            conv_positions = [
                m for m in head.conv_stack.children() if isinstance(m, nn.Conv2d)
            ]
            assert len(conv_positions) == len(conv_keys), (
                f"head built {len(conv_positions)} conv layers but checkpoint "
                f"has {len(conv_keys)}"
            )
            for (_, w_key), conv in zip(conv_keys, conv_positions):
                b_key = w_key[:-len(".weight")] + ".bias"
                conv.weight.copy_(state_dict[w_key])
                if b_key in state_dict:
                    conv.bias.copy_(state_dict[b_key])
        logger.info(
            "loaded conv-stack regression head for %s: %d conv layers, "
            "channels=%s",
            repo_id, len(channels), channels,
        )
    else:
        raise ValueError(f"unknown head_kind {head_kind!r} for repo {repo_id}")

    head.eval()
    model = FTModel(
        encoder_parent=encoder_parent,
        head=head,
        repo_id=repo_id,
        spec=spec,
        metadata=metadata,
    )
    if device is not None:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# Head shape inference
# ---------------------------------------------------------------------------


_CONV_STACK_RE = re.compile(r"^(model\.decoder(?:s\.[^.]+)?\.\d+)\.layers\.0\.(\d+)\.weight$")


def _infer_head_spec(state_dict: dict[str, torch.Tensor]) -> FTHeadSpec:
    """Inspect decoder keys to decide task type + class count.

    Rules — covering the FT repos published as of April 2026:
      - ``model.decoders.<key>.0.output_layer.weight`` 2D ``(C, D)`` — Linear
        head (Mangrove). Treat as regression when C == 1, else classification.
      - ``model.decoders.<key>.1.layer.weight`` 4D ``(C, D, 1, 1)`` — Conv1x1
        head, per-patch segmentation (AWF).
      - ``model.decoder.0.output_layer.weight`` — same as Linear above but
        with no task-key segment (no repo ships this today, kept for symmetry).
      - ``model.decoder.1.layer.weight`` — same as Conv above but no task-key
        segment. Ecosystem ships this pattern: the checkpoint uses singular
        ``model.decoder.*`` instead of the plural ``model.decoders.<key>.*``.
      - ``model.decoder.0.layers.0.<N>.weight`` with 4D Conv3×3 shape
        — 6-layer UNet-style conv stack (LFMC). Non-sequential indices
        (N ∈ {0, 3, 5, 8, 10, 12}) indicate parameter-free activations /
        upsamples interleaved in the original Sequential. The final
        layer's ``C_out`` tells task_type: 1 → regression, >1 →
        segmentation.

    Returned ``FTHeadSpec.head_prefix`` encodes the resolved prefix so the
    downstream weight-copy code doesn't need to branch on naming convention.
    """
    # (w_key, head_prefix) candidates by pattern
    classification_w_key: str | None = None
    classification_prefix: str = ""
    segmentation_w_key: str | None = None
    segmentation_prefix: str = ""
    # Conv-stack detection — aggregate all matching keys per prefix so we
    # can reason about the full stack (number of convs, final channel
    # count) rather than one-layer-at-a-time.
    conv_stack_prefix_to_layers: dict[str, dict[int, tuple[int, ...]]] = {}
    for k in state_dict.keys():
        # Plural, with task-key: model.decoders.<key>.0.output_layer.weight
        if k.startswith("model.decoders.") and k.endswith(".0.output_layer.weight"):
            classification_w_key = k
            # prefix = everything up to ".0.output_layer.weight"
            classification_prefix = k[: -len(".0.output_layer.weight")]
        elif k.startswith("model.decoders.") and k.endswith(".1.layer.weight"):
            segmentation_w_key = k
            segmentation_prefix = k[: -len(".1.layer.weight")]
        # Singular, no task-key: model.decoder.0.output_layer.weight / .1.layer.weight
        elif k == "model.decoder.0.output_layer.weight":
            classification_w_key = k
            classification_prefix = "model.decoder"
        elif k == "model.decoder.1.layer.weight":
            segmentation_w_key = k
            segmentation_prefix = "model.decoder"
        else:
            # Conv-stack pattern — e.g. model.decoder.0.layers.0.12.weight.
            # Match only 4D tensors so we don't get confused by biases or
            # unrelated sub-modules. Track (idx → shape) per prefix.
            m = _CONV_STACK_RE.match(k)
            if m is not None:
                w = state_dict[k]
                if hasattr(w, "ndim") and w.ndim == 4:
                    prefix = m.group(1)
                    idx = int(m.group(2))
                    conv_stack_prefix_to_layers.setdefault(prefix, {})[idx] = tuple(
                        w.shape  # (C_out, C_in, kH, kW)
                    )

    if classification_w_key is not None:
        w = state_dict[classification_w_key]
        if w.ndim != 2:
            raise ValueError(
                f"expected 2D Linear weight for {classification_w_key}, got {tuple(w.shape)}"
            )
        num_classes, _ = w.shape
        # decoder_key is the task-name segment for plural prefixes; empty
        # string for the singular-decoder pattern (Ecosystem).
        parts = classification_prefix.split(".")
        decoder_key = parts[2] if len(parts) >= 3 and parts[1] == "decoders" else ""
        task_type: TaskType = "regression" if num_classes == 1 else "classification"
        return FTHeadSpec(
            task_type=task_type,
            num_classes=num_classes,
            decoder_key=decoder_key,
            weight_shape=tuple(w.shape),
            head_prefix=classification_prefix,
        )

    if segmentation_w_key is not None:
        w = state_dict[segmentation_w_key]
        if w.ndim != 4:
            raise ValueError(
                f"expected 4D Conv2d weight for {segmentation_w_key}, got {tuple(w.shape)}"
            )
        num_classes, _, _, _ = w.shape
        parts = segmentation_prefix.split(".")
        decoder_key = parts[2] if len(parts) >= 3 and parts[1] == "decoders" else ""
        return FTHeadSpec(
            task_type="segmentation",
            num_classes=num_classes,
            decoder_key=decoder_key,
            weight_shape=tuple(w.shape),
            head_prefix=segmentation_prefix,
        )

    # Conv-stack pattern — LFMC's multi-layer Conv3×3 decoder.
    if conv_stack_prefix_to_layers:
        # Pick the prefix with the most conv layers (defensive — there
        # should only ever be one in a well-formed checkpoint).
        best_prefix = max(
            conv_stack_prefix_to_layers.keys(),
            key=lambda p: len(conv_stack_prefix_to_layers[p]),
        )
        layer_map = conv_stack_prefix_to_layers[best_prefix]
        sorted_indices = sorted(layer_map.keys())
        final_shape = layer_map[sorted_indices[-1]]  # (C_out, C_in, kH, kW)
        num_channels_out = final_shape[0]
        # Regression if the final conv produces a single channel; any
        # other positive count means per-pixel classification /
        # segmentation. No LFMC-style repo ships the latter today, but
        # keep the branch open so a future Conv-stack segmenter still
        # loads.
        task_type = "regression" if num_channels_out == 1 else "segmentation"
        # decoder_key parsing mirrors the other branches: for plural
        # ``model.decoders.<key>.N`` prefixes, extract <key>; for singular
        # ``model.decoder.N``, leave empty.
        parts = best_prefix.split(".")
        decoder_key = parts[2] if len(parts) >= 3 and parts[1] == "decoders" else ""
        return FTHeadSpec(
            task_type=task_type,  # type: ignore[arg-type]
            num_classes=num_channels_out,
            decoder_key=decoder_key,
            weight_shape=final_shape,
            head_prefix=best_prefix,
        )

    raise ValueError(
        "no recognized FT head found in state_dict — expected "
        "model.decoders.<key>.0.output_layer.weight (Linear) or "
        "model.decoders.<key>.1.layer.weight (Conv) or "
        "model.decoder.0.output_layer.weight / model.decoder.1.layer.weight "
        "or model.decoder.<n>.layers.0.<m>.weight (Conv-stack)"
    )


def _infer_embed_dim(spec: FTHeadSpec) -> int:
    """The encoder embedding dim is the second axis of the head weight for
    both Linear (C, D) and Conv2d (C, D, 1, 1) patterns."""
    return int(spec.weight_shape[1])


def class_names_for(
    repo_id: str, num_classes: int
) -> tuple[list[str], bool]:
    """Return (class_names, is_tentative). Falls back to ``class_<idx>``."""
    md = FT_TASK_METADATA.get(repo_id, {})
    names = md.get("class_names")
    if names and len(names) == num_classes:
        return list(names), bool(md.get("class_names_tentative"))
    return [f"class_{i}" for i in range(num_classes)], True


def class_colors_for(repo_id: str, num_classes: int) -> list[str] | None:
    """Return the published per-class hex colors, or ``None`` if the repo
    doesn't ship a color table (consumer falls back to colormap gradient).
    Sourced from olmoearth_projects' ``olmoearth_run.yaml`` legend blocks."""
    md = FT_TASK_METADATA.get(repo_id, {})
    colors = md.get("class_colors")
    if colors and len(colors) == num_classes:
        return list(colors)
    return None
