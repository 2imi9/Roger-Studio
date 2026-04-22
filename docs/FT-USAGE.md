# Using Fine-Tuned OlmoEarth Models in Roger Studio

> Four paths: **UI click-through** (easiest), **LLM chat** (conversational), **HTTP API** (scripting), **Python** (direct).
> Five fine-tuned tasks ship out of the box — Mangrove, LFMC, AWF, ForestLossDriver, EcosystemTypeMapping.

---

## TL;DR

```text
1. Draw (or load) a bounding box over your area of interest.
2. Pick an FT model that matches the area + question.
3. Click Load once (first use — 500 MB download), then Run.
4. Read the colored overlay + legend panel to interpret the prediction.
```

If you want sub-tile spatial detail or a scene-level task → turn on **sliding window** and optionally increase `max_size_px`.

---

## The five fine-tuned models

| Repo id | Task | Input | Output | Pick it when… |
|---|---|---|---|---|
| `allenai/OlmoEarth-v1-FT-Mangrove-Base` | 4-class segmentation (`nodata`, `mangrove`, `water`, `other`) | Single S2 L2A scene | Per-pixel class raster | Coastal, tropical / subtropical |
| `allenai/OlmoEarth-v1-FT-AWF-Base` | 10-class segmentation (southern-Kenya LULC) | Single S2 L2A scene | Per-pixel class raster | East-African rangeland / savanna |
| `allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base` | 110-class IUCN segmentation | Single S2 L2A scene | Per-pixel class raster | Global ecosystem typology (60 named classes, tail = placeholders) |
| `allenai/OlmoEarth-v1-FT-ForestLossDriver-Base` | 10-way classification (`agriculture`, `mining`, `logging`, …) | **Pre/post** S2 pair *(simplified — see Limitations)* | Scene-level class + softmax | You have a recently detected forest-loss alert polygon |
| `allenai/OlmoEarth-v1-FT-LFMC-Base` | Regression (live fuel moisture, %) | Single S2 L2A scene | Scalar value (30–200%) | Wildfire-risk screening over vegetated areas |

Class names + hex colors in Roger come from `allenai/olmoearth_projects/olmoearth_run_data/<task>/olmoearth_run.yaml` — the same legend the official OlmoEarth platform uses.

---

## Path A — UI click-through (recommended)

### 1. Draw or pick an area

- Click **Draw Rectangle** / **Draw Polygon** on the map and trace your area of interest, **or**
- Hit **Try Demo (Kenyan Coast)** for a quick test, **or**
- Click one of the **Sample Data** cards (Knoxville NDVI gives you real raster to compare against an inference layer).

The Sidebar header now shows bbox coords + perimeter + area + elevation. Keep bboxes **under ~5 km per side** for the first try on CPU — Nano and FT heads can handle up to ~25 km with `sliding_window=true` but latency scales with area.

### 2. Switch to the OlmoEarth tab → Finetune head subtab

You'll see:
- **Connected** chip + Refresh button + cache summary (e.g. `4 cached · 2.1 GB on disk`).
- Coverage panel showing which OlmoEarth project regions overlap your bbox.
- A **Recommended model** link — the keyword matcher picks one based on task hints in your polygon properties.

Below that, each published FT repo appears as a row with HF downloads / likes + an action pill.

### 3. Load once, then Run

- **Load** pulls the `model.ckpt` (~500 MB for a Base FT) into the HF cache under `~/.cache/huggingface/hub/`. First run only; subsequent runs reuse it.
- Pill flips through **Load → Loading… → Cached · size + ×**.
- Once cached and a bbox is selected, a **Run** pill appears. Click it.

### 4. Watch the layer appear

The backend fetches a cloud-free Sentinel-2 composite via Planetary Computer, runs the head, caches the prediction raster, and returns an XYZ tile URL. MapLibre adds the layer automatically (~5–30 s on CPU for a small bbox with Nano-sized encoder, longer for Base).

### 5. Read the legend panel

Appears below the OlmoEarth panel in the sidebar. For each active inference layer:
- Model name + scene id + date + cloud cover.
- Task badge (`segmentation`, `classification`, `regression`, or `embedding`).
- Per-class rows with the **published hex swatch + index + name** (for class tasks).
- Gradient bar + predicted value + units (for regression).
- Embedding dim (for encoder-only).
- **Remove layer** button.

### 6. Compare against ground truth

Once there's ≥ 1 layer on the map, a **⇌ Compare** button appears top-right. Click it to flip into a vertical-split view (A | B pickers on both sides). Drag the divider to swipe. Great for:
- Prediction vs. uploaded ground-truth GeoTIFF (e.g. Knoxville NDVI).
- Two FT models on the same bbox (e.g. WorldCover vs. AWF LULC).
- Same FT model at two dates (re-run with different `date_range`).

---

## Path B — LLM chat (conversational)

Switch to the **LLM** tab (Local Gemma, NIM, or Claude — they all share the same 8 tools) and just ask:

> **You:** "I drew a polygon over the Ganges delta. Is there mangrove here, and how much?"
>
> **LLM → `query_olmoearth` (catalog look-up)**
> **LLM → `run_olmoearth_inference`** `{bbox, model_repo_id: "allenai/OlmoEarth-v1-FT-Mangrove-Base", date_range: "2024-11-01/2025-02-28"}`
> **Backend:** fetches S2 + runs the 4-class head + returns `legend_classes` with published colors.
> **LLM:** *"I ran the Mangrove FT model. Over your polygon, the argmax class at most patches is `mangrove` (teal #94eb63) with some `water` (cyan #63d8eb) along the channels and `other` (tan #eba963) on upland fringes. I added the layer to your map — you can see the discrete class map now. Scene: 2024-12-14 L2A, 3% cloud."*

You don't have to name the tool — the LLM picks it when you describe the task. Helpful phrasings:

- *"run the mangrove model over this bbox"*
- *"classify land cover here using OlmoEarth"*
- *"what's the forest-loss driver for this polygon?"*
- *"map live fuel moisture in this area"*
- *"use sliding window so I get spatial variation"*

Pass `sliding_window: true` + optional `window_size: 32` + `max_size_px: 256` in natural language — the LLM forwards the flags.

---

## Path C — HTTP API (scripting)

### One-shot — scene-level

```bash
curl -s -X POST http://localhost:8000/api/olmoearth/infer \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west":-122.35,"south":47.60,"east":-122.32,"north":47.63},
    "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
    "date_range": "2024-06-01/2024-09-30",
    "max_size_px": 256
  }' | jq
```

**Response:**

```json
{
  "job_id": "9d753978038d3994",
  "tile_url": "/api/olmoearth/infer-tile/9d753978038d3994/{z}/{x}/{y}.png",
  "kind": "pytorch",
  "status": "ready",
  "task_type": "segmentation",
  "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
  "num_classes": 4,
  "class_names": ["nodata", "mangrove", "water", "other"],
  "class_names_tentative": false,
  "decoder_key": "mangrove_classification",
  "patch_size": 2,
  "scene_id": "S2B_MSIL2A_20240829T185919_R013_T10TET_20240829T230848",
  "scene_datetime": "2024-08-29T18:59:19.024000Z",
  "scene_cloud_cover": 0.0009,
  "legend": {
    "kind": "segmentation",
    "classes": [
      {"index": 0, "name": "nodata",   "color": "#6b7280"},
      {"index": 1, "name": "mangrove", "color": "#94eb63"},
      {"index": 2, "name": "water",    "color": "#63d8eb"},
      {"index": 3, "name": "other",    "color": "#eba963"}
    ],
    "colors_source": "published"
  }
}
```

Feed the `tile_url` straight into any XYZ-capable client:

```js
map.addSource("mangrove", { type: "raster", tiles: [`${origin}/${tile_url}`], tileSize: 256 });
map.addLayer({ id: "mangrove", type: "raster", source: "mangrove" });
```

### Sliding-window mode for spatial detail

```bash
curl -s -X POST http://localhost:8000/api/olmoearth/infer \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west":-122.40,"south":47.55,"east":-122.25,"north":47.70},
    "model_repo_id": "allenai/OlmoEarth-v1-FT-LFMC-Base",
    "date_range": "2024-06-01/2024-09-30",
    "max_size_px": 256,
    "sliding_window": true,
    "window_size": 32
  }'
```

Turns scene-level LFMC regression into a per-tile moisture map. Same for scene-level classification tasks (ForestLossDriver) — they get promoted to `task_type: "segmentation"` with a stitched class raster.

### Checking / unloading the model cache

```bash
curl -s http://localhost:8000/api/olmoearth/cache-status | jq
curl -s -X POST http://localhost:8000/api/olmoearth/load \
  -d '{"repo_id":"allenai/OlmoEarth-v1-FT-Mangrove-Base","repo_type":"model"}' \
  -H "Content-Type: application/json"
curl -s -X POST http://localhost:8000/api/olmoearth/unload \
  -d '{"repo_id":"allenai/OlmoEarth-v1-FT-Mangrove-Base"}' \
  -H "Content-Type: application/json"
```

---

## Path D — Python (skip the backend)

If you just want the raw OlmoEarth API without Roger Studio:

```python
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
import torch

model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).eval()

# Prepare a Sentinel-2 L2A tensor in the OlmoEarth band order:
#   B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
sample = MaskedOlmoEarthSample(
    sentinel2_l2a       = image_BHWTC,                                # (1, H, W, 1, 12)
    sentinel2_l2a_mask  = torch.ones(1, H, W, 1, 3) * MaskValue.ONLINE_ENCODER.value,
    timestamps          = torch.tensor([[[22, 7, 2025]]]),             # [day, month-0-idx, year]
)

with torch.no_grad():
    features = model.encoder(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a
# features shape: (B, H', W', T, S, D)
```

For fine-tuned models specifically, Roger ships a self-contained loader that sidesteps the 3 GB `olmoearth-runner` dep:

```python
from app.services.olmoearth_ft import load_ft_model
from app.services.olmoearth_model import run_ft_inference
from app.services.sentinel2_fetch import fetch_s2_composite, image_to_bhwtc, timestamp_from_iso
from app.models.schemas import BBox
import asyncio

bbox = BBox(west=39.6, south=-4.2, east=40.0, north=-3.9)
scene = asyncio.run(fetch_s2_composite(bbox, "2024-06-01/2024-09-30", max_size_px=256))

model = load_ft_model("allenai/OlmoEarth-v1-FT-AWF-Base").eval()
result = run_ft_inference(
    model,
    image_to_bhwtc(scene.image[: (scene.image.shape[0] // 4) * 4, : (scene.image.shape[1] // 4) * 4, :]),
    timestamp_dmy = timestamp_from_iso(scene.datetime_str),
    patch_size    = 4,
)
print(result.task_type)       # "segmentation"
print(result.class_raster)    # (H', W') int32 class indices
print(result.scalar)          # (H', W') float32 max-prob confidence in [0, 1]
print(result.class_names)     # ["woodland_forest", "open_water", …, "nodata"]
print(result.class_colors)    # ["#26734d", "#4682b4", …, "#6b7280"]
```

`FTInferenceResult` also carries `embedding` (not for FT — those are in `run_s2_inference`), `patch_size`, `decoder_key`, `repo_id`, and `prediction_value` (for regression).

---

## Reading the outputs

### Segmentation (Mangrove, AWF, Ecosystem)

- Tile renderer paints each 10 m patch in the legend hex for its argmax class. The map is a discrete thematic map — no gradient.
- `class_raster`: `(H', W')` int32 array of class indices. `H' = max_size_px / patch_size` (e.g. 64 for a 256-px fetch with patch_size=4).
- `scalar`: per-patch max-class probability in `[0, 1]`. Dim where the model is unsure, bright where it's confident. Kept in the response for future confidence overlays.
- `class_names_tentative`: `false` for all 5 shipped tasks — names are verified against the published `olmoearth_run.yaml`.

### Classification (ForestLossDriver)

- Scene-level output: `class_probs` is a `(C,)` softmax vector summing to 1.0.
- Without sliding window: the overlay is a **uniform** color over the bbox (argmax class) — fine for small alert polygons, useless for large bboxes.
- With `sliding_window=true`: the task gets promoted to `"segmentation"` and you get a spatial class map.

### Regression (LFMC)

- Scene-level scalar in task units (e.g. `prediction_value: 87.3` = 87.3 % live fuel moisture).
- Tile renderer paints the bbox with the red-yellow-green gradient at the normalized value (`(value - 30) / (200 - 30)`).
- With `sliding_window=true`: per-tile regression values → a spatial moisture map with dry (red) to wet (green) patches.

### Encoder-only (Nano / Tiny / Base / Large)

- Not fine-tuned — just the raw encoder.
- `scalar`: first principal component of the per-patch embedding, rescaled to `[0, 1]` and colormapped through the default `embedding` gradient. Useful as a "show me what the encoder sees here" sanity check.
- `embedding_dim`: 128 for Nano/Tiny, 384 for Base, 768 for Large.

---

## Picking the right settings

| Parameter | Default | When to change |
|---|---|---|
| `max_size_px` | 256 | ↓ for speed (64 on CPU), ↑ for larger areas (cap 512). Longer side of the fetched S2 tile in 10 m pixels. |
| `date_range` | `2024-04-01/2024-10-01` | Pick a season that matches the task: summer for vegetation/LFMC, dry-season for mangroves, a pre/post window for change detection. |
| `sliding_window` | `false` | `true` for spatial maps on scene-level tasks, for bboxes bigger than the training tile (~2.56 km), or when the single-pass fetch drops too much resolution. |
| `window_size` | `32` | Match the training tile. Must divide `patch_size` (2 for Mangrove, 4 for everything else). |

---

## Troubleshooting

**`kind: "stub"` with a `stub_reason` field.** The real pipeline failed (e.g. no cloud-free S2 for the date range, HF download timeout, forward-pass error). The tile layer still renders as a watermarked gradient so your UI doesn't silently lose the layer. Check the `stub_reason` text.

**`no Sentinel-2 scenes found`**. Widen `date_range`, raise `max_cloud_cover` (not currently an API param — edit your fetch call), or double-check the bbox isn't over ocean.

**Slow first call on a new FT repo.** The 500 MB `model.ckpt` downloads on first load. Use `/api/olmoearth/load` proactively to cache it in the background. After that, runs are only bottlenecked by S2 fetch + forward pass.

**Uniform color across a huge bbox.** You're hitting a scene-level task (ForestLossDriver, LFMC) — turn on `sliding_window: true` and re-run. The bbox also gets automatically cropped to `max_size_px` at 10 m/pixel, so a 100 km-wide bbox ends up at coarse resolution unless you tile it.

**`class_names_tentative: true`** in the response. You loaded a repo not in the `FT_TASK_METADATA` table — output still works but names default to `class_0`..`class_N`. File an issue or add the repo.

**Windows `ConnectError` on first STAC call.** Known cold-connect quirk; the backend retries 4× with exponential backoff. If it persists, check corporate firewall / VPN.

---

## Known limitations for FT models

Spelled out in [CAPABILITIES.md §11](./CAPABILITIES.md) but worth restating:

- **Torch is CPU-only** in the shipped env. Nano/FT-Base on a 256-px bbox runs in 5–30 s. For GPU speed, install a CUDA wheel (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
- **ForestLossDriver** was trained on pre/post S2 pairs fed through `SimpleTimeSeries`. Roger's loader reconstructs the final Linear and encoder, so predictions run but the temporal pairing is simplified — accuracy will be below the reference paper.
- **LFMC** in the public checkpoint is a per-pixel UNet regression head. We load the encoder + the final projection only, so output is scene-level by default. Use `sliding_window=true` for a spatial moisture map that approximates the full UNet's behaviour.
- **EcosystemTypeMapping**: 110 logits in the head, only 60 named publicly — the tail uses `class_<idx>` placeholders. Still useful but interpret unnamed classes carefully.

---

## Where the ground truth lives

Every class name + hex color Roger displays is sourced from:

```
https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data/
    ├── mangrove/           → olmoearth_run.yaml  (4 classes)
    ├── awf/                → olmoearth_run.yaml  (10 classes)
    ├── ecosystem_type_mapping/ → olmoearth_run.yaml  (60 named / 110 slots)
    ├── forest_loss_driver/ → olmoearth_run.yaml  (10 classes)
    └── lfmc/               → olmoearth_run.yaml  (regression, 30–200 % value_range)
```

No ad-hoc names, no guessing. If Ai2 updates these files, the easiest path is to edit `backend/app/services/olmoearth_ft.py::FT_TASK_METADATA` + `_ft_ecosystem_classes.json` and rerun the offline tests.
