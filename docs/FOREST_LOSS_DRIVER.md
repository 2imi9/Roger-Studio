# ForestLossDriver pre/post pipeline

`allenai/OlmoEarth-v1-FT-ForestLossDriver-Base` is a 10-class
scene-level classifier that identifies the **driver** of a forest-loss
event (agriculture, mining, logging, fire, …) from a pair of Sentinel-2
scenes — one before the event, one after.

Unlike Roger Studio's other FT heads (Mangrove, AWF, Ecosystem, LFMC),
ForestLossDriver requires **two** S2 scene groups concatenated along the
encoder feature dim. Feeding it a single scene produces off-distribution
output (a runtime channel-shape error from the conv-pool-fc decoder).
Roger handles the pre/post fetch + concat automatically when you supply
an `event_date`.

## Class list (10)

The output is a softmax over 10 driver classes, in the order the
checkpoint emits them:

| Index | Class | Color | Notes |
|---|---|---|---|
| 0 | `agriculture` | `#89f336` | Crop / pasture conversion |
| 1 | `mining` | `#ffde21` | Open-pit / strip mining |
| 2 | `airstrip` | `#ffc0cb` | Cleared linear strip |
| 3 | `road` | `#ffa500` | Road construction |
| 4 | `logging` | `#800080` | Selective or clear-cut harvest |
| 5 | `burned` | `#ff8c00` | Fire scar |
| 6 | `landslide` | `#ff0000` | Mass-movement ground loss |
| 7 | `hurricane` | `#f5f5dc` | Wind / tropical-storm damage |
| 8 | `river` | `#00ffff` | River-meander forest loss |
| 9 | `none` | `#ffffff` | No detected event |

Per `forest_loss_driver/olmoearth_run.yaml`. Class names are NOT
tentative — they're persisted in the published rslearn run config and
match the head's training labels.

Geographic coverage is **pantropical**. The head was trained on Hansen
forest-loss alerts in Latin America, central Africa, and SE Asia; for
events outside that belt (e.g. boreal logging) the predictions will
still emit but with reduced accuracy.

## Pipeline shape

```
event_date (e.g. 2022-08-15)
       │
       ├── pre window   = [event - 300 d - 4×30 d,  event - 300 d]    → 4 S2 scenes
       └── post window  = [event - 4×30 d,          event + 7 d]      → 4 S2 scenes

For each AOI chunk:
       fetch_s2_chunk_stack(pre_scenes)   →  (1, H, W, T_pre,  12)  S2 reflectance
       fetch_s2_chunk_stack(post_scenes)  →  (1, H, W, T_post, 12)  S2 reflectance
                            ↓                                          ↓
       encoder(pre)  → tokens_pre  (1, H', W', T_pre,  S, 768)
       encoder(post) → tokens_post (1, H', W', T_post, S, 768)
                            ↓                                          ↓
                    pool over (T, S)                          pool over (T, S)
                            ↓                                          ↓
                  pre_pooled (1, H', W', 768)         post_pooled (1, H', W', 768)
                                       ↓                ↓
                              concat along D
                                       ↓
                       combined (1, H', W', 1, 1, 1536)
                                       ↓
                          conv_pool_fc_classification head
                                       ↓
                              logits (1, 10) → softmax → class index per chunk
```

The 1536-channel input is the contract baked into the head's first
`Conv2d(1536, conv_out, 3)` layer; without the pre/post concat the
forward pass crashes with `expected input[1, 768, …] to have 1536
channels`.

## How to run it

### From the React UI

1. **Map → Import Data → Run inference**
2. Select **Forest-loss driver — Driver classification (pre/post S2 pair)**
3. The **Event date** picker auto-appears below the model selector.
   Default is one year ago; pick something in the post-event window
   you actually care about.
4. Either draw an AOI or click **↳ Use demo AOI (Pará, Brazilian
   Amazon)** — the demo presets `event_date=2022-08-15` for a verified
   deforestation event.
5. **Run + add to map**

Wall time on a 5 km × 5 km AOI is ≈ 2× a normal FT run because the
chunked path now fetches both pre + post stacks per chunk (in parallel,
but each chunk's network bill doubles). Typical: ~1-2 min cold,
~30-60 s warm.

### From the API directly

```bash
curl -X POST http://localhost:8000/api/olmoearth/infer \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west": -55.05, "south": -9.05, "east": -55.02, "north": -9.02},
    "model_repo_id": "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base",
    "event_date": "2022-08-15"
  }'
```

Response shape matches every other FT classification:

```json
{
  "kind": "pytorch",
  "task_type": "classification",
  "num_classes": 10,
  "class_names": ["agriculture","mining","airstrip","road","logging",
                  "burned","landslide","hurricane","river","none"],
  "class_probs": [0.32, 0.012, 0.065, 0.021, 0.068, 0.173, …],
  "present_class_ids": [0, 4],
  "scene_id": "S2B_MSIL2A_20220804T135709_R067_T21LYK_20240717T081055",
  …
}
```

`present_class_ids` is the set of distinct argmax values across all
chunks in the AOI — different chunks can land on different driver
classes when the AOI spans both an event and unaffected forest.

### Without `event_date`

Omitting `event_date` falls through to the legacy single-scene path
which **deliberately fails** with a clear runtime error
(`expected input[1, 768, …] to have 1536 channels`). The job becomes
a stub — no real prediction is rendered, and the response carries
`stub_reason` so the UI can prompt the user to set the date.

```bash
# This produces kind=stub
curl -X POST http://localhost:8000/api/olmoearth/infer \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west": -55.05, …},
    "model_repo_id": "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base"
  }'
```

## GeoJSON polygon export

Same workflow as Mangrove / AWF / Ecosystem — the
`/olmoearth/ft-classification/geojson` endpoint accepts the
ForestLossDriver model id + an `event_date` and returns vectorised
polygons coloured per class:

```bash
curl -X POST http://localhost:8000/api/olmoearth/ft-classification/geojson \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west": -55.05, "south": -9.05, "east": -55.02, "north": -9.02},
    "model_repo_id": "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base",
    "event_date": "2022-08-15",
    "min_pixels": 4,
    "simplify_tolerance_m": 5.0
  }' \
  -o pa_amazon_drivers.geojson
```

`event_date` is hashed into the upstream inference job_id, so the
GeoJSON export hits the same cached run as the map-tile call (no
duplicate forward pass).

## Knobs

| Param | Default | Notes |
|---|---|---|
| `event_date` | none (required) | ISO-8601 date; the post-event anchor. |
| `n_pre`, `n_post` | 4 each | Number of scenes per group; controlled by the head's `n_periods=8` metadata. Override only if you know what you're doing. |
| `pre_offset_days` | 300 | Days before `event_date` for the **end** of the pre group. |
| `post_offset_days` | 7 | Days after `event_date` for the **end** of the post group. |
| `period_days` | 30 | Per-scene search window. |

The defaults match the head's training-time spec (rslearn config:
4 atomic scenes per group, ~300d pre offset, ~7d post offset). Most
users should never need to override them.

## Audit / known limits

* **Output granularity is the chunk size** (5 km by default). The
  conv-pool-fc head pools spatially before classifying, so each chunk
  produces ONE class label for ~25 km². Sliding 64-px windows would
  give 64-px × 64-px granularity (the head's
  `predict_window_px` setting); not yet implemented.
* **Wall time is ~2× a single-stack FT run** because each chunk
  fetches and encodes pre + post separately. The two fetches run in
  parallel inside one chunk, so the slowdown is closer to 1.5× than
  full 2× in practice.
* **Off-distribution AOIs** outside the pantropical training belt
  still produce a 10-class softmax, but the model has not been
  validated there — treat the output as exploratory.

## Verifying end-to-end

The shipped pipeline was verified on:

* **AOI**: Pará, Brazilian Amazon (`-55.05, -9.05 → -55.02, -9.02`)
* **Event date**: `2022-08-15` (peak Amazon deforestation season)
* **Result**: `kind=pytorch`, `num_classes=10`,
  `present_class_ids=[0, 4]` (agriculture + logging), argmax softmax
  ~32% on agriculture / ~17% on burned.

Without `event_date`, the same AOI yields `kind=stub` with the
1536-vs-768 channel mismatch — confirming the dispatcher correctly
gates the pre/post path on the date.
