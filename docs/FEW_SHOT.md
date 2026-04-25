# Few-shot semantic segmentation

Roger Studio's few-shot tool (Embedding tools tab in the OlmoEarth
Import panel) lets you classify an entire AOI from just a handful of
labelled clicks. There's no model fine-tuning — the workflow is pure
embedding-space nearest-prototype matching, so it works on any base
encoder for any region without weights to download or training to run.

## When to use it

* You drew an AOI but no FT head fits — it's not Mangrove or Forest
  Loss or Ecosystem, but you can still describe what you want by
  pointing.
* You want a quick first-pass map to decide where to invest in a
  proper labelling pipeline (e.g. SamGeo / TIPSv2 polygons).
* You want to compare 3-4 candidate "regions of interest" (water vs
  vegetation vs urban etc.) without spinning up a fine-tune.

## How it works

```
1. Run the chunked base encoder forward over the AOI
   (same code path as PCA false-color and Similarity tools)
   →  global_embedding (H_patch, W_patch, D)  — D = 128/192/768/1024 per encoder

2. For each user-defined class:
     points (lon, lat) → patch (row, col) in the AOI grid
     prototype = mean(global_embedding[r, c]  for each labelled pixel)

3. For every pixel in the AOI:
     similarities = cos_sim(global_embedding, prototypes)   →  (H_patch, W_patch, K)
     class_idx    = argmax(similarities, axis=-1)           →  (H_patch, W_patch)
     confidence   = max(similarities,    axis=-1)           →  (H_patch, W_patch)

4. Pixels where the encoder returned an all-zero embedding (untouched
   chunks at AOI edges, scenes that failed to fetch) get class_idx = -1
   so the tile renderer paints them transparent rather than forcing
   them into the nearest class.
```

The pipeline reuses every safety guard of the chunked inference path —
RAM precheck, AOI-size cap, circuit breaker, disconnect-cancel — so
few-shot inherits the same operational behaviour as PCA / Similarity.

## How to run it

### From the React UI

1. Pick a base encoder (Nano / Tiny / Base / Large) on the
   **Embedding tools** tab.
2. Draw an AOI (or use the per-model demo button on the **Run
   inference** tab and switch back).
3. Open the **Few-shot classify** section.
4. Per class:
   * Click **+ point** to arm pixel-pick mode for that class
   * Click anywhere on the map → that pixel is labelled
   * Repeat to add more examples (more = more robust prototype)
   * Optional: click the class name to rename it; click `×` to clear
     that class's points
5. Click **Run few-shot classification** when at least 2 classes have
   ≥1 point each.
6. The result drops on the map as a coloured class raster, and shows
   up in the Added Layer panel — same surface as every other FT
   inference output.

### From the API directly

```bash
curl -X POST http://localhost:8000/api/olmoearth/embedding-tools/few-shot \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"west": -117.30, "south": 33.75, "east": -117.27, "north": 33.78},
    "model_repo_id": "allenai/OlmoEarth-v1-Tiny",
    "classes": [
      {
        "name": "vegetation",
        "color": "#22c55e",
        "points": [
          {"lon": -117.295, "lat": 33.762},
          {"lon": -117.291, "lat": 33.768}
        ]
      },
      {
        "name": "urban",
        "color": "#ef4444",
        "points": [
          {"lon": -117.282, "lat": 33.755}
        ]
      }
    ]
  }'
```

Response shape matches FT classification — `tile_url`,
`class_names`, `class_probs=null`, `present_class_ids`, etc — so the
existing `/ft-classification/geojson` polygon export and the map tile
renderer work unchanged.

## How many points per class?

* **1 point** is enough to get an output. The prototype is just that
  pixel's embedding vector — useful as a smoke test, but noisy.
* **5-10 points** spread around the class's typical appearance is
  the sweet spot. Each point's embedding adds robustness; the mean
  averages out single-pixel spikes.
* **More than ~30 points** has diminishing returns. The prototype is
  a mean — extra points just narrow the variance further. If the
  class is genuinely heterogeneous (e.g. "all water" includes deep
  ocean + turbid river + reservoir), consider splitting into
  sub-classes.

## Knobs

| Knob | Default | Where | Notes |
|---|---|---|---|
| `model_repo_id` | `allenai/OlmoEarth-v1-Tiny` | request | Any base encoder. Larger encoder = sharper class boundaries but slower forward pass. |
| `n_periods` | 3 | request | S2 temporal context window. Same default as PCA / Similarity. |
| `target_gsd_m` | 10 | request | S2 native resolution. |
| `patch_size` | 4 | request | Encoder patch token size. Output raster pixel = `target_gsd_m × patch_size` = 40 m. |

## Limitations

* **Class output granularity is the encoder patch, not the input
  pixel.** With default `patch_size=4` + `target_gsd_m=10`, each output
  cell is 40 m × 40 m. Sub-pixel features (a single car, a 5 m strip
  of trees) get averaged into the surrounding patch.
* **No calibrated probability.** Cosine similarity to a prototype is
  a *relative* score. The "confidence" raster (saved as
  `scalar_raster` for legacy compatibility) is `(cos + 1) / 2` mapped
  to `[0, 1]` — useful for ranking pixels but **not** a Bayesian
  probability.
* **Domain match matters.** The encoder was pretrained on Sentinel-2
  L2A; classes that hinge on spectral bands the encoder doesn't see
  (e.g. fine-grain mineral types from hyperspectral) won't separate
  in embedding space no matter how many points you click.
* **Imbalanced classes don't get reweighted.** If "background" gets
  100 clicks and "rare class" gets 2 clicks, the rare-class prototype
  stays accurate (means don't care about count), but the cosine
  similarity threshold per class is identical — you may need to tune
  by labelling tighter examples for the rare class.

## Comparison: few-shot vs FT vs Similarity

| Tool | When | Output | Click cost |
|---|---|---|---|
| **FT head** (Mangrove, AWF, …) | Class set matches your task | Calibrated per-class softmax | 0 (just press Run) |
| **Similarity** | "Find more like this one pixel" | 0-1 heatmap, single query | 1 click |
| **Few-shot** | Multi-class, ad-hoc taxonomy | K-class raster, K = #classes | ~5-30 clicks |

Few-shot is the natural escalation when Similarity isn't enough (you
have multiple things you care about) but FT is over-spec (you don't
have a checkpoint for your task and won't fine-tune).
