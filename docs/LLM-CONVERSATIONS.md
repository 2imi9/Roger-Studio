# Example LLM Conversations — Advanced OlmoEarth FT Use Cases

> Concrete multi-turn chat examples that exercise the full agent loop: catalog lookup → fine-tuned inference → supporting layers (NDVI, WorldCover, polygon stats) → synthesis.
> Works with any of the three chat backends (**Local Gemma 4**, **NVIDIA NIM**, **Claude**). Tool JSON is what each client emits via OpenAI-compatible `tool_calls`.

Each transcript shows three things per assistant turn:
1. **`tool_calls`** — the exact JSON the model emits (what the backend dispatches).
2. **Tool result summary** — trimmed output your backend returns (full raster / tile URL stays server-side on the job).
3. **Final assistant text** — what the user sees in the chat bubble.

---

## Conversation 1 — Mangrove mapping on the Kenyan coast

**Setup:**
- User draws a bbox over the Mombasa / Kilifi coast (`west=39.6, south=-4.2, east=40.0, north=-3.9` — the built-in *Try Demo* polygon).
- `scene_context` sent to the LLM: `{ bbox, area: "selected", datasets: [] }`.

### Turn 1 — user

> *"What OlmoEarth FT models would be useful for this bbox, and which should I run first if I'm studying mangrove extent?"*

### Turn 1 — assistant

**tool_calls:**
```json
[
  {
    "id": "call_01",
    "function": {
      "name": "query_olmoearth",
      "arguments": "{\"bbox\":{\"west\":39.6,\"south\":-4.2,\"east\":40.0,\"north\":-3.9}}"
    }
  }
]
```

**Tool result (trimmed):**
```json
{
  "highlight": {
    "project_coverage": [
      { "repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base", "task": "mangrove extent mapping" }
    ],
    "recommended_model": {
      "repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
      "reason": "fine-tuned head for alias 'mangrove'"
    }
  },
  "datasets": [ "...", "allenai/olmoearth_projects_mangrove" ],
  "models": [ "...5 FT repos..." ]
}
```

**Final text:**

> Your bbox intersects an OlmoEarth **mangrove** project region. Three FT models are relevant here:
> - `allenai/OlmoEarth-v1-FT-Mangrove-Base` — 4-class per-pixel segmentation (nodata / mangrove / water / other). **Start here.**
> - `allenai/OlmoEarth-v1-FT-LFMC-Base` — live fuel moisture, useful for post-fire mangrove stress screening.
> - `allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base` — 110-class IUCN ecosystem types for broader context.
>
> Want me to run the Mangrove segmentation first?

---

### Turn 2 — user

> *"Yes, run it over the dry season and show me the class breakdown."*

### Turn 2 — assistant

**tool_calls:**
```json
[
  {
    "id": "call_02",
    "function": {
      "name": "run_olmoearth_inference",
      "arguments": "{\"bbox\":{\"west\":39.6,\"south\":-4.2,\"east\":40.0,\"north\":-3.9},\"model_repo_id\":\"allenai/OlmoEarth-v1-FT-Mangrove-Base\",\"date_range\":\"2024-06-01/2024-09-30\",\"max_size_px\":256,\"sliding_window\":true,\"window_size\":32}"
    }
  }
]
```

**Tool result (trimmed):**
```json
{
  "status": "ready",
  "kind": "pytorch",
  "task_type": "segmentation",
  "num_classes": 4,
  "class_names": ["nodata", "mangrove", "water", "other"],
  "class_names_tentative": false,
  "decoder_key": "mangrove_classification",
  "tile_url": "/api/olmoearth/infer-tile/9d753978.../{z}/{x}/{y}.png",
  "job_id": "9d753978...",
  "scene_id": "S2A_MSIL2A_20240722T074621_R092_T37MBR_...",
  "scene_datetime": "2024-07-22T07:46:21Z",
  "scene_cloud_cover": 2.1,
  "patch_size": 2,
  "legend_classes": [
    { "index": 0, "name": "nodata",   "color": "#6b7280" },
    { "index": 1, "name": "mangrove", "color": "#94eb63" },
    { "index": 2, "name": "water",    "color": "#63d8eb" },
    { "index": 3, "name": "other",    "color": "#eba963" }
  ]
}
```

**Final text:**

> Done — I ran `FT-Mangrove` on a Sentinel-2 L2A scene from 2024-07-22 (2.1% cloud) with sliding window enabled (32 px tiles, patch_size=2).
>
> The result is live on the map as a discrete class map:
> - <span style="color:#94eb63">■ mangrove</span> — dense along the Kilifi Creek / Mida Creek estuaries
> - <span style="color:#63d8eb">■ water</span> — Indian Ocean + estuary channels
> - <span style="color:#eba963">■ other</span> — upland coastal terrace, settlements
> - <span style="color:#6b7280">■ nodata</span> — a few cloud-shadow patches near the edges
>
> Class names are sourced from the published `olmoearth_projects/mangrove/olmoearth_run.yaml` legend — not placeholders.
>
> Want a wet-season comparison, or should I check WorldCover for an independent cross-check?

---

### Turn 3 — user

> *"Yes, do a WorldCover cross-check and also an NDVI trend over the last 12 months so I can see seasonal stress."*

### Turn 3 — assistant

**tool_calls (parallel — both can run independently):**
```json
[
  {
    "id": "call_03a",
    "function": {
      "name": "query_ndvi_timeseries",
      "arguments": "{\"bbox\":{\"west\":39.6,\"south\":-4.2,\"east\":40.0,\"north\":-3.9},\"months\":12,\"max_size_px\":64}"
    }
  }
]
```

> *(WorldCover isn't yet exposed as a chat tool — the LLM should instead tell the user to hit the Analysis tab, OR the developer can wire a new `query_worldcover` tool. In the current build the chat path for WorldCover is "let the user click Analysis".)*

**Tool result (trimmed, 12 months):**
```json
{
  "status": "ok",
  "months_with_data": 11,
  "timeseries": [
    { "month": "2023-11", "ndvi_mean": 0.58, "ndvi_median": 0.61, "ndvi_p10": 0.42, "n_pixels": 4032 },
    { "month": "2023-12", "ndvi_mean": 0.62, "status": "ok" },
    { "month": "2024-01", "status": "no_scene", "detail": "no S2 below 40% cloud" },
    { "month": "2024-02", "ndvi_mean": 0.65, "status": "ok" },
    { "month": "2024-03", "ndvi_mean": 0.71, "status": "ok" },
    { "month": "2024-04", "ndvi_mean": 0.74, "status": "ok" },
    { "month": "2024-05", "ndvi_mean": 0.78, "status": "ok" },
    { "month": "2024-06", "ndvi_mean": 0.76, "status": "ok" },
    { "month": "2024-07", "ndvi_mean": 0.72, "status": "ok" },
    { "month": "2024-08", "ndvi_mean": 0.67, "status": "ok" },
    { "month": "2024-09", "ndvi_mean": 0.63, "status": "ok" },
    { "month": "2024-10", "ndvi_mean": 0.59, "status": "ok" }
  ],
  "formula": "NDVI = (B08 - B04) / (B08 + B04)",
  "source": "PC sentinel-2-l2a · least-cloudy-per-month"
}
```

**Final text:**

> **NDVI timeseries (Nov 2023 → Oct 2024):** seasonal greening peaks at **0.78 in May** (long-rains tail) and troughs at **0.58 in Nov** (short dry). January is missing — no S2 scene below 40 % cloud that month. The **annual amplitude of ~0.20 NDVI is typical for tropical mangroves**; no step drops that would indicate die-off.
>
> **WorldCover cross-check:** I can't run WorldCover from this chat (no tool wired yet). To sanity-check the mangrove classes against ESA's global 2021 map, click the **Analysis tab** — it returns the WorldCover histogram over the same bbox in ~2 s. The *Mangroves* (class 95, `#00cf75`) column should line up roughly with the `FT-Mangrove` **mangrove** class here (usually within ±5 % on intact coastlines).
>
> Would you like me to also run the **LFMC** model to see if any of the mangrove pixels show low fuel moisture (a proxy for salinity / heat stress)?

---

### Turn 4 — user

> *"Yes, run LFMC over the same bbox and dates."*

### Turn 4 — assistant

**tool_calls:**
```json
[
  {
    "id": "call_04",
    "function": {
      "name": "run_olmoearth_inference",
      "arguments": "{\"bbox\":{\"west\":39.6,\"south\":-4.2,\"east\":40.0,\"north\":-3.9},\"model_repo_id\":\"allenai/OlmoEarth-v1-FT-LFMC-Base\",\"date_range\":\"2024-06-01/2024-09-30\",\"sliding_window\":true,\"window_size\":32,\"max_size_px\":256}"
    }
  }
]
```

**Tool result (trimmed):**
```json
{
  "status": "ready",
  "kind": "pytorch",
  "task_type": "regression",
  "model_repo_id": "allenai/OlmoEarth-v1-FT-LFMC-Base",
  "prediction_value": null,
  "units": "% live fuel moisture",
  "tile_url": "/api/olmoearth/infer-tile/c2b11f.../{z}/{x}/{y}.png",
  "scene_id": "S2A_MSIL2A_20240722T074621_...",
  "legend_classes": [],
  "notes": [
    "Sliding window on — per-tile LFMC regression produces a spatial moisture map.",
    "Value range normalized to 30–200 % for the colormap gradient."
  ]
}
```

**Final text:**

> **LFMC layer added** — red-yellow-green gradient over the bbox. Because `sliding_window=true` the output varies per 32 px tile rather than being a single scene-level average.
>
> Observed pattern:
> - **Green (>130 %)** over the interior mangrove stands around Mida Creek — healthy canopy with saturated foliage.
> - **Yellow (~90 %)** at the landward fringe — transition to `other` class, lower canopy density.
> - **Red patches (<60 %)** on a handful of degraded / cleared polygons. **These are candidates for ground-truth follow-up** — cross-reference with the `FT-Mangrove` layer: if they're classified as **mangrove** but show low LFMC, they may be degrading.
>
> Flip into **⇌ Compare mode** (top-right of the map) and pick A = `FT-Mangrove`, B = `FT-LFMC` to swipe between the two and find exactly the pixels where both conditions hold. That's your stress shortlist.

---

## Conversation 2 — PA Karst forest-loss investigation (BAI case study)

**Setup:**
- User imports the PA Karst sample dataset (`PA Karst Features` from Sample Data) — vector sinkhole polygons over Lehigh / Lancaster / Cumberland counties.
- One polygon is selected (`polygon_id: "sink_42"` — a 1.2 ha depression).

### Turn 1 — user

> *"Polygon sink_42 is a confirmed sinkhole. Has there been any forest loss around it in the last two years, and if so, what caused it?"*

### Turn 1 — assistant

**tool_calls (parallel):**
```json
[
  {
    "id": "call_01a",
    "function": {
      "name": "query_polygon",
      "arguments": "{\"polygon_id\":\"sink_42\"}"
    }
  },
  {
    "id": "call_01b",
    "function": {
      "name": "query_polygon_stats",
      "arguments": "{\"polygon_id\":\"sink_42\",\"include_elevation\":true}"
    }
  }
]
```

**Tool results:**
```json
// call_01a
{
  "found": true, "polygon_id": "sink_42",
  "properties": { "KARST_TYPE": "sinkhole", "COUNTY": "Lancaster", "source": "PaGEODE" },
  "bbox": { "west": -76.31, "south": 40.12, "east": -76.29, "north": 40.14 }
}
// call_01b
{
  "perimeter_km": 0.43, "area_km2": 0.012,
  "centroid": { "lat": 40.131, "lon": -76.299 },
  "elevation": { "min_m": 112, "median_m": 119, "max_m": 124, "mean_m": 118.5 }
}
```

**Final text of Turn 1 (continuing directly to Turn 2 after more tools):**

> Confirmed sinkhole, Lancaster County, 1.2 ha, ~118 m elevation with a 12 m depression (normal karst). Now pulling forest-loss driver + NDVI timeseries to answer your question…

---

### Turn 2 — assistant (continuing without user turn — multi-tool chain)

**tool_calls (parallel — the agent loop fires both because they're independent):**
```json
[
  {
    "id": "call_02a",
    "function": {
      "name": "run_olmoearth_inference",
      "arguments": "{\"bbox\":{\"west\":-76.31,\"south\":40.12,\"east\":-76.29,\"north\":40.14},\"model_repo_id\":\"allenai/OlmoEarth-v1-FT-ForestLossDriver-Base\",\"date_range\":\"2024-04-01/2024-10-01\",\"sliding_window\":true}"
    }
  },
  {
    "id": "call_02b",
    "function": {
      "name": "query_ndvi_timeseries",
      "arguments": "{\"bbox\":{\"west\":-76.31,\"south\":40.12,\"east\":-76.29,\"north\":40.14},\"months\":24,\"max_size_px\":48}"
    }
  }
]
```

**Tool results:**
```json
// call_02a (ForestLossDriver — 10 classes, sliding_window upgraded to segmentation)
{
  "task_type": "segmentation",
  "class_names": ["agriculture","mining","airstrip","road","logging","burned","landslide","hurricane","river","none"],
  "class_names_tentative": false,
  "legend_classes": [
    { "index": 4, "name": "logging",     "color": "#800080" },
    { "index": 0, "name": "agriculture", "color": "#89f336" },
    { "index": 9, "name": "none",        "color": "#ffffff" }
  ],
  "tile_url": "/api/olmoearth/infer-tile/a71f2.../{z}/{x}/{y}.png"
}

// call_02b (24-month NDVI)
{
  "status": "ok", "months_with_data": 22,
  "timeseries": [
    { "month": "2022-11", "ndvi_mean": 0.31 },
    "...",
    { "month": "2023-07", "ndvi_mean": 0.82 },
    { "month": "2023-11", "ndvi_mean": 0.43 },
    { "month": "2024-05", "ndvi_mean": 0.48, "__flag": "delta_vs_2023_may = -0.31"},
    { "month": "2024-07", "ndvi_mean": 0.56 },
    "..."
  ]
}
```

**Final text:**

> **Finding: there IS forest loss around sink_42, and the driver is `logging`.**
>
> - `FT-ForestLossDriver` over a summer 2024 window flags a ~0.4 ha patch just north of the sinkhole as **logging** (class 4, `#800080`). No `agriculture` or `burned` signal — rules out clearing-for-planting and fire.
> - NDVI confirms: **May 2024 NDVI 0.48** vs. **May 2023 NDVI 0.79** over the same bbox — a **Δ = −0.31 drop**, which is consistent with a selective-cut event between July 2023 and May 2024. By July 2024 NDVI has partially rebounded to 0.56 (early-succession ground cover), so the stand was not clear-cut.
> - The logged patch is uphill of the sinkhole — worth monitoring for increased sediment runoff into the cavity.
>
> The Compare mode is a good next step: put the ForestLossDriver layer on A and a Sentinel-2 true-color composite (via the Analysis tab or `search_stac_imagery`) on B, and swipe between 2023 and 2024.

---

## Conversation 3 — SF Parks multi-class ecosystem audit

**Setup:** user loads the `SF Parks & Landmarks` sample dataset → 7 park polygons over San Francisco.

### Turn 1 — user

> *"For each park polygon, classify the ecosystem type using OlmoEarth and tell me which one has the most diverse classes."*

### Turn 1 — assistant

> *"7 polygons is a lot for a single bbox — running per-polygon gives per-park ecosystem profiles. I'll fire one `run_olmoearth_inference` per polygon bbox with the EcosystemTypeMapping FT head and sliding_window=true, then aggregate the class histograms."*

**tool_calls (one per polygon, agent loop issues them sequentially):**
```json
[
  {
    "id": "call_01_gg",
    "function": {
      "name": "run_olmoearth_inference",
      "arguments": "{\"bbox\":{\"west\":-122.51,\"south\":37.765,\"east\":-122.46,\"north\":37.785},\"model_repo_id\":\"allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base\",\"date_range\":\"2024-06-01/2024-09-30\",\"sliding_window\":true,\"window_size\":32,\"max_size_px\":128}"
    }
  }
  // (+ 6 more calls, one per park)
]
```

**Tool result (Golden Gate Park — trimmed; others similar):**
```json
{
  "task_type": "segmentation",
  "num_classes": 110,
  "class_names": [ /* 60 published, 50 placeholders */ ],
  "class_names_tentative": false,
  "legend_classes": [
    { "index": 14, "name": "T_7_2_SOWN_PASTURES_AND_FIELDS",         "color": "#ca0079" },
    { "index": 16, "name": "T_7_4_URBAN_AND_INDUSTRIAL_ECOSYSTEMS",  "color": "#8b0047" },
    { "index": 25, "name": "F_2_1_LARGE_PERMANENT_FRESHWATER_LAKES", "color": "#d6fbff" },
    { "index": 2,  "name": "T_1_3_TROPICAL/SUBTROPICAL_MONTANE_RAINFORESTS", "color": "#6fd9a1" }
  ]
}
```

**Final text (after all 7 tool results):**

> Per-park ecosystem profiles (dominant class + secondary classes):
>
> | Park | Dominant | Secondary | Classes present |
> |---|---|---|---|
> | Golden Gate Park | `T_7_2_SOWN_PASTURES_AND_FIELDS` | `T_7_4_URBAN_AND_INDUSTRIAL`, `F_2_1_FRESHWATER_LAKES` | **4** |
> | Presidio | `T_7_4_URBAN_AND_INDUSTRIAL` | `T_1_3_MONTANE_RAINFORESTS`, `MT_1_3_SANDY_SHORELINES` | **3** |
> | Lands End | `MT_1_1_ROCKY_SHORELINES` | `T_7_4_URBAN_AND_INDUSTRIAL` | 2 |
> | Dolores Park | `T_7_4_URBAN_AND_INDUSTRIAL` | `T_7_2_SOWN_PASTURES` | 2 |
> | Alamo Square | `T_7_4_URBAN_AND_INDUSTRIAL` | (uniform) | 1 |
> | Mission Creek | `F_2_1_FRESHWATER_LAKES` | `T_7_4_URBAN_AND_INDUSTRIAL` | 2 |
> | Crissy Field | `MT_1_3_SANDY_SHORELINES` | `T_7_4_URBAN`, `T_7_2_FIELDS` | **3** |
>
> **Winner: Golden Gate Park**, 4 distinct ecosystem classes (sown fields + urban + freshwater lakes + a montane-rainforest signal from the eucalyptus/Monterey-cypress canopy — that last one is likely a domain-shift artifact; the IUCN scheme doesn't have a good class for urban-planted temperate conifers).
>
> Note: the `T_1_3_MONTANE_RAINFORESTS` hit on both GG Park and the Presidio is worth flagging — the OlmoEarth pretraining corpus is global and the FT head was trained on the 110-class IUCN scheme; dense non-native canopy in a Mediterranean climate is an **out-of-distribution** case for this head. Treat those pixels as "dense broadleaf/evergreen canopy" rather than literal tropical rainforest.
>
> The 7 overlay layers are all live on the map — toggle them in the OlmoEarth panel or use ⇌ Compare to swipe between any two parks.

---

## How to read these transcripts

- **`tool_calls` JSON** is the *exact* wire format Gemma / NIM / Claude emit and the backend dispatches. Every argument shown is a real parameter of the registered tool schema.
- **Tool result** is what the `_tool_*` executor returns — normally embedded as a `role=tool` message the model reads in the next turn.
- **Final text** is the post-synthesis reply the user sees in the chat bubble.
- **Chained tools in one turn**: the agent loop supports parallel calls — the model issues multiple `tool_calls` in a single assistant message, the backend runs them concurrently, and the next assistant turn sees all the results together.

## Why these are "advanced" use cases

1. **Multi-tool chaining** — one user question triggers 2+ tool calls per turn (catalog + polygon stats + NDVI + FT inference).
2. **Cross-model inference** — running two FT heads (Mangrove + LFMC) over the same bbox and correlating their outputs.
3. **Temporal + spatial fusion** — combining a per-month NDVI timeseries with a per-patch segmentation raster.
4. **Out-of-distribution reasoning** — the assistant flags when an FT head's classes don't cleanly map to the user's domain (SF eucalyptus canopy as tropical rainforest).
5. **Actionable follow-ups** — every turn ends with a concrete next step: Compare mode, Analysis tab, a specific polygon to inspect.

---

## How to exercise these in the current build

1. Start the backend + frontend (see [CAPABILITIES.md §Quick commands](./CAPABILITIES.md)).
2. Launch the LLM of your choice (Local Gemma, NIM, or Claude) from the LLM tab.
3. Either load a sample (Try Demo / Sample Data card) or draw a polygon.
4. Paste one of the user turns above verbatim into the chat composer — the model should emit the same (or very similar) `tool_calls`.
5. If the local Gemma model narrates instead of calling the tool, the **auto-retry + amber chip** mechanism described in [FT-USAGE.md](./FT-USAGE.md) will kick in and repair the turn; for consistently clean tool use, switch to Claude Sonnet 4.6 or NIM GPT-OSS 20B in their respective tabs.

## Known limitations in these transcripts

- **Conversation 1, Turn 3**: there's no `query_worldcover` tool in the current build — the assistant correctly defers to the Analysis tab. A follow-up PR could wire that in as a chat tool for parity.
- **Conversation 2**: `FT-ForestLossDriver` expects pre/post S2 pairs via `SimpleTimeSeries` in the reference training recipe. The current Roger loader reconstructs the final classification head only, so the class assignments are an approximation — accurate on clean 6-month separations like the sink_42 example, degraded on subtler change windows.
- **Conversation 3**: issuing 7 parallel FT inferences on CPU torch will take ~3–7 min total; use a GPU torch build or switch to Nano for faster multi-polygon sweeps.

---

**See also:**
- [CAPABILITIES.md](./CAPABILITIES.md) — the full "what Roger can do" reference.
- [FT-USAGE.md](./FT-USAGE.md) — four paths to run FT models (UI / LLM / HTTP / Python).
