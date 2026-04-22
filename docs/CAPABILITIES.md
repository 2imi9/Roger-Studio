# Roger Studio — What You Can Do

> Earth Observation Copilot. Draw a polygon, pull live imagery, run real OlmoEarth foundation-model inference, overlay predictions, and talk to a geo-aware LLM — all in one interface.

**Stack in a nutshell:** React 18 + Vite + Tailwind + MapLibre (2D) + CesiumJS (3D) on port **3000**, FastAPI/uvicorn on port **8000**, optional vLLM on port **8001**.

---

## 1. Draw & inspect an area

- Click the map to draw **rectangles**, **polygons**, **points**, or **lines** using terra-draw.
- Load a preset with **Try Demo (Kenyan Coast)**, or pick a **Sample Data** card (SF Parks, PA Karst, Solar Sites, Knoxville NDVI).
- The Sidebar shows live **Selected Area** coords and **Polygon Stats** — perimeter (km), area (km²), elevation min/median/max/mean (m) from Open-Meteo.
- Basemap switcher: OSM · Esri Satellite · CartoDB Dark.
- Collapse toggles for header + sidebar to give the map back full width.

---

## 2. Pull real Sentinel-2 imagery

- Microsoft Planetary Computer STAC (`sentinel-2-l2a`, `sentinel-1-grd`, `landsat-c2-l2`, `naip`) — no API keys required for reads.
- `POST /api/stac/search` finds scenes; `POST /api/stac/composite-tile-url` registers a cloud-free mosaic and returns an XYZ tile URL MapLibre drops straight onto the map.
- Date-range picker + cloud-cover filter + per-collection sensible defaults (B04/B03/B02 true-color for S2, NIR composites, etc.).

---

## 3. Browse the live OlmoEarth catalog

- `GET /api/olmoearth/catalog` pulls the current `allenai/OlmoEarth-*` list from the Hugging Face API with a 10-min TTL + a hardcoded fallback so the UI stays alive offline.
- Sidebar **OlmoEarth tab** has three subtabs — **Encoder**, **Finetune head**, **Dataset** — each showing HF downloads/likes + Load / Cached · size / × unload.
- Heavy repos (~hundreds of GB training corpus) are gated behind a confirm dialog.
- **Coverage for this area** panel overlays polygons for project regions intersecting the current bbox.
- `recommend_model` picks an FT repo by keyword when the user asks.

---

## 4. Run **real** OlmoEarth inference on a bbox

Click **Run** on a cached model and the backend:

1. Fetches a least-cloudy **Sentinel-2 L2A** composite from Planetary Computer, reading all 12 bands in the official OlmoEarth band order (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09) at 10 m/pixel.
2. Normalizes via the official `Normalizer(Strategy.COMPUTED)`.
3. Builds a `MaskedOlmoEarthSample` and runs `model.encoder(sample, fast_pass=True, patch_size=…)`.
4. Caches the resulting prediction raster + CRS/transform.
5. Serves XYZ tiles from `/api/olmoearth/infer-tile/{job_id}/{z}/{x}/{y}.png` — MapLibre adds it as a raster layer automatically.

Four task paths are supported, dispatched by repo name:

| Repo | Task | Output |
|---|---|---|
| `allenai/OlmoEarth-v1-{Nano,Tiny,Base,Large}` | Encoder | PCA scalar of per-patch embedding → gradient colormap (generic "show me what the encoder sees"). |
| `allenai/OlmoEarth-v1-FT-Mangrove-Base` | 4-class per-pixel **segmentation** | Discrete class raster · `nodata`, `mangrove`, `water`, `other`. |
| `allenai/OlmoEarth-v1-FT-AWF-Base` | 10-class **segmentation** | `woodland_forest`, `open_water`, `shrubland_savanna`, `herbaceous_wetland`, `grassland_barren`, `agriculture_settlement`, `montane_forest`, `lava_forest`, `urban_dense_development`, `nodata`. |
| `allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base` | 110-class IUCN **segmentation** | Discrete class raster (60 named + placeholders for the tail). |
| `allenai/OlmoEarth-v1-FT-ForestLossDriver-Base` | 10-way **classification** | Scene-level driver: `agriculture`, `mining`, `airstrip`, `road`, `logging`, `burned`, `landslide`, `hurricane`, `river`, `none`. |
| `allenai/OlmoEarth-v1-FT-LFMC-Base` | Per-pixel **regression** | Live fuel moisture (%) mapped to a red-yellow-green gradient over 30–200%. |

**Sliding window mode** (optional `sliding_window=true`): tiles the bbox into non-overlapping 32-px training-size windows and runs the head per tile. Converts scene-level classification into a spatial class map and gives LFMC regression per-tile moisture values instead of a single scene average.

**Fallback:** if S2 fetch or the forward pass errors, the response still carries a valid tile URL — marked `kind="stub"` with a `stub_reason`, rendered as a watermarked gradient so no one confuses it with real output.

Class names + colors come from `allenai/olmoearth_projects/olmoearth_run_data/<task>/olmoearth_run.yaml` — no guessing.

---

## 5. Legend panel

The OlmoEarth sidebar tab renders a legend for every active inference layer:

- Model name + HF repo link.
- Scene id · date · cloud-cover percentage.
- Task badge (classification / segmentation / regression / embedding).
- For classification + segmentation: per-class rows with the published hex swatch, index, and name.
- For regression: gradient bar + predicted value with units (e.g. `52.3 % live fuel moisture`).
- For encoder-only: embedding dim + colormap bar.
- **Remove layer** button per card.

A small banner flags `class names tentative` when the label list hasn't been confirmed against the public rslearn config.

---

## 6. Land cover + NDVI timeseries (real, not heuristic)

**`POST /api/analyze`**
- Histogram of ESA **WorldCover 2021** (10 m/pixel COG via PC) over the bbox.
- 11 official classes: Tree cover, Shrubland, Grassland, Cropland, Built-up, Bare / sparse vegetation, Snow and ice, Permanent water bodies, Herbaceous wetland, Mangroves, Moss and lichen.
- Falls back to the legacy latitude heuristic only if PC is down — the `olmoearth.land_cover_source` field tells you which ran.

**`query_ndvi_timeseries` tool**
- Monthly NDVI mean / median / p10 / p90 over the last N months.
- For each month: least-cloudy S2 scene via PC, `NDVI = (B08 - B04) / (B08 + B04)` over non-zero pixels, returns scene id + date + cloud cover per month.

---

## 7. Upload your own raster — stream as tiles

- Drop a **GeoTIFF** (or NetCDF, Zarr, GeoPackage, GeoJSON, Shapefile `.zip`, LAS/LAZ, GeoParquet, CSV) onto the **Import Data** panel.
- Metadata is inspected on upload: dimensions, bands, dtype, nodata, resolution, CRS, bbox.
- **`+ Add as map layer`** in the dataset detail view turns the raw file into XYZ tiles via `GET /api/datasets/{filename}/tiles/{z}/{x}/{y}.png`.
- 1-band rasters → colormap gradient; 3+ band rasters → percentile-stretched RGB composite.
- Lets you compare an uploaded ground-truth raster against an OlmoEarth inference layer side-by-side.

---

## 8. Side-by-side Compare mode

- Once you have ≥ 1 imagery layer, a **⇌ Compare** button appears top-right of the map.
- Swaps the normal map for a vertical-split view (`@maplibre/maplibre-gl-compare`) with A / B pickers — pick any two layers (uploaded GeoTIFF vs inference prediction, or two different FT models on the same bbox).
- Drag the divider to swipe between the two. Exit restores the normal MapView.

---

## 9. Build Labels — manual annotation (OlmoEarth Studio-compatible)

- Name a project, pick label type (**Polygon / Point / Line**), pick a tag from an 8-item land-cover default list or add a custom tag.
- Start / Stop the additive draw toggle; terra-draw stays in the chosen mode until you stop.
- Per-tag count summary + per-feature delete.
- **Download GeoJSON** — standard FeatureCollection with a Roger-specific header (`project_name`, `generated_at`, `source: "cuvier-studio-manual-labels"`, `feature_count`).
- Shape matches OlmoEarth Studio's label-export schema so files round-trip cleanly.
- Persists to `sessionStorage` between reloads; no server-side storage required.

---

## 10. Geo-aware LLM chat (three providers)

The **LLM tab** has four subpanes:

- **💻 Local** — Gemma 4 E4B via vLLM (Docker / WSL) or Ollama, tool-calling.
- **☁️ NIM** — NVIDIA NIM cloud (6 preset models: GPT-OSS 20B/120B, Llama 3.3 70B, Nemotron Super 49B, DeepSeek V3.1, Qwen3 32B). Inline API-key input.
- **🧠 Claude** — Anthropic Messages API, 4 preset models (Opus 4.7 default, Opus 4.6, Sonnet 4.6, Haiku 4.5). Inline `sk-ant-…` input. Adaptive thinking + prompt caching.
- **⚙ Settings** — vLLM/Ollama config for the Local pane.

All three chat clients share the **same 8 tools**:

| Tool | What it does |
|---|---|
| `query_polygon` | Look up one polygon from the current scene by id / index. |
| `query_polygon_stats` | Perimeter / area / elevation stats (Google Earth-style). |
| `query_olmoearth` | Live catalog + OlmoEarth project regions intersecting a bbox. |
| **`run_olmoearth_inference`** | Fetch S2 + run a real OlmoEarth forward pass → tile URL + task-tagged prediction. Supports `sliding_window`. |
| `query_ndvi_timeseries` | Monthly NDVI mean / median / p10 / p90 over a bbox. |
| `search_stac_imagery` | PC STAC search. |
| `get_composite_tile_url` | PC mosaic endpoint → XYZ tile URL. |
| `get_higher_res_patch` | Zoomed basemap tile for a polygon (stub — pending). |

**The agent can now answer**: *"What's happening in this area?"* end-to-end — it calls `query_olmoearth` to see what FT models exist, picks one (e.g. Mangrove for tropical coast), calls `run_olmoearth_inference` to actually run it, gets back the tile URL + class legend, and describes the result in plain language while the layer appears on the map.

---

## 11. 3D globe

- **3D Globe** tab renders the bbox / polygon on CesiumJS.
- Uses `PolygonHierarchy` for real polygons, falls back to a bbox rectangle.
- Pulls analysis + env data onto the globe for a quick "see this area in terrain" view.

---

## What's cached / stateful

- **HF model cache**: `~/.cache/huggingface/hub/` — Load downloads the whole snapshot, × unloads it.
- **Inference job registry**: in-memory, keyed by spec hash so the same bbox + model returns the same job.
- **Live STAC / WorldCover / SAS tokens**: per-process TTL caches (10 min, 1 h, etc.).
- **Session state**: labels, project name, custom tags, chat history, picked models, API keys — all scoped to sessionStorage so a tab refresh keeps your work.

---

## Honest limitations — what we DON'T do yet

- **GPU**: the installed torch is CPU-only. Nano is fast (~1–3 s per forward); Base / Large will be slow until a CUDA wheel is pinned. The RTX 5090 is ready for it.
- **ForestLossDriver** expects a pre/post S2 pair via `SimpleTimeSeries`; our current loader extracts only the final Linear, so predictions work but the temporal pairing isn't yet wired.
- **LFMC** uses a UNet decoder in the real repo; we reconstruct only the last layer, so output is scene-level unless `sliding_window=true` gives per-tile values.
- **Ecosystem** head emits 110 logits but only 60 are named publicly — the tail falls back to `class_N` placeholders.
- **`get_higher_res_patch` tool** is still a stub (separate basemap subsystem).
- **Cloud Chat + Claude Chat** have been exercised up to the health / catalog hop; actual chat round-trips need a real key.

---

## Testing posture

- **76 backend tests, all passing in ~95 s** on CPU.
- Offline tier (64 tests): shape invariants, schema checks, head reconstruction, sliding-window stitching, raster-tile rendering, tool dispatch, stub fallback, sample-data parsers.
- Network tier (12 tests, marked `network`): Nano forward pass + weight cache, Seattle S2 fetch, ESA WorldCover histogram, real inference + tile serve, monthly NDVI, FastAPI end-to-end (TestClient), real FT-Mangrove segmentation with class legend.
- Windows PC cold-connect flakiness handled via `--reruns 2 --reruns-delay 2`.

---

## Quick commands

```bash
# Backend
cd geoenv-studio/backend
uv sync                       # or pip install -e .
python run.py                 # FastAPI on :8000

# Frontend
cd geoenv-studio/frontend
pnpm install
pnpm dev                      # Vite on :3000

# Tests
cd geoenv-studio/backend
PYTHONPATH=. python -m pytest tests/ -m "not network"     # fast offline
PYTHONPATH=. python -m pytest tests/ --reruns 2           # full
```

---

Studio wordmark: **Fraunces 900 + italic accent-blue "Studio"** + IBM Plex Mono caps tag. Logo: striped earth ring + viewfinder bracket in refractive-blue glass plate (hover-track + click-spin). Cream theme, 33-icon Figma-exported line-art set across all tabs.
