<div align="center">
  <img src="docs/assets/logo.svg" alt="Roger Studio" width="160" />

# Roger Studio

**An Earth Observation copilot for remote-sensing research.**

Draw a bounding box on a map → fetch Sentinel-2 imagery → run fine-tuned
Earth models → talk to a tool-augmented LLM agent. All from one
locally-hosted web workbench.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)
[![Tests: 223 passing](https://img.shields.io/badge/tests-223%20passing-brightgreen.svg)](#testing)
[![Built on OlmoEarth](https://img.shields.io/badge/built%20on-OlmoEarth-orange.svg)](https://huggingface.co/allenai)

[Quick start](#quick-start) · [Architecture](#architecture) · [Features](#features) · [Documentation](#documentation) · [Contributing](#contributing)

</div>

---

## What is Roger Studio?

Roger Studio is a desktop-class workbench for running production
Earth-observation models on user-drawn bounding boxes. It pairs
**[OlmoEarth](https://huggingface.co/allenai)** fine-tuned heads
(mangrove extent, land use, ecosystem type, fire fuel moisture, forest-loss
drivers) with **[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)**
imagery and a chat surface that drives every tool in the app via
function calling — the local LLM is any
[**Gemma**](https://ai.google.dev/gemma) family model
([tech report](https://arxiv.org/abs/2503.19786) — sizes from
~2B to ~27B, instruction-tuned variants, Google or Unsloth
quantizations all work) served via Ollama or vLLM, plus **Claude**,
**Gemini**, **GPT**, and **NVIDIA NIM** as cloud alternatives.

It is built for one job: getting a researcher from "I have a question
about this patch of Earth" to "I have a defensible classified raster +
GeoJSON polygons" without a GPU cluster, a Jupyter notebook, or a
remote-sensing PhD.

### Who it's for

| Audience | What you get |
|---|---|
| **Conservation NGO analyst** | One-click mangrove / forest-loss-driver maps over your field site, exportable as GeoJSON for QGIS or Google Earth Pro. |
| **EO research scientist** | Native 10 m/pixel inference on chunked AOIs, embedding COG exports for downstream sklearn pipelines, a few-shot tool for ad-hoc class taxonomies. |
| **ML engineer evaluating OlmoEarth** | A self-contained reproduction harness — clone, drop weights into HF cache, run any FT head on any AOI in 1-3 minutes. No notebooks. |

---

## Quick start

Roger Studio is a two-process app: a Python FastAPI backend (the
inference orchestrator) and a Vite/React frontend (the workbench).

### Prerequisites

- **Python 3.11+** with `pip`
- **Node 20+** with `pnpm` (or `npm`)
- **~10 GB disk** for the OlmoEarth FT head + base encoder weights
  (cached on first run via Hugging Face)
- **~16 GB RAM** for the smallest configuration; **24 GB VRAM** GPU
  recommended for real inference (CPU works for testing)

### 1. Clone

```bash
git clone https://github.com/2imi9/Roger-Studio.git
cd Roger-Studio
```

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: drop in cloud LLM API keys
cp .env.example .env  # edit ANTHROPIC_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

# Recommended safety env for a laptop GPU:
export OE_MAX_CONCURRENT_JOBS=1
export OE_MAX_CHUNKS=100
python run.py                         # serves http://localhost:8000
```

### 3. Frontend

```bash
cd frontend
pnpm install                          # or: npm install
pnpm dev                              # serves http://localhost:3000
```

### 4. Open http://localhost:3000

The **Introduction Doc** button next to the Project menu launches a
guided tour of every tab. The fastest path to a real result:

1. Map tab → Import Data → pick **Forest-loss driver**
2. Click **↳ Use demo AOI (Pará, Brazilian Amazon)** → AOI + event
   date auto-fill
3. Click **Run + add to map** → in ~2 minutes a real 10-class
   forest-loss classification renders on top of OpenStreetMap

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  React + MapLibre GL frontend  (port 3000)                           │
│  ┌─────────┬───────────┬───────────┬─────────┐                       │
│  │  Map    │ Analysis  │ OlmoEarth │   LLM   │     ← four top tabs   │
│  └────┬────┴─────┬─────┴─────┬─────┴────┬────┘                       │
└───────┼──────────┼───────────┼──────────┼────────────────────────────┘
        │          │           │          │
        │ /upload  │/analyze   │/infer    │/claude /gemini /openai
        │ /datasets│/polygon-  │/ft-      │/cloud  /auto-label/gemma
        │ /stac    │ stats     │ classifi-│
        │ /artifacts          │ cation/  │
        │          │/env-data  │ geojson  │
        │          │           │/embedding-tools/{similarity,pca-rgb,
        │          │           │  few-shot}  /export-embedding
        │          │           │/v1/projects/*  (save / load)
┌───────▼──────────▼───────────▼──────────▼────────────────────────────┐
│  FastAPI backend  (port 8000)                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Inference orchestrator                                        │  │
│  │    chunked AOI plan → per-chunk fetch → encoder → FT head      │  │
│  │    safety stack: RAM gate · circuit breaker · disconnect-cancel│  │
│  │  Sentinel-2 fetch                                              │  │
│  │    Planetary Computer STAC search · per-band windowed reads    │  │
│  │    on-disk scene cache (~50× faster on warm runs)              │  │
│  │  OlmoEarth model loader                                        │  │
│  │    Hugging Face → torch.nn.Module · LRU cache (1 model in VRAM)│  │
│  │  Tool registry                                                 │  │
│  │    11 geo-tools shared across 5 LLM backends + an MCP server   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────┬─────────────────────────────────────┬──────────────────────┘
          │                                     │
          ▼                                     ▼
   Microsoft Planetary Computer          Hugging Face Hub
     (Sentinel-2 L2A, WorldCover)          (OlmoEarth weights, Gemma)
```

**Key design choices** (full rationale in [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md)):

* **Real inference, no stubs** in the happy path. When real fetch /
  forward fails, the response carries `kind: "stub"` + a
  `stub_reason` so the UI can prompt a retry. Auto-retry is wired for
  common transient failures.
* **Chunked AOI plan** at 5 km × 5 km tiles so a 50 km AOI doesn't
  blow out VRAM, with a per-chunk RAM gate that refuses to fetch when
  the OS is approaching swap.
* **Pinned UTM grid** across all chunks so per-tile outputs paste into
  one global pixel-aligned raster — no reprojection seams.
* **Lifted state for picks + AOIs** (App.tsx) so the popover and
  sidebar share one source of truth; persisted via the
  `/v1/projects` endpoint.
* **Tool functions live in one registry** (`backend/app/services/geo_tools.py`)
  so adding a new tool surfaces it everywhere — Local Gemma, NIM,
  Claude, Gemini, OpenAI, and the standalone MCP server.

---

## Features

### Fine-tuned OlmoEarth heads

Five fine-tuned heads ship out of the box. All run on Sentinel-2 L2A
through the chunked native-resolution pipeline; outputs render as map
tiles + downloadable GeoJSON polygons.

| Head | Task | Output | Trained region | Status |
|---|---|---|---|---|
| `Mangrove-Base` | Mangrove extent | 4-class segmentation | Tropical coastal belt | ✅ |
| `AWF-Base` | Land use | 10-class segmentation | Southern Kenya | ✅ |
| `EcosystemTypeMapping-Base` | IUCN ecosystem L3 | 110-class segmentation | North Africa | ✅ |
| `LFMC-Base` | Live fuel moisture | per-pixel regression (%) | Fire-prone regions | ✅ |
| `ForestLossDriver-Base` | Driver of forest loss | 10-class scene-level | Pantropical | ✅ pre/post pair |

ForestLossDriver uses a pre/post Sentinel-2 pair concatenated along the
encoder feature dim (768 + 768 → 1536). See
[`docs/FOREST_LOSS_DRIVER.md`](docs/FOREST_LOSS_DRIVER.md) for the
event-date contract and class list.

#### Sliding-window inference (`sliding_window: true`)

Every classification head also accepts a **sliding-window** mode that
runs the head per ``predict_window_px`` window inside each chunk
instead of once per chunk. For ForestLossDriver this turns "one
driver class per ~25 km² chunk" into "one class per ~64 px window"
— **16× finer spatial granularity**. The window size is chosen
automatically from each head's rslearn metadata
(`predict_window_px`); fall back is the request's `window_size`
parameter (default 64).

The OlmoEarthImport panel toggles this on by default for any FT head.
Trade-off is wall time: a 5 km chunk emits ~49 windowed forwards
instead of 1, so total runtime roughly doubles. For pre/post heads
(ForestLossDriver) it triples — pre + post are encoded per window.

When sliding-window is on, classification heads' `task_type` is
reported as `segmentation` since the output now varies spatially —
the existing GeoJSON export and tile renderer paths handle this
uniformly.

### Embedding tools (no fine-tuning needed)

When no FT head fits the AOI or task, run a base encoder (Nano / Tiny /
Base / Large) and use the raw embedding directly:

| Tool | When to reach for it | Docs |
|---|---|---|
| **PCA false-color** | Quick visual sanity check; see landscape diversity at a glance | [EMBEDDINGS.md](docs/EMBEDDINGS.md) |
| **Cosine similarity** | "Where else looks like this pixel?" — click anywhere on the map to set the query | [EMBEDDINGS.md](docs/EMBEDDINGS.md) |
| **Few-shot segmentation** | Multi-class ad-hoc taxonomy from ~5-30 labelled clicks per class | [FEW_SHOT.md](docs/FEW_SHOT.md) |
| **COG export** | Stream the float32 → int8 quantized embedding GeoTIFF for offline sklearn / change-detection / similarity workflows | [EMBEDDINGS.md](docs/EMBEDDINGS.md) |

The COG layout is bit-for-bit compatible with Ai2 OlmoEarth Studio's
published format — recover float vectors via
`olmoearth_pretrain.evals.embedding_transforms.dequantize_embeddings`.

### LLM tool layer

11 geo-tools (catalog lookup, `/analyze`, polygon stats, NDVI timeseries,
WorldCover histogram, FT inference, raster explanation, …) shared across
five chat backends:

| Backend | Provider | Tool calling |
|---|---|---|
| **Local Gemma** ([family / tech report](https://arxiv.org/abs/2503.19786)) | Ollama or vLLM, any size from 2B → 27B (24 GB GPU recommended for 27B) | ✅ via OpenAI-shape function-calling |
| **NVIDIA NIM** | Cloud, NIM-hosted models | ✅ |
| **Claude** | Anthropic API | ✅ |
| **Gemini** | Google AI Studio | ✅ |
| **ChatGPT** | OpenAI API | ✅ |

Multi-round conversation examples live in
[`docs/LLM-CONVERSATIONS.md`](docs/LLM-CONVERSATIONS.md). For local
Gemma setup (Ollama vs vLLM trade-offs, Docker recipes, GPU sizing):
[`docs/llm-setup.md`](docs/llm-setup.md) and
[`docs/gemma-vllm.md`](docs/gemma-vllm.md).

### Workflow primitives

* **Per-model demo AOIs** — one click sets a small (~3 km) bbox over a
  region the selected model was trained on; ForestLossDriver also
  auto-fills the event date.
* **Pixel-pick on map** — crosshair-cursor mode for the similarity tool
  and few-shot labelling.
* **Project save/load** — every session (AOI, layers, labels, chat,
  picks) persists to SQLite via `/api/v1/projects/*`.
* **Compare mode** — side-by-side A/B raster inspection via
  `@maplibre/maplibre-gl-compare`.
* **Sample data** — six curated samples (3 vector, 3 raster) one click
  away for tour / demo flows.
* **Auto-Label** — TIPSv2 (zero-shot text-prompted), Spectral (k-means
  clustering), or SamGeo (Meta SAM-based pixel-accurate) for
  bootstrapping labels on imported imagery.
* **GeoJSON export** — every classification job (FT head or few-shot)
  vectorises into RFC-7946 polygons with per-class colour, area, and
  pixel count. See [`docs/GEOJSON_EXPORT.md`](docs/GEOJSON_EXPORT.md).

---

## Performance

Roger Studio targets **native 10 m/pixel inference**. Three layers of
optimization keep that practical on a residential connection:

| Layer | Win |
|---|---|
| **Chunked AOI** | 5 km tiles run independently; a 50 km AOI uses bounded RAM |
| **Parallel fetching** | 4 chunks × 6 periods × 12 bands = up to 288 concurrent reads, GDAL tuned for HTTP/2 multiplex |
| **On-disk scene cache** | First run writes every band window to `data/s2_cache/`; subsequent runs over the same scenes skip Planetary Computer entirely (~50× speedup) |

Typical wall times (laptop RTX, residential 100 Mbps connection):

| AOI | Chunks | Cold | Warm |
|---|---|---|---|
| 2 km × 2 km | 1 | 5–15 s | ~1 s |
| 22 km × 14 km | 12 | 30–60 s | ~3 s |
| 50 km × 35 km | 70 | 2–4 min | ~10 s |

PCA / Similarity / Few-shot share the same chunked encoder pass, so
they all benefit from the cache.

ForestLossDriver runs ~2× slower than single-stack FT heads because
each chunk fetches pre + post stacks separately (parallelised inside
the chunk).

**Sliding-window inference** (default ON for FT heads in the UI) adds
~50× more forward passes per chunk vs. one scene-level forward, so
wall time roughly doubles for single-stack heads and triples for
ForestLossDriver. The trade-off buys 16× finer spatial granularity
on classification outputs — without it, ForestLossDriver emits one
class for the entire 5 km chunk. Toggle off in the OlmoEarthImport
panel to recover the faster scene-level path when you only need a
single dominant class per chunk.

---

## Documentation

| Doc | What's in it |
|---|---|
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md) | Full feature matrix + design constraints + scientific honesty notes |
| [docs/FT-USAGE.md](docs/FT-USAGE.md) | Per-FT-head usage notes, recommended date ranges, calibration caveats |
| [docs/FOREST_LOSS_DRIVER.md](docs/FOREST_LOSS_DRIVER.md) | Pre/post pipeline shape, event-date contract, 10-class table |
| [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) | COG format, dequantization, 4 downstream-analysis recipes (similarity / few-shot / change detection / PCA) |
| [docs/FEW_SHOT.md](docs/FEW_SHOT.md) | UI walkthrough + API + comparison vs FT vs Similarity |
| [docs/GEOJSON_EXPORT.md](docs/GEOJSON_EXPORT.md) | Endpoint usage, polygon shape, supported heads, consumer recipes (Google Earth, QGIS, geopandas, leaflet) |
| [docs/LLM-CONVERSATIONS.md](docs/LLM-CONVERSATIONS.md) | Multi-round prompts that drive end-to-end OlmoEarth + tools workflows on every supported chat backend |
| [docs/llm-setup.md](docs/llm-setup.md) | Ollama vs vLLM trade-offs, GPU sizing matrix, supported Gemma variants |
| [docs/gemma-vllm.md](docs/gemma-vllm.md) | Docker recipe for the in-app `Start LLM` button |
| [docs/RESEARCH-UX-IMPROVEMENTS.md](docs/RESEARCH-UX-IMPROVEMENTS.md) | Prioritized backlog of research-workflow gaps |

---

## Project layout

```
Roger-Studio/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + CORS + router includes
│   │   ├── routers/                 # 14 routers, 54 endpoints
│   │   ├── services/                # inference orchestration + S2 fetch + LLM clients
│   │   ├── models/                  # pydantic schemas
│   │   └── mcp_server.py            # standalone MCP wrapper for external agents
│   ├── tests/                       # 223 pytest tests (offline) + 12 marked @network
│   └── run.py                       # uvicorn entry
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # top-level state + route to Sidebar / MapView / SplitMap
│   │   ├── components/              # 30+ React components
│   │   ├── api/client.ts            # typed fetch wrappers for every backend endpoint
│   │   └── types/                   # shared TS types
│   └── vite.config.ts               # dev proxy to backend:8000
├── docs/                            # 10 task-specific guides (this README links to all)
├── data/                            # gitignored — s2_cache/ + sample artifacts
└── README.md                        # you are here
```

---

## Testing

```bash
cd backend
pytest -m "not network"               # 223 offline tests, ~15 s
pytest -m network                     # 12 network tests (PC STAC, HF) — gated
pytest backend/tests/test_olmoearth_inference.py -v   # one suite verbose
```

Every test that mutates global model cache state uses fixtures to
reset; tests are safe to run in parallel via `pytest-xdist` if you
need the speed.

Frontend type-check:

```bash
cd frontend
pnpm exec tsc --noEmit                # zero errors on every changed file
```

---

## Contributing

This is research-grade code, not a product. PRs welcome — especially
for:

* New OlmoEarth fine-tuned heads (drop a `class_names_for(repo_id, …)`
  entry + a published colour palette and the loader picks them up
  automatically)
* Additional geo-tools (the cross-reference between FT outputs and
  vector data is genuinely under-served)
* Cloud deployment recipes (Azure East US is the natural co-location
  for Planetary Computer; AWS us-east-1 works too)
* Accessibility + i18n
* Tests for any path you change

For prioritized larger items see
[`docs/RESEARCH-UX-IMPROVEMENTS.md`](docs/RESEARCH-UX-IMPROVEMENTS.md).

### Discipline

A small set of conventions that have already prevented several
classes of bug — please honour them in PRs:

1. **Verify end-to-end before declaring done.** DOM snapshots are not
   sufficient. Drive features through preview, watch the worker,
   confirm the output.
2. **Catch specific exceptions.** A broad `except OSError` once
   silently swallowed a chunk-pipeline crash for an entire week.
3. **Per-chunk RAM gate, not just submit-time.** Free RAM may have
   dropped since the precheck — `chunk_ram_ok()` exists for a reason.
4. **No emoji in source files unless the user asked.** Keep code +
   docs ASCII-clean except where Markdown rendering benefits.
5. **Honest legends.** Calibration matters; uncalibrated softmax is
   not "probability". Surface the distinction in any new legend copy.

---

## License

[MIT](LICENSE) — see `LICENSE`.

## Credits

* **[OlmoEarth](https://huggingface.co/allenai)** — fine-tuned weights
  and base encoders from Allen AI
* **[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)**
  — Sentinel-2 L2A and ESA WorldCover data
* **[Google DeepMind Gemma family](https://ai.google.dev/gemma)** — local
  LLM backbone, any instruction-tuned size from 2B to 27B works
  (see the [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
  for architecture + benchmarks; we don't pin a specific variant — the
  model selector in the LLM Settings pane defers to whatever Ollama / vLLM
  is currently serving)
* **[MapLibre GL JS](https://maplibre.org/)**, **[terra-draw](https://terradraw.io/)**,
  **[rasterio](https://rasterio.readthedocs.io/)**,
  **[FastAPI](https://fastapi.tiangolo.com/)** — the foundations this
  studio is built on
* **AllenAI olmoearth_projects** — published class-name registries for
  every FT head

<div align="center">

---

Made with care by [Ziming Qi](https://github.com/2imi9) at Northeastern University.<br/>
OlmoEarth partner · Sentinel-2 L2A · 223 tests passing · MIT.

</div>
