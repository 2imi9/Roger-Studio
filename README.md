# Roger Studio

An Earth Observation copilot for remote-sensing research. Draw a bounding
box on a map, fetch Sentinel-2 imagery, run fine-tuned Earth models
(mangrove extent, land cover, forest loss, ecosystem type, live fuel
moisture), and chat with a tool-augmented LLM agent to drive the
analysis, all from a single web workbench.

Built on top of [AllenAI OlmoEarth](https://huggingface.co/allenai)
fine-tuned heads and the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
STAC catalog. Five LLM providers share 11 geo-tools so the same prompt
can drive analysis through Local Gemma (via Ollama or vLLM), NVIDIA NIM,
Claude, Gemini, or ChatGPT.

## Why it exists

Remote-sensing ML usually demands PhD skills and a GPU cluster.
Researchers, NGOs, and conservation groups who need to monitor mangroves,
deforestation, or seasonal drought often lack that stack. Roger Studio
closes the gap: real PyTorch inference and a tool-driven agent run on a
laptop RTX card with Gemma 4, or on free cloud API tiers if you prefer.

## Screenshots

See `docs/` for walkthroughs. The main surfaces:

- **Map tab**: draw polygons, load sample rasters, import GeoJSON / GeoTIFF
- **OlmoEarth tab**: Load → Run any base encoder or fine-tuned head
- **Analysis tab**: class breakdowns, download-GeoTIFF, raster summaries
- **3D Globe tab**: CesiumJS view of your selection
- **LLM tab**: 5 chat providers, tool-aware, copy-paste example prompts

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind v4, MapLibre GL, CesiumJS, terra-draw |
| Backend | Python 3.11, FastAPI, uvicorn, SQLite |
| Models | OlmoEarth base + FT heads (PyTorch), Gemma 4 via vLLM / Ollama |
| Data | Sentinel-2 L2A + ESA WorldCover via Microsoft Planetary Computer STAC |
| LLM routing | Shared `geo_tools` registry, 11 tools, OpenAI-shape schemas |

## Quick start

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

# Copy env template and fill in the providers you want to use
cp .env.example .env
# Minimum for local-only Gemma: leave .env as-is and start a local Gemma
# via Ollama (`ollama pull gemma-4-e4b-it`) or vLLM — see docs/llm-setup.md.
# For cloud providers: drop in your ANTHROPIC_API_KEY / NVIDIA_API_KEY /
# GEMINI_API_KEY / OPENAI_API_KEY as needed.

python run.py   # serves http://localhost:8000
```

### 3. Frontend

```bash
cd frontend
pnpm install    # or: npm install
pnpm dev        # serves http://localhost:3000
```

### 4. Open http://localhost:3000

The **Introduction Doc** button next to the Project menu launches a
10-step guided tour of every tab.

## Using Gemma 4 locally

Roger ships with `unsloth/gemma-4-e4b-it` (~8 GB, ungated) as the
default Local Chat backend. Two supported runtimes:

- **Ollama** (simpler, Windows-native):
  `ollama pull gemma-4-e4b-it && ollama serve` then set
  `LLM_RUNTIME=ollama` in `backend/.env`.
- **vLLM** (faster, better throughput, Linux / WSL / Docker):
  see `docs/gemma-vllm.md` for the Docker build the backend will
  auto-launch if you click "Start LLM" in the LLM Settings pane.

Larger variants (26B A4B, 31B) are supported via the model picker but
need a stronger GPU. See `docs/llm-setup.md` for the full matrix.

## Key features

- **Real PyTorch inference** on Sentinel-2 L2A composites. No stubs or
  mock rasters in the happy path.
- **Fine-tuned model dispatch** (`olmoearth_ft.py`) for Mangrove, AWF,
  ForestLossDriver, EcosystemTypeMapping, LFMC, with tentative-or-confirmed
  class-name metadata.
- **Sliding-window tiled inference** for spatial class maps on large bboxes.
- **Auto-retry on stub fallback** — when an inference stubs, the backend
  suggests concrete retry params (`suggested_retries`) and applies the
  first one once before surfacing the failure to the user.
- **Projects persistence** — named session bundles (bbox, layers, labels,
  chat history) stored in SQLite, recoverable across browser reloads and
  machine restarts. OlmoEarth-Studio-shaped API at `/api/v1/projects`.
- **Guided tour** (Shepherd-style, no dep) for onboarding.
- **Compare mode** — side-by-side A/B raster inspection via
  `@maplibre/maplibre-gl-compare`.
- **MCP server** — optional `fastmcp` wrapper around the geo-tools for
  external agents like Claude Desktop. See `backend/app/mcp_server.py`.

## Performance and limitations

Roger Studio targets **native 10 m/pixel inference** — the same resolution
the OlmoEarth FT heads were trained on — rather than the legacy
downsample-to-256-px approach. Reaching that quality means fighting
network latency, so the pipeline is built around three layers of
optimization:

1. **Chunked AOI inference.** The AOI is sliced into 5 km × 5 km tiles;
   each tile fetches + infers independently and outputs are stitched into
   a single pixel-aligned global raster. One STAC search per 30-day
   period covers every chunk (S2 scenes are ~110 km wide), and per-chunk
   windowed reads pull only the bytes that tile needs.
2. **Parallel fetching at two levels.** Chunks run 4-wide via an
   asyncio semaphore; within each chunk, all 6 periods × 12 bands
   (72 reads) fire concurrently via `asyncio.gather` + `asyncio.to_thread`.
   GDAL is tuned for HTTP/2 multiplex and 10 MB read chunks to match
   Planetary Computer's COG layout.
3. **On-disk scene cache.** The first run over a bbox writes every
   fetched band window to `data/s2_cache/` as `.npy`. Every subsequent
   inference over the same bbox (different FT head, different date — as
   long as the scenes overlap) skips PC entirely and runs ~50× faster.
   Cache is keyed on `(scene_id, bbox, gsd, band)`; S2 scenes on PC are
   immutable so the cache is valid indefinitely. Set
   `S2_CACHE_DISABLED=1` to force re-fetch, or delete the cache dir to
   clear it.

### Typical timings

| AOI size | Chunks | First run | Cached run |
|---|---|---|---|
| 2 km × 2 km | 1 | ~5–15 s | ~1 s |
| 22 km × 14 km | 12 | ~30–60 s | ~3 s |
| 50 km × 35 km | 70 | ~2–4 min | ~10 s |

The spread in first-run times depends almost entirely on your internet
connection to Microsoft's US East region. On home connections (50–200
Mbps) the pipeline is **network-bound**: your GPU is idle ~95 % of the
time during a first fetch. The real production answer for instant
interactivity is deploying the backend inside Azure East US so the
backend and PC's COGs share a ~10 Gbps intra-datacenter link.

### Known limitations

**Fine-tuned head training regions.** The published OlmoEarth FT heads
were each trained on a specific geographic slice. Running them outside
that slice produces confident-looking but scientifically meaningless
output (the ecosystem head will happily classify New Jersey pine forest
as "tropical rainforest" because its training set was north Africa
only). Current head coverage:

| Head | Training region |
|---|---|
| `EcosystemTypeMapping-Base` | north Africa only |
| `AWF-Base` | southern Kenya only |
| `Mangrove-Base` | global tropical coastal belt |
| `LFMC-Base` | fire-prone regions (California, Mediterranean, Australia) |
| `ForestLossDriver-Base` | pantropical |

For AOIs outside these regions the **base encoder** (`OlmoEarth-v1-Base`,
-Nano, -Tiny, -Large) still produces useful unsupervised embeddings
visualized via PCA. Classification heads need to be fine-tuned on
in-region labels — not shipped in this repo.

**LFMC and ForestLossDriver need extra modalities.** LFMC was trained on
Sentinel-1 + Sentinel-2 multi-modal input; ForestLossDriver needs a
pre/post Sentinel-2 pair with a `CONTAINS` space mode. Roger Studio's
S2-only fetch path does not yet support either, so both heads
silently fall back to a single-scene S2 path — you get output, but it's
known off-distribution. The dispatcher logs a warning
(`falling back to legacy S2-only single-scene path`) on every run of
these heads. Fixing this requires a new S1 fetcher + the pre/post
grouping logic in `fetch_s2_temporal_stack`, not in this release.

**Home-internet latency bounds everything.** Chunked + parallel +
cached together cut first-run times by ~5× vs the naive path, but
moving 600 MB of Sentinel-2 bytes from Microsoft to a residential ISP
still takes ~1 minute minimum. The only way below that is local
caching (done) or colocated hosting (future).

**Hallucinated super-resolution is a trap.** Using a diffusion model
(cBottle, CorrDiff, similar) to upsample cheap low-res fetches would
sound like a speedup but would destroy scientific validity — the
downstream classifier would see plausible-but-fake pixels and emit
confident nonsense, without any flag telling the user. Roger Studio
deliberately does not go this route.

## Custom embedding exports

When no fine-tuned head covers your region, compute **OlmoEarth
embeddings** instead and run lightweight downstream analysis on your own
labels. The **Export embeddings as COG** button in the OlmoEarth Import
panel (visible for base encoders) produces a multi-band int8 GeoTIFF
bit-for-bit compatible with Ai2 OlmoEarth Studio's published format.
Use the exported file for:

- **Similarity search** — "where else looks like this?"
- **Few-shot segmentation** — ~60 labels + sklearn = wall-to-wall map
- **Change detection** — diff two date ranges
- **PCA false-color** — unsupervised exploration

See [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md) for copy-paste-ready
Python recipes for all four workflows, including AlphaEarth-compatible
int8 dequantization via
`olmoearth_pretrain.evals.embedding_transforms.dequantize_embeddings`.

## Running the tests

```bash
cd backend
pytest                      # 74 offline tests
pytest -m network           # 11 network tests (PC STAC, HF, etc.)
```

## Contributing

This is research-grade code, not a product. PRs welcome, especially for:

- New fine-tuned OlmoEarth heads
- Additional geo-tools (pixel-wise masked stats, for example, would
  close the "mangrove ∧ low LFMC" cross-reference gap)
- Deployment recipes for cloud GPU hosts
- Accessibility + i18n

See `docs/RESEARCH-UX-IMPROVEMENTS.md` for a prioritized backlog of
research-workflow gaps.

## License

MIT — see `LICENSE`.

## Credits

- **OlmoEarth** — fine-tuned weights and base encoder from AllenAI
  ([allenai/OlmoEarth-v1](https://huggingface.co/allenai))
- **Microsoft Planetary Computer** — Sentinel-2 L2A and WorldCover data
- **Google DeepMind Gemma 4** — local LLM backbone
- **AllenAI olmoearth_projects** — published class-name registries for
  the FT heads
