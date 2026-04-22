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

## Running the tests

```bash
cd backend
pytest                      # 64 offline tests
pytest -m network           # 12 network tests (PC STAC, HF, etc.)
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
