# Roger Studio MCP Server

A Model Context Protocol (MCP) server that exposes Roger Studio's
OlmoEarth toolbox to any MCP-compatible LLM client: Claude Desktop,
Cursor, Claude Code, OpenAI Responses API, NVIDIA NIM tool-calling,
custom agents, etc.

## What it exposes

Ten tools covering the full Earth-observation workflow:

| Tool | Purpose |
|---|---|
| `list_olmoearth_ft_heads` | Curated FT heads + cache status |
| `get_olmoearth_catalog` | Live HuggingFace catalog (10-min TTL) |
| `get_loaded_olmoearth_models` | In-memory model cache snapshot |
| `run_olmoearth_inference` | Run inference on a bbox, return tile URL + metadata |
| `get_olmoearth_demo_pairs` | Curated A/B compare presets |
| `explain_raster` | LLM-backed plain-language explanation of a raster |
| `search_sentinel2_imagery` | STAC search over Planetary Computer |
| `get_sentinel2_composite_tile_url` | Least-cloudy mosaic tile URL |
| `classify_land_cover` | ESA WorldCover class percentages |
| `polygon_stats` | Perimeter, area, elevation for a GeoJSON polygon |
| `load_olmoearth_repo` | Pre-warm a HuggingFace repo into local cache |

## Architecture

The MCP server is a **thin client** over the FastAPI backend. Tools
call the existing `http://localhost:8000/api/...` routes via `httpx`,
so:

- **Single source of truth** — inference logic lives once, in FastAPI.
- **Shared state** — model cache, in-flight jobs, session data are
  shared between the browser UI and any MCP client. Click a tile in
  Claude Desktop; it's already warm in the Roger Studio browser tab.
- **Credentials stay server-side** — `NVIDIA_API_KEY`,
  `ANTHROPIC_API_KEY`, `HF_TOKEN` live in `backend/.env`. MCP clients
  never see them.

## Run locally

Prerequisites: the FastAPI backend must be running.

```bash
cd geoenv-studio/backend
uvicorn app.main:app --port 8000          # in one terminal
python -m app.mcp_server                  # in another (stdio server)
```

The second command launches an MCP stdio server that your client will
normally spawn for you — see the next section.

## Register with Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or
`~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) and add:

```json
{
  "mcpServers": {
    "roger-studio": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "C:/Users/Frank/OneDrive/Desktop/Github/geoenv-studio/backend",
      "env": {
        "ROGER_API_BASE": "http://localhost:8000"
      }
    }
  }
}
```

Restart Claude Desktop. The tools appear under the hammer icon in the
chat composer. Ask Claude "run mangrove inference on the Niger Delta"
and it'll call `run_olmoearth_inference` → `explain_raster` end-to-end.

## Register with Cursor

Cursor reads the same format via Settings → MCP. Use the same JSON
snippet above.

## Register with Claude Code

```bash
claude mcp add roger-studio \
  --command python \
  --args '-m,app.mcp_server' \
  --cwd C:/Users/Frank/OneDrive/Desktop/Github/geoenv-studio/backend
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ROGER_API_BASE` | `http://localhost:8000` | FastAPI backend URL. Set this to a remote instance when you're running the MCP server locally but Roger Studio is deployed. |
| `ROGER_MCP_TIMEOUT_S` | `300` | Per-tool HTTP timeout in seconds. Raise for very large inference AOIs. |

## Debugging

Use the MCP inspector to call tools interactively:

```bash
npx @modelcontextprotocol/inspector python -m app.mcp_server
```

Logs go to stderr (stdout is reserved for the JSON-RPC protocol). When
a tool returns `{error: ..., hint: ...}`, the `hint` field tells you
how to fix it — common cause is the FastAPI backend not running.

## Adding a tool

1. Pick a FastAPI route you want to expose.
2. In `mcp_server.py`, add an `@mcp.tool()`-decorated async function
   whose signature becomes the tool's input schema (FastMCP autogenerates
   JSON schema from the type annotations).
3. Call `_api("POST"/"GET", "/api/...", json=..., params=...)` — returns
   the route's JSON response or a structured `{error, hint}` dict on
   failure.
4. Docstring becomes the tool description the LLM sees; keep it sharp.
