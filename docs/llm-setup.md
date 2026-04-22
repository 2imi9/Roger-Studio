# LLM Setup Guide — Cuvier Studio

The LLM tab in Cuvier Studio speaks the **OpenAI-compatible chat
completions protocol**, which means you can point it at:

- a local vLLM Docker container on your GPU, **or**
- any cloud inference provider that offers an OpenAI-compatible endpoint
  (OpenAI, Google Gemini, Anthropic via compat shim, NVIDIA NIM, OpenRouter,
  Groq, Together, Fireworks, etc.), **or**
- an MCP-enabled LLM server once the connector path is wired (see §6).

Pick whichever matches your constraints. You can mix — e.g. use Gemma 4 on
NIM while iterating, then flip a single env var to local Docker for
privacy-sensitive runs.

---

## 1. TL;DR — Which option should I pick?

| Priority | Recommended | Why |
|---|---|---|
| Fastest to first chat (no setup) | **OpenAI** or **Anthropic** | One API key, one env var, works in 30 s |
| **Best local option on RTX 5090** | **Ollama + `gemma4:26b`** | Host-native on :11434, auto-GPU (CUDA), MoE (3.8B active / 25.2B total), 256K context |
| Want the Gemma 4 flagship without buying one | **Google AI Studio** | `google/gemma-4-31b-it` served free up to a daily quota |
| Maximum model variety, one key | **OpenRouter** | Proxies 100+ models incl. Gemma 4, Claude 4.6, GPT-5 |
| Data privacy (nothing leaves your box) | **Ollama** (default) | Fully local, llama.cpp under the hood, no outbound traffic |
| Lowest latency per token | **Groq** or **Cerebras** | Hardware accelerators, ~500 t/s |
| You already pay for Claude | **Anthropic** | Claude Opus 4.6 / Sonnet 4.6 work well for geo reasoning |

> **Why Ollama instead of Docker Model Runner?**
> Docker Model Runner's OpenAI endpoint on port 12434 requires a "host-side TCP" toggle that's missing on some Docker Desktop versions, and its default llama.cpp backend is CPU-only (needs a separate GPU-variant install). Ollama listens on `localhost:11434` by default, auto-detects your RTX 5090, and takes zero configuration. Same llama.cpp under the hood, same GGUF weights, just a simpler wrapper. Four Gemma 4 tags cover every VRAM budget:
> - `gemma4:e2b` (~2 GB) — edge-size, audio + image + text
> - `gemma4:e4b` (~5 GB) — small laptop-size, audio + image + text, 128K context
> - `gemma4:26b` (~15-18 GB) — **current default**, MoE (3.8B active), image + text, 256K context
> - `gemma4:31b` (~20-24 GB) — flagship dense, best quality, 256K context

---

## 2. Configuration model

The backend reads four env vars (or values from `.env` in the backend dir):

```bash
LLM_RUNTIME=ollama|vllm|cloud    # which lifecycle the backend drives
GEMMA_BASE_URL=http://<host>/v1  # OpenAI-compatible endpoint
GEMMA_MODEL=<provider-model-id>  # model tag the endpoint serves
GEMMA_API_KEY=<your-api-key>     # optional for local, required for cloud
```

### What `LLM_RUNTIME` controls

The value decides how **Start / Stop buttons and status messages behave**.
It does **not** affect chat or validation — those always speak OpenAI
protocol to `GEMMA_BASE_URL`.

| `LLM_RUNTIME` | Start button | Status check | Recommended when |
|---|---|---|---|
| `ollama` (default) | Runs `ollama pull` + auto-loads on first request | `ollama list` / `ollama ps` | Easiest local path, Windows-native |
| `vllm` | Hidden — vLLM is user-managed | Endpoint reachability only | You run `docker run vllm/vllm-openai:nightly ...` in WSL |
| `cloud` | Hidden — nothing to start | Just hits `/v1/models` at the remote | OpenRouter, OpenAI, Anthropic, NIM, Google AI Studio, etc. |

### Auto-detection if `LLM_RUNTIME` is unset

- Non-localhost URL → `cloud`
- Port 11434 → `ollama`
- Anything else localhost → `vllm`

### Variable naming

The `GEMMA_` prefix is kept for backward compatibility with earlier versions
of the backend. Any OpenAI-compatible endpoint works regardless of which
model family it serves.

Restart the backend after changing these. No rebuild needed.

---

## 3. Cloud provider recipes

### 3.1 OpenAI

```bash
export GEMMA_BASE_URL=https://api.openai.com/v1
export GEMMA_MODEL=gpt-4o
export GEMMA_API_KEY=sk-proj-...
```

Models: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, `gpt-5` (if available).

### 3.2 Anthropic Claude

Anthropic has an OpenAI-compatible endpoint since early 2026.

```bash
export GEMMA_BASE_URL=https://api.anthropic.com/v1
export GEMMA_MODEL=claude-opus-4-6
export GEMMA_API_KEY=sk-ant-...
```

Models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.

### 3.3 Google Gemini / AI Studio (for Gemma 4)

Google AI Studio serves Gemma 4 directly via an OpenAI-compatible shim:

```bash
export GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export GEMMA_MODEL=google/gemma-4-31b-it
export GEMMA_API_KEY=<ai-studio-key>
```

This is the **easiest way to run the actual Gemma 4 31B** without local
hardware — same reasoning quality, no NVFP4 gotchas.

### 3.4 NVIDIA NIM (Gemma 4 NVFP4 on NVIDIA Cloud)

NIM hosts `nvidia/Gemma-4-31B-IT-NVFP4` as a managed endpoint.

```bash
export GEMMA_BASE_URL=https://integrate.api.nvidia.com/v1
export GEMMA_MODEL=nvidia/gemma-4-31b-it-nvfp4
export GEMMA_API_KEY=nvapi-...
```

Get a key at <https://build.nvidia.com>. Free tier available.

### 3.5 OpenRouter (aggregator — one key, any model)

```bash
export GEMMA_BASE_URL=https://openrouter.ai/api/v1
export GEMMA_MODEL=google/gemma-4-31b-it
export GEMMA_API_KEY=sk-or-...
```

Lets you flip between `anthropic/claude-opus-4-6`,
`google/gemma-4-31b-it`, `openai/gpt-5`, etc., with the same code path.

### 3.6 Groq (for speed)

```bash
export GEMMA_BASE_URL=https://api.groq.com/openai/v1
export GEMMA_MODEL=gemma2-9b-it   # Groq hasn't shipped Gemma 4 yet
export GEMMA_API_KEY=gsk_...
```

### 3.7 Together / Fireworks / Cerebras

All use OpenAI-compatible endpoints. Substitute their docs URLs; same
three-var pattern.

---

## 4. Local — Gemma 4 via Ollama (default)

Google DeepMind's Gemma 4 family, served by Ollama. This is the project's
default local path because it works out of the box on Windows with an NVIDIA
GPU — no Docker Desktop knobs, no kernel patches, no CDN flakes.

### 4.1 Install + pull (two commands)

```bash
# 1. Install Ollama (Windows installer registers a background service)
#    Download from https://ollama.com/download

# 2. Pull Gemma 4 26B (the project default)
ollama pull gemma4:26b
```

Ollama listens on `http://localhost:11434` by default. The OpenAI-compatible
endpoint is at `http://localhost:11434/v1/chat/completions`. Ollama
auto-loads a model into VRAM on first request and auto-unloads after an
idle timeout — no separate serve step.

### 4.2 Backend config (default — nothing to set)

```bash
# These are the built-in defaults — no .env edit needed.
GEMMA_BASE_URL=http://localhost:11434/v1
GEMMA_MODEL=gemma4:26b
# leave GEMMA_API_KEY empty for local
```

Verify with `ollama list` (pulled models) and `ollama ps` (loaded into VRAM).

### 4.3 Which tag for which use case

| Tag | Size | Params | Context | Modalities | When to pick |
|---|---|---|---|---|---|
| `gemma4:e2b` | ~2 GB | 2.3B effective | 128K | text + image + audio | Lightest fit; edge devices, CI runners |
| `gemma4:e4b` | ~5 GB | 4.5B effective | 128K | text + image + audio | Laptop-size, if 26b is too heavy |
| **`gemma4:26b`** | **~15-18 GB** | **25.2B total / 3.8B active (MoE)** | **256K** | **text + image** | **Default — best speed/quality on 5090** |
| `gemma4:31b` | ~20-24 GB | 30.7B dense | 256K | text + image | Flagship quality, tightest VRAM fit |

Change `GEMMA_MODEL` in `backend/.env`, then `ollama pull <tag>`.

### 4.4 Gemma 4 sampling parameters (from the official model card)

Standard config for all tasks:

```python
temperature=1.0
top_p=0.95
top_k=64
```

Our validator uses `temperature=0.2` (for deterministic classification) and
our chat endpoint uses `temperature=0.4` — both lower than Gemma's default
because we value structured, stable output over creativity. If you prefer
richer narration in the chat tab, bump the chat `temperature` up to 1.0.

### 4.5 Thinking mode

Gemma 4 supports a **built-in thinking mode** — the model emits an internal
reasoning block before its final answer. Toggle by prepending `<|think|>`
to the system prompt:

- Thinking ON (richer reasoning, more tokens, slower):
  ```
  system = "<|think|>You are a remote-sensing geoscientist..."
  ```
- Thinking OFF (default, fastest):
  ```
  system = "You are a remote-sensing geoscientist..."
  ```

Libraries like llama.cpp (what Docker Model Runner uses under the hood)
strip the internal reasoning from the history automatically — you just see
the final answer. See the **Multi-Turn Conversations** note in the Gemma 4
model card for why thinking content must *not* be carried across turns.

### 4.6 Visual token budget

Gemma 4 accepts a configurable image-token budget: **70, 140, 280, 560, 1120**.

| Budget | Use for |
|---|---|
| 70–140 | Classification, captioning, video-frame batches (speed-critical) |
| 280 | Most polygon validations (our current default) |
| 560 | Detailed patch inspection, terrain feature finding |
| 1120 | OCR, document parsing, reading small text in maps |

---

## 5. Local vLLM Docker — Gemma 4 NVFP4 (RTX 5090 / Blackwell sm_120)

The LLM tab ships with a **Start** button that builds and launches a local
vLLM container. This is the fallback for when you need full-local
inference.

See [`gemma-vllm.md`](./gemma-vllm.md) for the full Blackwell-specific
walkthrough. Short version:

- Uses `vllm/vllm-openai:cu130-nightly` (the only image with SM120 kernels)
- Uses FlashInfer as attention backend (FA4 doesn't run on SM120)
- Runs at ~6.7 tok/s for 31B dense — **much slower than any cloud option**

When local is running, point the frontend at it:

```bash
export GEMMA_BASE_URL=http://localhost:8001/v1
export GEMMA_MODEL=nvidia/Gemma-4-31B-IT-NVFP4
# no API key needed for local
unset GEMMA_API_KEY
```

---

## 6. Which provider for which task?

| Task | Best provider | Reason |
|---|---|---|
| Polygon validation (hundreds in parallel) | OpenRouter → Gemma 4 or Gemini Flash | Cheap + parallel + multimodal |
| Deep geo reasoning on ambiguous tiles | Anthropic Claude Opus 4.6 | Best single-shot reasoning |
| Ultra-fast chat in the UI | Groq or Cerebras | Sub-second first token |
| Exactly-match forum benchmark (DGX Spark) | Local vLLM, 31B NVFP4 | Reproducibility |
| Working offline on a plane | Local vLLM | Only option |

---

## 7. Future: MCP connector discovery

Rather than wiring each provider by hand, we can use the MCP registry to
discover LLM-serving MCP servers at runtime. Planned flow:

1. The LLM tab shows a **Connectors** dropdown.
2. Clicking it calls `mcp-registry.search_mcp_registry(keywords=["llm", "chat"])`.
3. Returned MCP servers (e.g. Claude via MCP, Gemini via MCP, OpenRouter
   via MCP) are rendered as connectable cards.
4. User clicks **Connect** → MCP auth flow → the backend now has a
   registered MCP tool for chat completion.
5. The chat widget uses whichever connector was last selected.

This removes the need to juggle env vars and API keys per provider — MCP
handles auth, rate limits, and routing.

**Status**: MCP registry search is already available in the harness
(`mcp__mcp-registry__search_mcp_registry` and
`mcp__mcp-registry__suggest_connectors`). Wiring them into the backend as
a new `/llm/connectors` endpoint is scoped but not yet built.

---

## 8. Security notes

- `.env` files in `backend/` are gitignored (verify before committing).
- Never paste API keys into the chat itself. The chat is a user prompt and
  will be logged by the provider.
- For team use, rotate the OpenRouter or NIM key rather than giving every
  developer their own — centralized spend + audit logs.
- Local vLLM gives full data isolation: satellite patches, polygon coords,
  chat history, none of it leaves your machine.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Wrong or expired API key | Regenerate at provider dashboard |
| `model not found` | Wrong `GEMMA_MODEL` for provider | Check provider's model list |
| `Context length exceeded` | Trying to send a huge polygon batch | Lower `max_concurrent` in validate call |
| Chat works but validator errors on images | Provider doesn't support vision | Switch to Gemini / Claude / Gemma / GPT-4o |
| Local vLLM "No NvFp4 MoE backend" | SM120 kernel gap | See [`gemma-vllm.md`](./gemma-vllm.md) |
| Backend didn't pick up env var change | Uvicorn caches os.environ | Restart the backend process |

---

## 10. Minimum viable setup

### Fully local, no API key (recommended for this project)

```bash
# 1. Install Ollama from https://ollama.com/download
#    (Windows installer registers a background service on :11434)

# 2. Pull Gemma 4 26B (or just click Start LLM in the UI — it calls ollama pull)
ollama pull gemma4:26b

# 3. Nothing else to set — these are already the project defaults:
#    GEMMA_BASE_URL=http://localhost:11434/v1
#    GEMMA_MODEL=gemma4:26b
```

Open the LLM tab → click **▶ Start LLM** if the badge is offline. ~15-18 GB
pull on first run, ~15 GB VRAM used at Q4, multimodal (text + image),
256K context. Runs at ~4B speed because the MoE only activates 3.8B params
per token.

### Fully cloud, 5 minutes, one API key

```bash
# In backend/.env
GEMMA_BASE_URL=https://openrouter.ai/api/v1
GEMMA_MODEL=google/gemma-4-31b-it   # or anthropic/claude-opus-4-6, openai/gpt-5, etc.
GEMMA_API_KEY=sk-or-v1-<your-openrouter-key>
```

Restart backend. Open the LLM tab. Chat. Switch between 100+ models by
editing `GEMMA_MODEL`.
