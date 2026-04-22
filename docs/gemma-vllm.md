# Gemma 4 31B NVFP4 via vLLM — Launch Guide

The "Validate with Gemma" button in Cuvier Studio calls a vLLM server on
port 8001. This doc is how you start it and verify it's reachable.

## Hardware

- **RTX 5090 (Blackwell, 32GB, sm_120)** — target hardware.
- **Model VRAM footprint**: ~16GB (31B weights in NVFP4) + ~2GB KV cache at
  8192-context → ~18GB. Leaves ~12GB for TIPSv2, SamGeo (unload after use), and
  the rest of the pipeline.

### Blackwell-specific dependency pinning (critical)

Running vLLM on RTX 5090 (sm_120) has three gotchas that the stock
`vllm/vllm-openai:latest` or `:nightly` images don't handle:

1. **sm_120 kernels**: stable PyTorch and stock vLLM images are compiled for
   sm_90 and below → "CUDA capability sm_120 is not compatible". Fix: use the
   **`vllm/vllm-openai:cu130-nightly`** tag which ships kernels for SM 12.0.
   Reference: [vllm #37242](https://github.com/vllm-project/vllm/issues/37242).
2. **NVFP4 kernels** for Blackwell workstation GPUs require vLLM ≥ 0.13.0.
   Reference: [vllm #31085](https://github.com/vllm-project/vllm/issues/31085).
3. **Attention backend**. **FA4 does NOT run on RTX 5090**, despite FA4 being
   advertised as "Blackwell-optimized". That applies only to SM100 datacenter
   Blackwell (B200). RTX 5090 and RTX PRO 6000 are SM120 and lack the TMEM
   subsystem FA4 requires. FA3 is also broken on SM120.
   **Use FlashInfer as primary, FA2 as fallback**:
   - `VLLM_ATTENTION_BACKEND=FLASHINFER` (~8% faster than FA2 on SM120;
     default for NVFP4 MoE since vLLM 0.19)
   - `VLLM_FLASH_ATTN_VERSION=2` (fallback if FlashInfer dispatch fails)
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (prevents fragmentation)
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn` (Blackwell community config default)

   Refs: [vllm #36865](https://github.com/vllm-project/vllm/issues/36865),
   [vllm #38971 NVFP4 MoE SM120 backend](https://github.com/vllm-project/vllm/issues/38971),
   [vllm #37242 5090 community config](https://github.com/vllm-project/vllm/issues/37242),
   [Hardware Corner on RTX PRO 6000 / FA4](https://www.hardware-corner.net/rtx-pro-6000-blackwell-flashattention-4/),
   [vLLM forum: FA4 on B200](https://discuss.vllm.ai/t/how-to-apply-fa4-on-b200/2133).

The `backend/docker/Dockerfile.vllm` in this repo bakes #1 and #2 into an
image tagged `geoenv-vllm:latest`, and the Start button in the LLM panel
passes #3 as an env var.

### Gemma 4 specific

Even with the right Blackwell image, stock vLLM didn't know about the
`gemma4` architecture until Transformers v5.5.0. The Dockerfile upgrades
Transformers so the model config parses. The NVIDIA DGX Spark forum confirms
the proven flag set:

```
--tool-call-parser gemma4 --reasoning-parser gemma4
--load-format fastsafetensors --kv-cache-dtype fp8
```

Reference: <https://forums.developer.nvidia.com/t/how-to-run-gemma-4-nvfp4-in-vllm-docker>

### Performance warning

The 31B NVFP4 dense model runs at **~6.7 tokens/sec** on DGX Spark (GB10)
per the forum thread above. On RTX 5090 the NVFP4 tensor cores give a
speedup, but don't expect > 20 tok/s. If throughput matters, consider the
**26B A4B MoE** variant (`nvidia/Gemma-4-26B-A4B-IT-NVFP4` once released, or
the RedHatAI mirror) — same quality class, 4B active params, much faster.

## Prerequisites

```bash
# Python 3.11+ virtualenv
pip install "vllm>=0.7.0" modelopt
# Blackwell needs CUDA 12.8+ and a recent PyTorch wheel; check vllm docs.

# Gate the model on HF (Gemma license acceptance required once per account)
huggingface-cli login
```

## Launch (single RTX 5090)

```bash
vllm serve nvidia/Gemma-4-31B-IT-NVFP4 \
    --port 8001 \
    --dtype auto \
    --quantization modelopt \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=4" \
    --gpu-memory-utilization 0.65 \
    --enable-prefix-caching
```

- `--tensor-parallel-size 1` — single-GPU. NVIDIA's reference `tp=8` is for H100
  clusters.
- `--max-model-len 32768` — model supports **256K native** but 32K covers every
  realistic polygon batch and halves KV cache VRAM vs full context.
- `--quantization modelopt` — load NVFP4 weights via NVIDIA ModelOpt.
- `--limit-mm-per-prompt image=4` — matches `gemma_client.chat_with_vision`.
- `--gpu-memory-utilization 0.65` — leaves VRAM headroom for TIPSv2 / SamGeo.
  Drop to 0.8 if you unload SAM after use.

First launch downloads ~16GB from HF. Subsequent launches cache locally.

### Visual token budget

Gemma 4 accepts configurable image-token budgets: **70, 140, 280, 560, 1120**.
Default in vLLM is 256 per image. For land-cover patches, 280–560 is the sweet
spot — enough detail to see roads and textures, without bloating prompt tokens.
If you're validating thousands of polygons, drop to 140 to 3× throughput.

### Video support (future)

Gemma 4 31B natively handles **video up to 60s at 1fps**. This opens the door to
a "temporal validator": feed a monthly Landsat/Sentinel animation over a
polygon and let Gemma catch seasonal mismatches (e.g. a polygon labeled "forest"
that goes brown in July → actually a pine beetle kill zone). Tracked as
roadmap item in `roadmap.md`.

## Verify

```bash
# From the backend host:
curl http://localhost:8001/v1/models

# Or through the FastAPI backend:
curl http://localhost:8000/auto-label/gemma/health
```

Frontend: the "Validate with Gemma" button will be disabled until
`/auto-label/gemma/health` returns `{"reachable": true}`.

## Environment overrides

Set any of these to point the backend at a different server/model:

| Env var           | Default                              |
|-------------------|--------------------------------------|
| `GEMMA_BASE_URL`  | `http://localhost:8001/v1`           |
| `GEMMA_MODEL`     | `nvidia/Gemma-4-31B-IT-NVFP4`        |
| `GEMMA_TIMEOUT`   | `120` seconds                        |

## Fallback: smaller model for dev

If 31B is too heavy while iterating, swap in E4B (stays under 4GB VRAM):

```bash
vllm serve prithivMLmods/gemma-4-E4B-it-NVFP4 --port 8001 --quantization modelopt
export GEMMA_MODEL=prithivMLmods/gemma-4-E4B-it-NVFP4
```

Reasoning quality drops noticeably on hard cases, but the validation chain
still works end-to-end for wiring verification.
