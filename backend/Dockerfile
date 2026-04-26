# Roger Studio backend — FastAPI + OlmoEarth.
#
# CPU-only image by default — torch is installed from PyPI's default
# index, which resolves to the CPU build on Linux. Real OlmoEarth
# inference needs CUDA, so a CPU container will fall back to STUB
# results (the orchestrator already has the stub pathway wired). This
# is fine for UI development; the frontend renders the stub-mode
# ResultPanel correctly so design + state-machine work doesn't need a
# GPU.
#
# To run with GPU passthrough:
#   1. Install nvidia-container-toolkit on the host (Windows: WSL2 +
#      Docker Desktop in WSL2 mode + the toolkit inside WSL).
#   2. Replace the FROM line with `nvidia/cuda:12.8.0-runtime-ubuntu24.04`
#      and pip-install torch from the cu128 wheel index:
#          pip install torch --index-url https://download.pytorch.org/whl/cu128
#   3. Add `runtime: nvidia` (or the deploy-resources block) to the
#      api service in docker-compose.yml.
# Real-inference work is still better off running natively until the
# GPU image is verified end-to-end.

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# System libs required by rasterio (GDAL) + shapely (GEOS). Pinned to
# Debian Bookworm versions that match the Python wheels' expected ABI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal32 \
        libgeos-c1v5 \
        libproj25 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying app code so the dep layer caches
# across code-only edits. pyproject.toml + the editable install pull
# in the full dep tree without needing a separate requirements.txt.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# App code — copied last so changes don't bust the deps layer.
COPY . .

# Persistent data — bind-mounted from the host via docker-compose so
# the Sentinel-2 scene cache survives container restarts and warm
# inferences stay fast across `docker compose down/up` cycles.
VOLUME ["/app/data"]

EXPOSE 8000

# Recommended safety env for laptop GPUs / CPU-only fallback. Override
# at compose-time if you have a workstation with more headroom.
ENV OE_MAX_CONCURRENT_JOBS=1 \
    OE_MAX_CHUNKS=100 \
    PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
