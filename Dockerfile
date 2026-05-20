# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  x-socials Moderation Service — Dockerfile                              ║
# ║                                                                          ║
# ║  Multi-stage build:                                                      ║
# ║    builder  — installs Python deps (including PyTorch for detoxify)      ║
# ║    runtime  — slim image with only what's needed to run                  ║
# ║                                                                          ║
# ║  Model weights (detoxify / HuggingFace) are downloaded at runtime on    ║
# ║  first request and cached in the volume mounted at /root/.cache.         ║
# ║  Pre-warm them at build time by uncommenting the RUN line below.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (aiomysql, motor, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY pytest.ini .

# HuggingFace model cache lives here — mount as a named volume in production
# so weights survive container restarts without re-downloading.
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Optional: pre-warm the detoxify model at build time (adds ~800 MB to image).
# Uncomment if you prefer a larger image over a slow cold start.
# RUN python -c "from detoxify import Detoxify; Detoxify('original')"

# FastAPI port
EXPOSE 8001

# Healthcheck — polls /health every 30 s, fails after 3 missed checks
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" \
    || exit 1

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
