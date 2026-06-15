# Base image: CUDA 12.8 + cuDNN developer headers on Ubuntu 24.04.
# Ubuntu 24.04 ships Python 3.12 natively; CUDA 12.8+ is required for
# Blackwell (sm_120) GPU support (RTX 5000 series).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during apt installs (e.g. tzdata)
ENV DEBIAN_FRONTEND=noninteractive

# Single shared virtual environment for all Python packages
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/root/.local/bin:/opt/venv/bin:$PATH"

# uv default timeout (30 s) is too short for large CUDA wheel downloads (>500 MB)
ENV UV_HTTP_TIMEOUT=300

# ── System packages ─────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0: runtime shared libraries required by opencv-python
# build-essential + python3.12-dev: needed to compile flash-attn-3 and cc_torch
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── uv (fast Python package manager) ────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── Virtual environment ──────────────────────────────────────────────────────
RUN uv venv $VIRTUAL_ENV --python 3.12

# ── PyTorch with CUDA 12.8 ──────────────────────────────────────────────────
# cu128 index includes sm_120 (Blackwell) kernels; first supported in torch 2.7+.
RUN uv pip install "torch>=2.7.0" torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ── SAM3 (Facebook Research, commit 86ed770) ────────────────────────────────
# Installed as an editable package so `from sam3.model_builder import ...` works.
# einops / ninja / pycocotools / psutil are SAM3 runtime requirements.
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && git -C /opt/sam3 checkout 86ed770 \
    && cd /opt/sam3 && uv pip install -e . \
    && uv pip install einops ninja pycocotools psutil

# ── Flash Attention 3 (optional) ────────────────────────────────────────────
# Speeds up SAM3 attention layers; skipped silently if the build fails
# (e.g. on machines where ninja or CUDA headers are unavailable).
RUN uv pip install flash-attn-3 --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    || echo "[INFO] flash-attn-3 skipped"

# ── CC Torch (connected-components kernel, used by SAM3) ────────────────────
RUN uv pip install git+https://github.com/ronghanghu/cc_torch.git \
    || echo "[INFO] cc_torch skipped"

# ── Project Python dependencies ──────────────────────────────────────────────
# easyocr, opencv-python, pillow — declared in pyproject.toml [project.dependencies]
RUN uv pip install \
    "easyocr>=1.7.2" \
    "opencv-python>=4.13.0.92" \
    "pillow>=12.2.0"

# ── Application source ───────────────────────────────────────────────────────
WORKDIR /app
COPY pipeline/  ./pipeline/
COPY setups/    ./setups/
COPY test/      ./test/
COPY pyproject.toml ./

# Default: run the full pipeline with verbose logging.
# Override via `docker compose run pipeline <command>`.
CMD ["python", "pipeline/app.py", "--verbose"]
