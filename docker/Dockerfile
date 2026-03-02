# Unitree G1 Deployment Stack — Docker image
# Supports headless (EGL) and GUI (GLX/X11) rendering.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t unitree-launcher .
#
# Run headless:
#   docker run --rm unitree-launcher sim --headless --policy assets/policies/stance_29dof.onnx --duration 5
#
# Run with X11 (Linux):
#   xhost +local:docker
#   docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e MUJOCO_GL=glx \
#       unitree-launcher sim --policy assets/policies/stance_29dof.onnx

# ── Build args ─────────────────────────────────────────────────────────
# Set BUILD_CPP_BACKEND=1 to include the C++ unitree_interface pybind11
# module for real robot deployment (requires Linux, adds ~2 min build time).
ARG BUILD_CPP_BACKEND=0

# ── Builder stage ──────────────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS builder

ARG BUILD_CPP_BACKEND

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        $([ "$BUILD_CPP_BACKEND" = "1" ] && echo "patchelf" || true) \
    && rm -rf /var/lib/apt/lists/*

# Build CycloneDDS C library from source (required by the Python binding).
RUN git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && mkdir /tmp/cyclonedds/build \
    && cd /tmp/cyclonedds/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    && cmake --build . --parallel $(nproc) \
    && cmake --install . \
    && rm -rf /tmp/cyclonedds

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds

# Install uv for fast dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency metadata first (cache layer).
COPY pyproject.toml ./

# Copy source and assets needed for editable install.
COPY src/ src/
COPY assets/ assets/
COPY configs/ configs/
COPY scripts/ scripts/

# Create venv and install all dependencies (including dev for pytest).
RUN uv venv /app/.venv --python 3.10 \
    && . /app/.venv/bin/activate \
    && uv pip install -e ".[dev]"

# Patch unitree_sdk2py: remove 'b2' submodule import (doesn't exist, causes
# circular import errors). We don't use the b2 API.
RUN sed -i 's/from \. import idl, utils, core, rpc, go2, b2/from . import idl, utils, core, rpc, go2/' \
    /app/.venv/lib/python3.10/site-packages/unitree_sdk2py/__init__.py \
    && sed -i '/"b2",/d' \
    /app/.venv/lib/python3.10/site-packages/unitree_sdk2py/__init__.py

# ── Optional: C++ unitree_interface backend ────────────────────────────
# Build the pybind11 module from amazon-far/unitree_sdk2 for real robot
# deployment. Adds 500 Hz jitter-free DDS command publishing.
RUN if [ "$BUILD_CPP_BACKEND" = "1" ]; then \
        set -e \
        && echo "[docker] Building C++ unitree_interface backend..." \
        && ARCH="$(uname -m)" \
        && . /app/.venv/bin/activate \
        && uv pip install pybind11 --quiet \
        && git clone --depth 1 --branch dev \
            https://github.com/amazon-far/unitree_sdk2.git /tmp/unitree_sdk2 \
        && mkdir -p /tmp/unitree_sdk2/build \
        && cmake -S /tmp/unitree_sdk2 -B /tmp/unitree_sdk2/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_PYTHON_BINDING=ON \
            -DBUILD_EXAMPLES=OFF \
            -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
        && cmake --build /tmp/unitree_sdk2/build --target unitree_interface --parallel $(nproc) \
        && SITE_PKG="$(python -c 'import site; print(site.getsitepackages()[0])')" \
        && cp /tmp/unitree_sdk2/build/python_binding/unitree_interface*.so "$SITE_PKG/" \
        && DDS_DIR="/tmp/unitree_sdk2/thirdparty/lib/$ARCH" \
        && cp -a "$DDS_DIR"/libddsc.so* "$DDS_DIR"/libddscxx.so* "$SITE_PKG/" \
        && SO_FILE="$(ls "$SITE_PKG"/unitree_interface*.so)" \
        && patchelf --set-rpath '$ORIGIN' "$SO_FILE" \
        && python -c "import unitree_interface; print('[docker] unitree_interface OK:', dir(unitree_interface))" \
        && rm -rf /tmp/unitree_sdk2 \
        && echo "[docker] C++ backend installed successfully"; \
    else \
        echo "[docker] Skipping C++ backend (BUILD_CPP_BACKEND=0)"; \
    fi

# ── Runtime stage ──────────────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS runtime

# Rendering libraries:
#   libegl1-mesa-dev  — EGL (headless offscreen rendering)
#   libgl1-mesa-dri   — DRI drivers for software rendering
#   libglx-mesa0      — GLX (X11 GUI rendering)
#   libx11-6 etc.     — X11 client libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libx11-6 \
        libxext6 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/*

# CycloneDDS C library (needed by Python binding at runtime).
COPY --from=builder /opt/cyclonedds /opt/cyclonedds
ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV LD_LIBRARY_PATH=/opt/cyclonedds/lib

# Default to EGL (headless). Override to "glx" for X11 GUI.
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy venv and app from builder.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/assets /app/assets
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Activate venv by putting it on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Tests live outside the installed package — copy them separately.
COPY tests/ tests/

ENTRYPOINT ["python", "-m", "unitree_launcher.main"]
CMD ["--help"]
