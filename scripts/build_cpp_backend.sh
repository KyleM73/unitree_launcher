#!/usr/bin/env bash
# Build the unitree_interface C++ pybind11 module from source.
#
# The C++ backend wraps amazon-far/unitree_sdk2 and provides jitter-free
# 500 Hz DDS command publishing for real robot deployment.
#
# NOTE: libunitree_sdk2.a is closed-source and only available as pre-compiled
# static libraries for Linux x86_64 and aarch64. There is no macOS binary.
# This script will exit with an error on non-Linux platforms.
#
# Usage:
#   ./scripts/build_cpp_backend.sh           # build and install into active venv
#   ./scripts/build_cpp_backend.sh --clean   # remove cached clone and rebuild
#   CLONE_DIR=/path/to/dir ./scripts/...     # override clone location
#
# Requirements (auto-checked):
#   - Linux (x86_64 or aarch64)
#   - cmake, g++ (or build-essential)
#   - pybind11 (auto-installed into venv if missing)
#   - patchelf (auto-installed via pip if missing)
#   - Active Python venv (checks $VIRTUAL_ENV or .venv/)

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[build]${NC} $*"; }
warn()  { echo -e "${YELLOW}[build]${NC} $*"; }
error() { echo -e "${RED}[build]${NC} $*" >&2; }

# ── Platform check ──────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This script only works on Linux."
    error "libunitree_sdk2.a is closed-source and only distributed for Linux x86_64/aarch64."
    error "macOS users: use --backend python for the pure-Python DDS backend,"
    error "or run sim-only (no real robot backend needed)."
    exit 1
fi

# ── Architecture detection ──────────────────────────────────────────────
MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64)  ARCH="x86_64" ;;
    aarch64) ARCH="aarch64" ;;
    *)
        error "Unsupported architecture: $MACHINE (need x86_64 or aarch64)"
        exit 1
        ;;
esac
info "Architecture: $ARCH"

# ── Parse flags ─────────────────────────────────────────────────────────
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo "  --clean   Remove cached clone and rebuild from scratch"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ── Find Python venv ────────────────────────────────────────────────────
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    VENV_DIR="$VIRTUAL_ENV"
elif [[ -d ".venv" ]]; then
    VENV_DIR="$(pwd)/.venv"
else
    error "No active Python virtual environment found."
    error "Activate a venv or create one with: python -m venv .venv && source .venv/bin/activate"
    exit 1
fi

PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    error "Python not found at $PYTHON"
    exit 1
fi

SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
info "Venv: $VENV_DIR"
info "Site-packages: $SITE_PACKAGES"

# ── Dependency checks ──────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "Required command not found: $1"
        error "Install with: $2"
        exit 1
    fi
}

check_cmd cmake "apt-get install cmake"
check_cmd g++ "apt-get install build-essential"

# pybind11 and patchelf are installed automatically on Linux via pyproject.toml
# (sys_platform == 'linux' markers). Verify they're present.
if ! "$PYTHON" -c "import pybind11" &>/dev/null; then
    error "pybind11 not found. Run: uv sync (should auto-install on Linux)"
    exit 1
fi

PATCHELF="patchelf"
if ! command -v patchelf &>/dev/null; then
    PATCHELF="$VENV_DIR/bin/patchelf"
    if [[ ! -x "$PATCHELF" ]]; then
        error "patchelf not found. Run: uv sync (should auto-install on Linux)"
        exit 1
    fi
fi

# ── Clone source ────────────────────────────────────────────────────────
CLONE_DIR="${CLONE_DIR:-/tmp/unitree_sdk2_build}"

if [[ "$CLEAN" -eq 1 ]] && [[ -d "$CLONE_DIR" ]]; then
    info "Cleaning previous build at $CLONE_DIR"
    rm -rf "$CLONE_DIR"
fi

if [[ ! -d "$CLONE_DIR" ]]; then
    info "Cloning amazon-far/unitree_sdk2 (dev branch)..."
    git clone --depth 1 --branch dev \
        https://github.com/amazon-far/unitree_sdk2.git \
        "$CLONE_DIR"
else
    info "Using cached clone at $CLONE_DIR"
fi

# ── Verify expected files exist ─────────────────────────────────────────
SDK_LIB="$CLONE_DIR/lib/$ARCH/libunitree_sdk2.a"
DDS_LIB_DIR="$CLONE_DIR/thirdparty/lib/$ARCH"
BINDING_DIR="$CLONE_DIR/python_binding"

if [[ ! -f "$SDK_LIB" ]]; then
    error "Static library not found: $SDK_LIB"
    error "The clone may be incomplete or the architecture ($ARCH) is not supported."
    exit 1
fi

if [[ ! -d "$BINDING_DIR" ]]; then
    error "Python binding source not found: $BINDING_DIR"
    exit 1
fi

# ── CMake configure + build ─────────────────────────────────────────────
BUILD_DIR="$CLONE_DIR/build"
mkdir -p "$BUILD_DIR"

info "Configuring CMake..."
cmake -S "$CLONE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDING=ON \
    -DBUILD_EXAMPLES=OFF \
    -DPYTHON_EXECUTABLE="$PYTHON" \
    -DPython_EXECUTABLE="$PYTHON" \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DPYBIND11_FINDPYTHON=ON \
    -Dpybind11_DIR="$("$PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

info "Building unitree_interface..."
cmake --build "$BUILD_DIR" --target unitree_interface --parallel "$(nproc)"

# ── Find built .so ──────────────────────────────────────────────────────
SO_FILE="$(find "$BUILD_DIR" -name 'unitree_interface*.so' -type f | head -1)"
if [[ -z "$SO_FILE" ]]; then
    error "Build succeeded but unitree_interface*.so not found in $BUILD_DIR"
    exit 1
fi
info "Built: $SO_FILE"

# ── Install .so into site-packages ──────────────────────────────────────
info "Installing to $SITE_PACKAGES"
cp "$SO_FILE" "$SITE_PACKAGES/"

# ── Install DDS shared libraries alongside ──────────────────────────────
DDS_LIBS=(libddsc.so libddscxx.so)
for lib in "${DDS_LIBS[@]}"; do
    # Find the actual file (may have version suffix like libddsc.so.0.10.2)
    SRC="$(find "$DDS_LIB_DIR" -name "${lib}*" -type f | head -1)"
    if [[ -z "$SRC" ]]; then
        # Try symlink
        SRC="$(find "$DDS_LIB_DIR" -name "${lib}*" | head -1)"
    fi
    if [[ -n "$SRC" ]]; then
        # Copy the actual file
        REAL_SRC="$(readlink -f "$SRC")"
        DEST_NAME="$lib"
        cp "$REAL_SRC" "$SITE_PACKAGES/$DEST_NAME"
        info "  Installed $DEST_NAME"
    else
        warn "  DDS library not found: $lib in $DDS_LIB_DIR (may be statically linked)"
    fi
done

# ── Set RPATH so .so finds DDS libs without LD_LIBRARY_PATH ─────────────
INSTALLED_SO="$SITE_PACKAGES/$(basename "$SO_FILE")"
"$PATCHELF" --set-rpath '$ORIGIN' "$INSTALLED_SO"
info "Set RPATH to \$ORIGIN on $(basename "$SO_FILE")"

# ── Patch unitree_sdk2py: remove broken 'b2' import ─────────────────────
SDK2PY_INIT="$SITE_PACKAGES/unitree_sdk2py/__init__.py"
if [[ -f "$SDK2PY_INIT" ]] && grep -q 'b2' "$SDK2PY_INIT"; then
    info "Patching unitree_sdk2py (removing broken 'b2' import)..."
    sed -i 's/from \. import idl, utils, core, rpc, go2, b2/from . import idl, utils, core, rpc, go2/' "$SDK2PY_INIT"
    sed -i '/"b2",/d' "$SDK2PY_INIT"
fi

# ── Verify ──────────────────────────────────────────────────────────────
info "Verifying import..."
if "$PYTHON" -c "import unitree_interface; print('unitree_interface loaded:', dir(unitree_interface))"; then
    echo ""
    info "SUCCESS — unitree_interface installed and importable."
    info "Use with: python -m unitree_launcher.main real --backend cpp ..."
else
    echo ""
    error "FAILED — unitree_interface could not be imported."
    error "Check that DDS libraries are available. You may need:"
    error "  export LD_LIBRARY_PATH=$SITE_PACKAGES:\$LD_LIBRARY_PATH"
    exit 1
fi
