#!/usr/bin/env bash
# Build the unitree_cpp Python binding for real robot deployment.
#
# Two-step process:
#   1. Install unitree_sdk2 C++ library (headers + static lib to /usr/local)
#   2. Build unitree_cpp pybind11 module (from HansZ8/unitree_cpp)
#
# unitree_cpp provides a high-performance Python binding for the Unitree SDK2
# with a background command re-publishing thread, CRC handling, and wireless
# controller data — matching the RoboJuDo deployment stack.
#
# Usage:
#   ./scripts/build_cpp_backend.sh           # build and install
#   ./scripts/build_cpp_backend.sh --clean   # remove caches and rebuild
#
# Requirements (auto-checked):
#   - Linux (x86_64 or aarch64)
#   - cmake, g++ (or build-essential)
#   - uv (Python package manager)

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[build]${NC} $*"; }
warn()  { echo -e "${YELLOW}[build]${NC} $*"; }
error() { echo -e "${RED}[build]${NC} $*" >&2; }

# ── Platform check ──────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This script only works on Linux (the C++ SDK is Linux-only)."
    exit 1
fi

MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64|aarch64) info "Architecture: $MACHINE" ;;
    *)
        error "Unsupported architecture: $MACHINE (need x86_64 or aarch64)"
        exit 1
        ;;
esac

# ── Parse flags ─────────────────────────────────────────────────────────
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo "  --clean   Remove cached clones and rebuild from scratch"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ── Check uv ────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    error "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── Dependency checks ──────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "Required command not found: $1"
        error "Install with: $2"
        exit 1
    fi
}

check_cmd cmake "sudo apt-get install cmake"
check_cmd g++ "sudo apt-get install build-essential"

# ── Cache directories ──────────────────────────────────────────────────
SDK2_DIR="${SDK2_DIR:-/tmp/unitree_sdk2_build}"
UCPP_DIR="${UCPP_DIR:-/tmp/unitree_cpp_build}"

if [[ "$CLEAN" -eq 1 ]]; then
    info "Cleaning previous builds..."
    rm -rf "$SDK2_DIR" "$UCPP_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Install unitree_sdk2 C++ library
# ═══════════════════════════════════════════════════════════════════════

# Check if already installed
if [[ -f /usr/local/lib/libunitree_sdk2.a ]] && \
   [[ -d /usr/local/include/unitree ]]; then
    info "unitree_sdk2 already installed in /usr/local (skipping)"
else
    info "=== Step 1: Install unitree_sdk2 C++ library ==="

    if [[ ! -d "$SDK2_DIR" ]]; then
        info "Cloning unitreerobotics/unitree_sdk2..."
        git clone --depth 1 \
            https://github.com/unitreerobotics/unitree_sdk2.git \
            "$SDK2_DIR"
    else
        info "Using cached clone at $SDK2_DIR"
    fi

    mkdir -p "$SDK2_DIR/build"
    info "Configuring unitree_sdk2..."
    cmake -S "$SDK2_DIR" -B "$SDK2_DIR/build" -DCMAKE_BUILD_TYPE=Release

    info "Building unitree_sdk2..."
    cmake --build "$SDK2_DIR/build" --parallel "$(nproc)"

    info "Installing unitree_sdk2 to /usr/local (requires sudo)..."
    sudo cmake --install "$SDK2_DIR/build"

    info "unitree_sdk2 installed to /usr/local"
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Build and install unitree_cpp Python binding
# ═══════════════════════════════════════════════════════════════════════

info "=== Step 2: Build unitree_cpp Python binding ==="

if [[ ! -d "$UCPP_DIR" ]]; then
    info "Cloning HansZ8/unitree_cpp..."
    git clone --depth 1 \
        https://github.com/HansZ8/unitree_cpp.git \
        "$UCPP_DIR"
else
    info "Using cached clone at $UCPP_DIR"
fi

# Patch out verbose "Gains set:" print that fires every tick
CTRL_CPP="$UCPP_DIR/src/unitree_controller.cpp"
if grep -q 'std::cout << "Gains set:' "$CTRL_CPP" 2>/dev/null; then
    info "Patching out verbose gains logging..."
    sed -i '/std::cout << "Gains set:/,/std::endl;/d' "$CTRL_CPP"
fi

info "Installing unitree_cpp via uv pip install..."
cd "$UCPP_DIR"
uv pip install --project /home/unitree/unitree_launcher .

# ── Verify ──────────────────────────────────────────────────────────────
info "Verifying import..."
cd /home/unitree/unitree_launcher
if uv run python -c "from unitree_cpp import UnitreeController; print('unitree_cpp OK')"; then
    echo ""
    info "SUCCESS — unitree_cpp installed and importable."
    info "Run with: uv run real --gantry"
else
    echo ""
    error "FAILED — unitree_cpp could not be imported."
    error "Check that unitree_sdk2 is installed: ls /usr/local/lib/libunitree_sdk2.a"
    exit 1
fi
