#!/bin/bash
#
# Download script for llama.cpp third-party dependencies
# Only abseil-cpp is required, other dependencies are included in vendor/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ABSEIL_VERSION="20240722.0"
ABSEIL_URL="https://github.com/abseil/abseil-cpp/archive/refs/tags/${ABSEIL_VERSION}.tar.gz"

echo "========================================"
echo "llama.cpp Third-Party Dependencies Download"
echo "========================================"
echo ""

# Check if abseil already exists
if [ -d "abseil" ] && [ -f "abseil/CMakeLists.txt" ]; then
    echo "abseil-cpp already exists, skipping download."
else
    echo "Downloading abseil-cpp ${ABSEIL_VERSION}..."

    # Try git first, fallback to curl
    if command -v git &> /dev/null; then
        echo "Using git to clone abseil-cpp..."
        git clone --depth 1 --branch "${ABSEIL_VERSION}" https://github.com/abseil/abseil-cpp.git abseil
        # Remove hidden files/directories
        rm -rf abseil/.git abseil/.github abseil/.gitignore abseil/.clang-format 2>/dev/null || true
    elif command -v curl &> /dev/null; then
        echo "Using curl to download abseil-cpp..."
        curl -L "${ABSEIL_URL}" -o abseil.tar.gz
        tar xzf abseil.tar.gz
        mv "abseil-cpp-${ABSEIL_VERSION}" abseil
        rm abseil.tar.gz
    elif command -v wget &> /dev/null; then
        echo "Using wget to download abseil-cpp..."
        wget "${ABSEIL_URL}" -O abseil.tar.gz
        tar xzf abseil.tar.gz
        mv "abseil-cpp-${ABSEIL_VERSION}" abseil
        rm abseil.tar.gz
    else
        echo "Error: Neither git, curl, nor wget is available."
        echo "Please install one of them and run this script again."
        exit 1
    fi

    echo "abseil-cpp downloaded successfully!"
fi

echo ""
echo "========================================"
echo "Dependency Summary"
echo "========================================"
echo ""
echo "Required dependencies:"
echo "  - abseil-cpp: $( [ -d "abseil" ] && echo 'OK' || echo 'MISSING' )"
echo ""
echo "Included in vendor/ directory (no download needed):"
echo "  - nlohmann/json"
echo "  - cpp-httplib"
echo "  - miniaudio"
echo "  - stb"
echo "  - sheredom"
echo ""
echo "Optional system dependencies (auto-detected):"
echo "  - OpenSSL (HTTPS support)"
echo "  - BLAS (linear algebra acceleration)"
echo "  - OpenMP (parallel computing)"
echo "  - CUDA/Vulkan (GPU acceleration)"
echo ""
echo "========================================"
echo "Ready to build llama.cpp!"
echo ""
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  make -j\$(nproc)"
echo "========================================"
