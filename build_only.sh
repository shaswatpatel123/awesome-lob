#!/bin/bash

# Build-Only Script - Just compile, don't run anything
# Use this to verify the code compiles correctly

set -e  # Exit on any error

echo "========================================="
echo "CUDA Orderbook - Build Verification"
echo "========================================="
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: Clean previous builds
# ============================================================================
echo "[1/3] Cleaning previous builds..."
echo "-----------------------------------"

cd build 2>/dev/null && make clean 2>/dev/null || true
cd "$PROJECT_ROOT"

cd tests 2>/dev/null && make -f Makefile_tests clean 2>/dev/null || true
cd "$PROJECT_ROOT"

cd benchmarks 2>/dev/null && make clean 2>/dev/null || true
cd "$PROJECT_ROOT"

echo "✓ Clean complete"
echo ""

# ============================================================================
# STEP 2: Build core library
# ============================================================================
echo "[2/3] Building core library..."
echo "-----------------------------------"

cd build

echo "Running CMake configuration..."
if ! cmake ..; then
    echo ""
    echo "❌ CMAKE CONFIGURATION FAILED"
    echo ""
    echo "Possible issues:"
    echo "  - CUDA toolkit not installed"
    echo "  - CMake version too old"
    echo "  - Missing dependencies"
    echo ""
    echo "Check errors above ↑"
    exit 1
fi

echo ""
echo "Compiling CUDA kernels and CPU code..."
if ! make; then
    echo ""
    echo "❌ COMPILATION FAILED"
    echo ""
    echo "Possible issues:"
    echo "  - Syntax errors in code"
    echo "  - Missing header files"
    echo "  - CUDA compilation errors"
    echo "  - Linker errors"
    echo ""
    echo "Check errors above ↑"
    exit 1
fi

if [ ! -f "libcuda_orderbook.a" ]; then
    echo ""
    echo "❌ LIBRARY FILE NOT FOUND"
    echo "Expected: libcuda_orderbook.a"
    exit 1
fi

echo ""
echo "✓ Core library built successfully"
echo "  Output: build/libcuda_orderbook.a"
ls -lh libcuda_orderbook.a

cd "$PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 3: Build test suite (compilation only, don't run)
# ============================================================================
echo "[3/3] Building test suite..."
echo "-----------------------------------"

cd tests

echo "Compiling test suite..."
if ! make -f Makefile_tests; then
    echo ""
    echo "❌ TEST SUITE BUILD FAILED"
    echo ""
    echo "Possible issues:"
    echo "  - Missing data_generator.cpp"
    echo "  - Syntax errors in test_suite.cu"
    echo "  - Linker errors with core library"
    echo ""
    echo "Check errors above ↑"
    exit 1
fi

if [ ! -f "test_suite" ]; then
    echo ""
    echo "❌ TEST BINARY NOT FOUND"
    echo "Expected: test_suite"
    exit 1
fi

echo ""
echo "✓ Test suite built successfully"
echo "  Output: tests/test_suite"
ls -lh test_suite

cd "$PROJECT_ROOT"
echo ""

# ============================================================================
# SUCCESS!
# ============================================================================
echo ""
echo "========================================="
echo "✅ BUILD VERIFICATION SUCCESSFUL!"
echo "========================================="
echo ""
echo "All components compiled without errors:"
echo "  ✓ Core library (libcuda_orderbook.a)"
echo "  ✓ Test suite (test_suite)"
echo ""
echo "Your code compiles correctly! 🎉"
echo ""
echo "Next steps:"
echo "  1. Run tests: cd tests && ./test_suite"
echo "  2. Or use full script: ./run_all.sh"
echo ""

