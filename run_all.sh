#!/bin/bash

# Complete Build and Test Script for CUDA Orderbook
# Run this from project root: ./run_all.sh

set -e  # Exit on any error

echo "========================================="
echo "CUDA Orderbook - Complete Build & Test"
echo "========================================="
echo ""

# Get project root (where this script is located)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: Clean previous builds
# ============================================================================
echo "[1/6] Cleaning previous builds..."
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
echo "[2/6] Building core library..."
echo "-----------------------------------"

cd build

if ! cmake ..; then
    echo "❌ CMake configuration FAILED"
    exit 1
fi

if ! make; then
    echo "❌ Core library build FAILED"
    exit 1
fi

if [ ! -f "libcuda_orderbook.a" ]; then
    echo "❌ Library file not found"
    exit 1
fi

echo "✓ Core library built: libcuda_orderbook.a"
cd "$PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 3: Build test suite
# ============================================================================
echo "[3/6] Building test suite..."
echo "-----------------------------------"

cd tests

if ! make -f Makefile_tests; then
    echo "❌ Test suite build FAILED"
    exit 1
fi

if [ ! -f "test_suite" ]; then
    echo "❌ Test suite binary not found"
    exit 1
fi

echo "✓ Test suite built: test_suite"
echo ""

# ============================================================================
# STEP 4: Run test suite (CRITICAL!)
# ============================================================================
echo "[4/6] Running test suite..."
echo "-----------------------------------"
echo "⚠️  This is the most important step!"
echo ""

if ! ./test_suite; then
    echo ""
    echo "❌❌❌ TESTS FAILED ❌❌❌"
    echo ""
    echo "Some tests did not pass. This means:"
    echo "  - CPU and GPU implementations don't match"
    echo "  - There may be bugs in the code"
    echo ""
    echo "⚠️  DO NOT PROCEED until all tests pass!"
    echo ""
    exit 1
fi

echo ""
echo "✓✓✓ All tests PASSED! ✓✓✓"
cd "$PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 5: Build benchmarks
# ============================================================================
echo "[5/6] Building benchmarks..."
echo "-----------------------------------"

cd benchmarks

if ! make; then
    echo "❌ Benchmark build FAILED"
    exit 1
fi

if [ ! -f "benchmark_cpu_vs_gpu" ]; then
    echo "❌ Benchmark binary not found"
    exit 1
fi

echo "✓ Benchmarks built: benchmark_cpu_vs_gpu"
echo ""

# ============================================================================
# STEP 6: Run small benchmark
# ============================================================================
echo "[6/6] Running small benchmark..."
echo "-----------------------------------"

if ! make run-small; then
    echo "❌ Benchmark run FAILED"
    exit 1
fi

cd "$PROJECT_ROOT"
echo ""

# ============================================================================
# SUCCESS!
# ============================================================================
echo ""
echo "========================================="
echo "✅✅✅ ALL BUILDS AND TESTS SUCCESSFUL! ✅✅✅"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✓ Core library built"
echo "  ✓ Test suite built"
echo "  ✓ All 13 tests PASSED (CPU == GPU)"
echo "  ✓ Benchmarks built"
echo "  ✓ Small benchmark completed"
echo ""
echo "Next steps:"
echo "  - Run more benchmarks: cd benchmarks && make run-medium"
echo "  - Profile GPU code: cd benchmarks && make profile-nsys"
echo "  - Scale up testing: cd tests && edit test_suite.cu"
echo ""
echo "🎉 Your CUDA orderbook is working correctly!"
echo ""

