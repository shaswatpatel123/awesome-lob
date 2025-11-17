#!/bin/bash
# Rebuild script for optimized parallel code

echo "============================================"
echo "REBUILDING WITH PARALLEL OPTIMIZATION"
echo "============================================"
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  ERROR: nvcc not found in PATH"
    echo ""
    echo "Please load CUDA module:"
    echo "  module load cuda/12.0"
    echo "  module load gcc/11.2.0"
    echo ""
    echo "Or set PATH manually:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

echo "✓ nvcc found: $(which nvcc)"
echo "✓ CUDA version: $(nvcc --version | grep release | cut -d' ' -f5)"
echo ""

# Set base directory
BASE_DIR="/Users/kvlnraju/Desktop/courses/semester_3/GPU/Project/awesome-lob"
cd "$BASE_DIR"

echo "============================================"
echo "STEP 1: Rebuilding Test Suite"
echo "============================================"
cd "$BASE_DIR/tests"

echo "Cleaning..."
make -f Makefile_tests clean

echo "Compiling..."
if make -f Makefile_tests -j4; then
    echo "✓ Test suite compiled successfully!"
else
    echo "✗ Test suite compilation failed!"
    exit 1
fi
echo ""

echo "============================================"
echo "STEP 2: Rebuilding Benchmarks"
echo "============================================"
cd "$BASE_DIR/benchmarks"

echo "Cleaning..."
make clean

echo "Compiling..."
if make -j4; then
    echo "✓ Benchmarks compiled successfully!"
else
    echo "✗ Benchmark compilation failed!"
    exit 1
fi
echo ""

echo "============================================"
echo "BUILD COMPLETE! ✓"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Run quick correctness test:"
echo "   cd tests"
echo "   ./test_suite --functional-only 1000"
echo ""
echo "2. Run performance benchmark:"
echo "   cd benchmarks"
echo "   ./benchmark_cpu_vs_gpu 10000 1000 1000 100"
echo ""
echo "Expected improvements:"
echo "  - GPU time: 30-80x faster"
echo "  - Speedup vs CPU: 45-113x (was 1.38x)"
echo ""
echo "See PARALLEL_OPTIMIZATION.md for details!"
echo ""

