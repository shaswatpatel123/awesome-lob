# Executable Options Guide

Complete reference for running benchmarks and test suite with all available options.

---

## 📊 Benchmark Executable (`benchmark_cpu_vs_gpu`)

### Location
```bash
cd benchmarks
```

### Build Commands
```bash
# Build with auto-detected GPU architecture
make

# Build with specific GPU architecture
make GPU_ARCH=sm_75        # For T4, V100
make GPU_ARCH=sm_80        # For A100, RTX 30/40 series

# Build for multiple architectures (fat binary)
# Automatically enabled if no GPU_ARCH specified
```

### Executable Syntax
```bash
./benchmark_cpu_vs_gpu [num_books] [messages_per_book] [orders_per_side] [max_trades] [block_size]
```

### Parameters

| Parameter | Position | Default | Description |
|-----------|----------|---------|-------------|
| `num_books` | 1 | 100 | Number of orderbooks to process |
| `messages_per_book` | 2 | 1000 | Messages per orderbook |
| `orders_per_side` | 3 | 100 | Maximum orders per side per book |
| `max_trades` | 4 | 100 | Maximum trades per book |
| `block_size` | 5 | 256 | Threads per block (must be power of 2: 32, 64, 128, 256, 512, 1024) |

### Quick Start Commands (via Makefile)

```bash
# Default workload
make run
# Equivalent to: ./benchmark_cpu_vs_gpu 100 1000 100 100

# Small workload (fast, for quick tests)
make run-small
# Equivalent to: ./benchmark_cpu_vs_gpu 10 100 50 50

# Medium workload (typical)
make run-medium
# Equivalent to: ./benchmark_cpu_vs_gpu 100 1000 100 100

# Large workload (stress test)
make run-large
# Equivalent to: ./benchmark_cpu_vs_gpu 1000 10000 200 200
```

### Direct Executable Examples

```bash
# Minimal (uses all defaults)
./benchmark_cpu_vs_gpu

# Small test
./benchmark_cpu_vs_gpu 10 100 50 50

# Typical workload
./benchmark_cpu_vs_gpu 1000 1000 100 100

# Large scale
./benchmark_cpu_vs_gpu 10000 1000 100 100

# Custom block size
./benchmark_cpu_vs_gpu 1000 1000 100 100 128

# Very large scale (requires sufficient GPU memory)
./benchmark_cpu_vs_gpu 10000 2000 200 200 256
```

### Profiling Commands

```bash
# Basic profiling with nvprof
make profile
# Or manually:
nvprof ./benchmark_cpu_vs_gpu 1000 1000 100 100

# Detailed profiling with Nsight Systems (if available)
make profile-nsys
# Or manually:
nsys profile --stats=true ./benchmark_cpu_vs_gpu 1000 1000 100 100

# Kernel-level analysis with Nsight Compute (if available)
make profile-ncu
# Or manually:
ncu --set full ./benchmark_cpu_vs_gpu 1000 1000 100 100

# Simple timing (always available)
make profile-simple
# Or manually:
time ./benchmark_cpu_vs_gpu 1000 1000 100 100
```

### Example Output
```
=== CPU vs GPU Orderbook Benchmark ===

Configuration:
  Number of orderbooks: 1000
  Messages per orderbook: 1000
  Orders per side: 100
  Max trades: 100
  Block size (threads per block): 256
  Total messages: 1000000

=== CPU Benchmark ===
CPU Allocation Time: 125.5 ms
CPU Time: 527.556 ms
CPU Throughput: 1.89553e+06 messages/sec

=== GPU Benchmark ===
GPU Allocation Time: 15.2 ms
GPU Time: 195.375 ms
GPU Throughput: 5.11837e+06 messages/sec

=== Comparison ===
CPU Time: 527.556 ms
GPU Time: 195.375 ms
GPU Speedup: 2.70023x
✓ GPU is 2.70023x faster than CPU!
```

---

## 🧪 Test Suite Executable (`test_suite`)

### Location
```bash
cd tests
```

### Build Commands
```bash
# Build with auto-detected GPU architecture
make -f Makefile_tests

# Build with specific GPU architecture
make -f Makefile_tests GPU_ARCH=sm_75

# Clean build artifacts
make -f Makefile_tests clean
```

### Executable Syntax
```bash
./test_suite [max_messages] [max_orders]
```

### Parameters

| Parameter | Position | Default | Description |
|-----------|----------|---------|-------------|
| `max_messages` | 1 | 10000 | Maximum messages for functional tests |
| `max_orders` | 2 | 1000 | Maximum orders per side for functional tests |

### Quick Start Commands (via Makefile)

```bash
# Run all tests with defaults
make -f Makefile_tests run
# Equivalent to: ./test_suite 10000 1000

# Build only
make -f Makefile_tests

# Clean
make -f Makefile_tests clean
```

### Direct Executable Examples

```bash
# Minimal (uses defaults: 10000 max_messages, 1000 max_orders)
./test_suite

# Quick test (smaller functional tests)
./test_suite 1000 100

# Standard test suite
./test_suite 10000 1000

# Large scale tests (includes 5000 and 10000 message tests)
./test_suite 20000 2000

# Very large scale
./test_suite 50000 5000
```

### Test Levels

The test suite runs tests in three levels:

1. **Level 1: Unit Tests** (always runs)
   - Add order test
   - Cancel order test
   - Simple match test

2. **Level 2: Integration Tests** (always runs)
   - Partial fill
   - No match
   - Price improvement
   - Cancel test
   - Market order
   - Price-time priority
   - Multi-level book

3. **Level 3: Functional Tests** (scales with max_messages)
   - Small (100 messages)
   - Medium (500 messages)
   - Large (1000 messages)
   - Very Large (5000 messages) - only if max_messages >= 5000
   - Massive (10000 messages) - only if max_messages >= 10000

### Example Output
```
============================================================
CUDA ORDERBOOK TEST SUITE
Comprehensive Testing: Unit → Integration → Functional
============================================================

============================================================
LEVEL 1: UNIT TESTS (Individual Operations)
============================================================

------------------------------------------------------------
TEST: Unit Test: Add Order
------------------------------------------------------------
  ✓ PASS: Order added correctly, CPU == GPU

[... more tests ...]

============================================================
TEST SUMMARY
============================================================
Total tests: 13
✓ Passed: 13
✗ Failed: 0

🎉 ALL TESTS PASSED!
============================================================
```

---

## 🔧 Common Workflows

### Quick Benchmark Test
```bash
cd benchmarks
make run-small
```

### Full Benchmark Suite
```bash
cd benchmarks
make run-small && make run-medium && make run-large
```

### Test Suite Quick Check
```bash
cd tests
make -f Makefile_tests run
```

### Test Suite Large Scale
```bash
cd tests
./test_suite 50000 5000
```

### Profile Specific Workload
```bash
cd benchmarks
nvprof ./benchmark_cpu_vs_gpu 5000 2000 200 200 256
```

### Custom Workload with Profiling
```bash
cd benchmarks
make
time ./benchmark_cpu_vs_gpu 10000 1000 100 100 32
nvprof ./benchmark_cpu_vs_gpu 10000 1000 100 100 32
```

---

## 📋 Parameter Guidelines

### Benchmark Parameters

**num_books:**
- Small: 10-100
- Medium: 100-1000
- Large: 1000-10000
- Very Large: 10000+ (requires sufficient GPU memory)

**messages_per_book:**
- Quick test: 100-500
- Typical: 1000-2000
- Stress test: 5000-10000

**orders_per_side:**
- Small: 50-100
- Typical: 100-200
- Large: 200-500

**block_size:**
- Must be power of 2
- Common: 32, 64, 128, 256, 512, 1024
- Default 256 works well for most cases
- Lower values (32-128) may reduce GPU utilization
- Higher values (512-1024) may exceed GPU limits

### Test Suite Parameters

**max_messages:**
- Quick: 1000
- Standard: 10000
- Comprehensive: 50000+

**max_orders:**
- Quick: 100-500
- Standard: 1000
- Comprehensive: 2000-5000

---

## ⚠️ Memory Considerations

### GPU Memory
- With reduced hash tables (2048): ~200 KB per book
- 10,000 books ≈ 2 GB GPU memory
- T4 (16GB) can handle 10,000+ books easily

### CPU Memory
- Same memory requirements as GPU
- Ensure sufficient RAM available
- Check memory warnings in output

### If Memory Errors Occur:
1. Reduce number of books
2. Reduce hash table sizes (already done: 2048)
3. Reduce orders_per_side or max_trades
4. Use smaller workloads for testing

---

## 🎯 Recommended Workloads

### For Development/Quick Tests
```bash
# Benchmark
cd benchmarks && make run-small

# Tests
cd tests && ./test_suite 1000 100
```

### For Standard Benchmarking
```bash
cd benchmarks && make run-medium
```

### For Performance Evaluation
```bash
cd benchmarks
./benchmark_cpu_vs_gpu 1000 1000 100 100 256
nvprof ./benchmark_cpu_vs_gpu 1000 1000 100 100 256
```

### For Stress Testing
```bash
cd benchmarks && make run-large
cd tests && ./test_suite 50000 2000
```

---

## 🔍 Help Commands

```bash
# Benchmark help
cd benchmarks && make help

# Check what GPU architecture will be used
cd benchmarks && make  # Shows detected architecture during build
```

---

**Happy Benchmarking!** 🚀

