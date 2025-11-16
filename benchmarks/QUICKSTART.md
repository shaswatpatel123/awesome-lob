# Quick Start Guide - CPU vs GPU Benchmarks

## 1-Minute Setup

### Step 1: Navigate to benchmarks directory
```bash
cd benchmarks
```

### Step 2: Build
```bash
make
```

### Step 3: Run
```bash
make run
```

That's it! 🚀

---

## Example Output

```
=== CPU vs GPU Orderbook Benchmark ===

Configuration:
  Number of orderbooks: 100
  Messages per orderbook: 1000
  Orders per side: 100
  Max trades: 100
  Total messages: 100000

Generating test messages...

=== CPU Benchmark ===
Allocating CPU memory...
Processing messages on CPU...
CPU Time: 125.5 ms
CPU Throughput: 796812.7 messages/sec

=== GPU Benchmark ===
Allocating GPU memory...
Initializing GPU orderbooks...
Copying messages to GPU...
Warm-up run...
Processing messages on GPU...
GPU Time: 15.2 ms
GPU Throughput: 6578947.4 messages/sec

=== Comparison ===
CPU Time: 125.5 ms
GPU Time: 15.2 ms
GPU Speedup: 8.26x
✓ GPU is 8.26x faster than CPU!

=== Benchmark Complete ===
```

---

## Common Commands

```bash
# Build
make

# Clean
make clean

# Run with default params
make run

# Run small workload (fast)
make run-small

# Run medium workload
make run-medium

# Run large workload (stress test)
make run-large

# Custom parameters
./benchmark_cpu_vs_gpu 100 1000 100 100
#                      ^^^ ^^^^ ^^^ ^^^
#                       |    |    |   └─ Max trades
#                       |    |    └───── Orders per side
#                       |    └────────── Messages per book
#                       └─────────────── Number of books

# Profile with nvprof
make profile

# Profile with Nsight Systems
make profile-nsys
```

---

## Troubleshooting

### "nvcc: command not found"
```bash
# Check CUDA installation
which nvcc

# If not found, install CUDA Toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

### Wrong GPU architecture
```bash
# Check your GPU
nvidia-smi

# Build for your GPU (example: RTX 3080 = sm_86)
make GPU_ARCH=sm_86
```

### Out of memory
```bash
# Run smaller workload
./benchmark_cpu_vs_gpu 10 100 50 50
```

---

## Understanding Results

**Speedup > 10x**: Excellent! GPU is much faster ✅

**Speedup 5-10x**: Good! Expected for this workload ✅

**Speedup 2-5x**: Fair. GPU overhead is significant ⚠️

**Speedup < 2x**: Poor. Workload may be too small ❌

### Tips for Better Performance

1. **Increase workload**: More books, more messages
2. **Check GPU**: Ensure discrete GPU, not integrated
3. **Optimize build**: `-O3` flag should be present
4. **Profile**: Use `make profile` to find bottlenecks

---

## Next Steps

1. ✅ Run basic benchmark
2. Try different workload sizes
3. Profile with Nsight tools
4. Compare results across GPUs
5. Optimize bottlenecks

See `README.md` for detailed documentation.

---

**Happy Benchmarking!** 🚀

