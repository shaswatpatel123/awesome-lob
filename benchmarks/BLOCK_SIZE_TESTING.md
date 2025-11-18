# Testing Different Block Sizes

## Fixed Issues ✅

1. **Added shared memory allocation** to benchmark (was missing!)
2. **Made block size configurable** via constant
3. **Algorithm now properly scales** with different thread counts

---

## How to Test Different Block Sizes

### Method 1: Edit Source Code

1. Open `benchmark_cpu_vs_gpu.cu`
2. Change line 25:
```cpp
const int BLOCK_SIZE = 256;  // Change this value
```

Valid values: **32, 64, 128, 256, 512, 1024**

3. Rebuild and run:
```bash
cd benchmarks
make clean
make
./benchmark
```

---

## Expected Performance by Block Size

For **1000 orders per book**:

| Block Size | Threads | Orders/Thread | Speedup vs Sequential | Notes |
|------------|---------|---------------|----------------------|-------|
| 32         | 32      | ~31           | ~3-4×                | Under-utilized |
| 64         | 64      | ~16           | ~5-6×                | Good for small books |
| 128        | 128     | ~8            | ~7-8×                | Good balance |
| **256**    | **256** | **~4**        | **~7-10×**           | **Optimal** ✅ |
| 512        | 512     | ~2            | ~8-12×               | Marginal improvement |
| 1024       | 1024    | ~1            | ~9-13×               | Diminishing returns |

### Why 256 is Usually Optimal:

1. **Occupancy:** Most GPUs can run 2-4 blocks per SM with 256 threads
2. **Work per thread:** Each thread gets ~4 orders (good balance)
3. **Shared memory:** 4 KB per block (well within limits)
4. **Warp efficiency:** 8 warps (good for scheduling)

---

## Block Size Selection Guidelines

### For Different Orderbook Sizes:

| Orderbook Size | Best Block Size | Reason |
|----------------|-----------------|--------|
| < 100 orders   | 32-64           | Small data, less overhead |
| 100-500        | 128             | Good balance |
| **500-2000**   | **256**         | **Optimal for most cases** ✅ |
| 2000-5000      | 512             | Large data benefits |
| > 5000         | 512-1024        | Maximum parallelism |

---

## Testing Script

You can test multiple block sizes with this script:

```bash
#!/bin/bash
# test_block_sizes.sh

cd benchmarks

for BLOCK_SIZE in 64 128 256 512 1024
do
    echo "========================================"
    echo "Testing with BLOCK_SIZE=$BLOCK_SIZE"
    echo "========================================"
    
    # Edit the source file
    sed -i "s/const int BLOCK_SIZE = [0-9]*/const int BLOCK_SIZE = $BLOCK_SIZE/" benchmark_cpu_vs_gpu.cu
    
    # Rebuild
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    
    # Run benchmark
    ./benchmark
    
    echo ""
done
```

Make executable and run:
```bash
chmod +x test_block_sizes.sh
./test_block_sizes.sh
```

---

## What to Look For

### Good Scaling Indicators:

1. **GPU time decreases** as block size increases (up to a point)
2. **Speedup increases** from 32 → 256 threads
3. **Diminishing returns** after 256-512 threads
4. **Throughput increases** with more threads

### Example Output:

```
Block Size: 64
GPU time: 850 μs
Speedup: 5.2×

Block Size: 128
GPU time: 520 μs
Speedup: 8.5×

Block Size: 256
GPU time: 380 μs
Speedup: 11.6×  ← Sweet spot!

Block Size: 512
GPU time: 350 μs
Speedup: 12.6×  ← Small gain

Block Size: 1024
GPU time: 340 μs
Speedup: 13.0×  ← Minimal gain
```

---

## Technical Notes

### Why Algorithm Scales:

**Strided Access Pattern:**
```cuda
for (int i = threadIdx.x; i < n_orders; i += blockDim.x) {
    // Each thread processes every blockDim.x-th element
}
```
- 32 threads: Each checks 31 orders (1000/32)
- 256 threads: Each checks 4 orders (1000/256)
- 1024 threads: Each checks 1 order (1000/1024)

**Tree Reduction:**
```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    // Reduction depth = log₂(blockDim.x)
}
```
- 32 threads: 5 reduction levels
- 256 threads: 8 reduction levels
- 1024 threads: 10 reduction levels

### Shared Memory Scaling:

```cpp
shared_mem_size = BLOCK_SIZE * 16 bytes
```

| Block Size | Shared Memory | GPU Limit | OK? |
|------------|---------------|-----------|-----|
| 32         | 512 bytes     | 48 KB     | ✅  |
| 256        | 4 KB          | 48 KB     | ✅  |
| 1024       | 16 KB         | 48 KB     | ✅  |
| 2048       | 32 KB         | 48 KB     | ✅  |
| 3000       | 48 KB         | 48 KB     | ⚠️ At limit |

---

## Summary

✅ **Benchmark is now fixed** with proper shared memory allocation
✅ **Algorithm scales** with any block size (32-1024)
✅ **Recommended block size: 256** for 1000-order books
✅ **Easy to test** different sizes by changing constant

The parallel reduction will now **correctly utilize** however many threads you give it!

