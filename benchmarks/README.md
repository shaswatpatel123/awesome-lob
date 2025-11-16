# CPU vs GPU Orderbook Benchmarks

## Overview

This directory contains benchmarking tools to compare the performance of CPU sequential implementation against GPU CUDA implementation.

## Files

- `benchmark_cpu_vs_gpu.cpp` - Main benchmark comparing CPU and GPU performance
- `README.md` - This file

## Building

### Prerequisites

- CUDA Toolkit (11.0 or later)
- C++ Compiler (g++ or clang++)
- CMake (optional)

### Compile with nvcc

```bash
# From project root
nvcc -o benchmark_cpu_vs_gpu \
  benchmarks/benchmark_cpu_vs_gpu.cpp \
  src/orderbook_cpu.cpp \
  src/kernels.cu \
  src/operations.cu \
  src/utils.cu \
  -I./include \
  -std=c++14 \
  -O3 \
  -arch=sm_70  # Adjust for your GPU architecture
```

### Common GPU Architectures

- Tesla V100: `-arch=sm_70`
- RTX 2080/2080 Ti: `-arch=sm_75`
- RTX 3080/3090: `-arch=sm_86`
- RTX 4080/4090: `-arch=sm_89`

## Running Benchmarks

### Basic Usage

```bash
./benchmark_cpu_vs_gpu
```

Default parameters:
- 100 orderbooks
- 1000 messages per orderbook
- 100 orders per side
- 100 max trades

### Custom Parameters

```bash
./benchmark_cpu_vs_gpu <num_books> <messages_per_book> <orders_per_side> <max_trades>
```

Examples:

```bash
# Small workload (fast, good for testing)
./benchmark_cpu_vs_gpu 10 100 50 50

# Medium workload
./benchmark_cpu_vs_gpu 100 1000 100 100

# Large workload (stress test)
./benchmark_cpu_vs_gpu 1000 10000 200 200
```

## Interpreting Results

### Output Format

```
=== CPU Benchmark ===
CPU Time: 125.5 ms
CPU Throughput: 796812.7 messages/sec

=== GPU Benchmark ===
GPU Time: 15.2 ms
GPU Throughput: 6578947.4 messages/sec

=== Comparison ===
CPU Time: 125.5 ms
GPU Time: 15.2 ms
GPU Speedup: 8.26x
✓ GPU is 8.26x faster than CPU!
```

### Key Metrics

1. **Time (ms)**: Total time to process all messages
   - Lower is better
   - Includes initialization and processing

2. **Throughput (messages/sec)**: Processing rate
   - Higher is better
   - Formula: `(num_books × messages_per_book) / time_ms × 1000`

3. **Speedup**: GPU time / CPU time
   - >1x means GPU is faster
   - Expected: 5-15x for typical workloads
   - Higher speedup with more parallelism

### Expected Performance

| Workload | Expected Speedup |
|----------|-----------------|
| Small (10 books, 100 msgs) | 2-5x |
| Medium (100 books, 1000 msgs) | 8-12x |
| Large (1000 books, 10000 msgs) | 10-15x |

## Troubleshooting

### GPU Not Faster Than CPU

**Possible causes:**
1. **Workload too small**: GPU has overhead, needs larger workload
   - Solution: Increase number of books or messages

2. **Debug mode**: Code compiled without optimizations
   - Solution: Use `-O3` flag

3. **GPU not detected**: Running on CPU
   - Check: `nvidia-smi` to verify GPU is available

4. **Memory transfer overhead**: Data transfer dominates
   - This benchmark includes transfer time (realistic scenario)

### Compilation Errors

**"cuda_runtime.h not found":**
- Ensure CUDA toolkit is installed
- Check `nvcc --version`

**Undefined reference errors:**
- Make sure all source files are included in compilation
- Check that paths to include directories are correct

### Runtime Errors

**CUDA error: out of memory:**
- Reduce number of books or orders
- Check available GPU memory with `nvidia-smi`

**Segmentation fault:**
- Check array bounds
- Verify memory allocation succeeded

## Advanced Benchmarking

### Profile with nvprof

```bash
nvprof ./benchmark_cpu_vs_gpu 100 1000 100 100
```

### Profile with Nsight Systems

```bash
nsys profile --stats=true ./benchmark_cpu_vs_gpu 100 1000 100 100
```

### Profile with Nsight Compute

```bash
ncu --set full ./benchmark_cpu_vs_gpu 100 1000 100 100
```

## Customization

### Add New Benchmark Scenarios

Edit `benchmark_cpu_vs_gpu.cpp` to add custom message patterns:

```cpp
// Example: All CANCEL operations
std::vector<Message> generate_cancel_messages(int num_messages) {
    std::vector<Message> messages(num_messages);
    for (int i = 0; i < num_messages; i++) {
        messages[i].type = Message::CANCEL;
        messages[i].order_id = i + 1000;
        messages[i].quantity = 10;
        // ...
    }
    return messages;
}
```

### Measure Specific Operations

Modify `process_messages_batch_cpu` or kernel calls to benchmark:
- ADD-only workloads
- CANCEL-only workloads
- MATCH-heavy workloads
- Mixed workloads

## Validation

To verify CPU and GPU produce same results:

1. Run both implementations on same input
2. Compare final orderbook state
3. Compare trade records

```cpp
// Example validation
OrderbookCPU cpu_book;
OrderbookBatch gpu_batch;

// Process same messages
process_messages_sequential_cpu(cpu_book, messages, num_messages);
// ... process on GPU ...

// Compare results
bool match = compare_orderbooks_cpu(cpu_book, gpu_book);
std::cout << "Results match: " << (match ? "YES" : "NO") << std::endl;
```

## Performance Tips

### For Better GPU Performance

1. **Increase parallelism**: More orderbooks, more messages
2. **Batch processing**: Process multiple books together
3. **Optimize transfers**: Use pinned memory, streams
4. **Profile**: Identify bottlenecks with profilers

### For Fair Comparison

1. **Same algorithms**: CPU and GPU should use identical logic
2. **Optimized CPU code**: Use `-O3`, compiler optimizations
3. **Warm-up runs**: Run once before timing (GPU kernel loading)
4. **Multiple runs**: Average over 5-10 runs for stable results

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)

## Support

For issues or questions:
1. Check CUDA installation: `nvcc --version`
2. Check GPU availability: `nvidia-smi`
3. Verify code compiles without errors
4. Review benchmark output for error messages

