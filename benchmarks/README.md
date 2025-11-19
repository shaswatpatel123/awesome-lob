# Orderbook Benchmarks

## Overview

This directory contains benchmarking tools for the CUDA orderbook implementation:

1. **CPU vs GPU Benchmark** - Compares CPU sequential vs GPU parallel performance
2. **Operation Timing Benchmark** - Measures timing for specific operations (ADD, MATCH, CANCEL, MARKET)

## Files

- `benchmark_cpu_vs_gpu.cu` - Main benchmark comparing CPU and GPU performance
- `benchmark_operations.cu` - Operation-specific timing benchmark
- `Makefile` - Build system with convenient targets
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

---

# Operation Timing Benchmark

## Overview

The `benchmark_operations` tool measures timing for specific orderbook operations in isolation:

1. **ADD Operations** - Non-matching limit order inserts (pure add overhead)
2. **MATCH Operations** - Limit orders that cross the spread and match
3. **CANCEL Operations** - Order cancellation by ID
4. **MARKET Operations** - Market order aggressive matching

## Building

### Using Makefile

```bash
cd benchmarks
make operations        # Build only operations benchmark
make                   # Build all benchmarks
```

### Manual Compilation

```bash
nvcc -o benchmark_operations \
  benchmark_operations.cu \
  ../src/kernels.cu \
  ../src/operations.cu \
  ../src/utils.cu \
  -I../include \
  -std=c++14 -O3 -arch=sm_80
```

## Running

### Using Makefile Targets

```bash
make run-ops              # Default: 100 books, 1000 msgs
make run-ops-small        # Small: 10 books, 100 msgs
make run-ops-medium       # Medium: 100 books, 1000 msgs
make run-ops-large        # Large: 1000 books, 10000 msgs
```

### Direct Execution

```bash
./benchmark_operations                # Default parameters
./benchmark_operations 100 1000       # Custom: 100 books, 1000 msgs/book
./benchmark_operations 1000 10000     # Large workload
```

### Help

```bash
./benchmark_operations --help
```

## Output Format

```
╔════════════════════════════════════════════════════════════════════╗
║           ORDERBOOK OPERATION TIMING BENCHMARK                     ║
╚════════════════════════════════════════════════════════════════════╝

📊 Configuration:
  Orderbooks:        100
  Messages per book: 1000
  Total messages:    100000
  Orders per side:   1000
  Max trades:        2000
  GPU block size:    256 threads

======================================================================
SCENARIO 1: ADD Operations (Non-Matching Inserts)
======================================================================

Running: ADD Operations...
  Setting up orderbook state...
  Warm-up run...
  Timed run...
  ✓ Complete

=== ADD Operations ===
Operation Type:     ADD
Messages Processed: 100000
Total Time:         15.234 ms
Time per Operation: 0.152 μs
Throughput:         6564551 ops/sec

======================================================================
SCENARIO 2: LIMIT Order Insert + Match
======================================================================

Running: LIMIT Order Insert+Match...
  Setting up orderbook state...
  Warm-up run...
  Timed run...
  ✓ Complete

=== LIMIT Order Insert+Match ===
Operation Type:     MATCH
Messages Processed: 100000
Total Time:         28.456 ms
Time per Operation: 0.285 μs
Throughput:         3514376 ops/sec

======================================================================
SCENARIO 3: CANCEL Operations
======================================================================

Running: CANCEL Operations...
  Setting up orderbook state...
  Warm-up run...
  Timed run...
  ✓ Complete

=== CANCEL Operations ===
Operation Type:     CANCEL
Messages Processed: 100000
Total Time:         12.789 ms
Time per Operation: 0.128 μs
Throughput:         7819417 ops/sec

======================================================================
SCENARIO 4: MARKET Order Insert + Match
======================================================================

Running: MARKET Order Insert+Match...
  Setting up orderbook state...
  Warm-up run...
  Timed run...
  ✓ Complete

=== MARKET Order Insert+Match ===
Operation Type:     MARKET
Messages Processed: 100000
Total Time:         32.123 ms
Time per Operation: 0.321 μs
Throughput:         3112840 ops/sec

╔════════════════════════════════════════════════════════════════════╗
║              PERFORMANCE COMPARISON SUMMARY                        ║
╚════════════════════════════════════════════════════════════════════╝

Operation                       Time (ms)     μs/op        ops/sec
-----------------------------------------------------------------------
ADD Operations                     15.234      0.152        6564551
LIMIT Order Insert+Match           28.456      0.285        3514376
CANCEL Operations                  12.789      0.128        7819417
MARKET Order Insert+Match          32.123      0.321        3112840

=== Relative Performance (normalized to fastest) ===
ADD Operations                      1.19x
LIMIT Order Insert+Match            2.22x
CANCEL Operations                   1.00x ← FASTEST
MARKET Order Insert+Match           2.51x (slower)

=== Key Insights ===
• Lower μs/op = faster operation
• Higher ops/sec = better throughput
• Relative performance shows operation cost ratios

✓ Benchmark Complete!
```

## Understanding Results

### Metrics Explained

1. **Time (ms)**: Total execution time for all operations
   - Measured using CUDA events (GPU time only, no CPU overhead)
   - Lower is better

2. **μs/op (microseconds per operation)**: Average time per single operation
   - Most useful metric for comparing operation costs
   - Formula: `(time_ms × 1000) / num_messages`

3. **ops/sec (operations per second)**: Throughput
   - Higher is better
   - Formula: `num_messages / (time_ms / 1000)`

4. **Relative Performance**: Operations compared to fastest
   - Shows which operations are more expensive
   - Example: 2.22x means operation takes 2.22× longer than fastest

### Expected Performance Hierarchy

**Typical Performance Ranking (fastest to slowest):**

1. **CANCEL** - Fastest (simple lookup + quantity update)
   - ~0.1-0.2 μs per operation
   - No matching logic, minimal state changes

2. **ADD** - Fast (find empty slot + insert)
   - ~0.15-0.25 μs per operation
   - Linear search for empty slot
   - No matching overhead (non-matching orders)

3. **LIMIT Match** - Moderate (match algorithm + partial add)
   - ~0.25-0.4 μs per operation
   - Price-time priority search
   - Trade recording
   - Remainder insertion if not fully matched

4. **MARKET** - Slowest (aggressive matching at any price)
   - ~0.3-0.5 μs per operation
   - May traverse multiple price levels
   - Multiple trade records

### Scenario Details

#### Scenario 1: ADD Operations
- **Setup**: Empty orderbook
- **Test**: Non-matching LIMIT orders (wide spread)
  - Bids at 9000-9900 (won't match asks)
  - Asks at 11000-11900 (won't match bids)
- **Measures**: Pure insertion overhead
- **Why it matters**: Baseline for order addition cost

#### Scenario 2: LIMIT Order Insert + Match
- **Setup**: Pre-populated orderbook with liquidity
  - Asks at 10050, 10060, 10070, ...
  - Bids at 9950, 9940, 9930, ...
- **Test**: LIMIT orders that cross spread
  - Buy at 10060 (matches asks)
  - Sell at 9940 (matches bids)
- **Measures**: Matching algorithm + trade recording
- **Why it matters**: Most common operation in active markets

#### Scenario 3: CANCEL Operations
- **Setup**: Orderbook populated with non-matching orders (from Scenario 1)
- **Test**: CANCEL messages for existing order IDs
- **Measures**: Order lookup + cancellation overhead
- **Why it matters**: Order modification/cancellation is frequent

#### Scenario 4: MARKET Orders
- **Setup**: Pre-populated orderbook with liquidity (same as Scenario 2)
- **Test**: MARKET orders (match at any price)
  - Buy MARKET (sweeps asks)
  - Sell MARKET (sweeps bids)
- **Measures**: Aggressive matching without price limits
- **Why it matters**: Represents urgent order execution

## Use Cases

### Performance Analysis
Compare relative costs of different operations to understand bottlenecks:
```bash
make run-ops-large
```

### Optimization Validation
Before/after comparisons to validate optimizations:
```bash
# Before optimization
./benchmark_operations 1000 10000 > before.txt

# After optimization
./benchmark_operations 1000 10000 > after.txt

# Compare
diff before.txt after.txt
```

### Hardware Comparison
Compare performance across different GPUs:
```bash
# On GPU 1
./benchmark_operations 100 1000

# On GPU 2
./benchmark_operations 100 1000
```

### Scalability Testing
Test how performance scales with workload:
```bash
for size in 10 100 1000 10000; do
  echo "=== $size books ==="
  ./benchmark_operations $size 1000
done
```

## Tips

1. **Warm-up**: Each scenario includes a warm-up run (not timed) to eliminate cold-start effects

2. **Reproducibility**: Results may vary slightly between runs due to GPU scheduling
   - Run multiple times and average for accurate measurements

3. **Interpretation**: Focus on relative performance ratios rather than absolute times
   - Ratios are more stable across different hardware

4. **Workload Size**: Larger workloads (more books/messages) provide more accurate timing
   - Small workloads may be dominated by kernel launch overhead

---

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

