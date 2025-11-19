# Operation Timing Benchmark - Quick Start Guide

## Overview

This benchmark isolates and measures timing for four key orderbook operations:

1. **ADD** - Insert non-matching orders (pure insertion overhead)
2. **MATCH** - Limit orders that cross spread (matching + trade recording)
3. **CANCEL** - Order cancellation by ID (lookup + removal)
4. **MARKET** - Market orders (aggressive matching at any price)

## Quick Start

```bash
cd benchmarks
make operations      # Build
make run-ops         # Run with defaults (100 books, 1000 msgs)
```

## What It Does

### Scenario 1: ADD Operations
- **Empty orderbook** → Insert non-matching LIMIT orders
- Creates wide spread: bids @9000-9900, asks @11000-11900
- **Measures**: Pure order insertion (find slot + add)
- **Timed**: The insertion phase itself

### Scenario 2: LIMIT Order Match
- **Pre-populated orderbook** with liquidity (asks @10050+, bids @9950-)
- Insert LIMIT orders that cross spread
- **Measures**: Matching algorithm + trade generation + remainder insertion
- **Timed**: Only the matching orders (not the setup)

### Scenario 3: CANCEL Operations
- **Pre-populated orderbook** from Scenario 1 (reused, not re-timed)
- Send CANCEL messages for existing order IDs
- **Measures**: Order lookup + quantity reduction + cleanup
- **Timed**: Only the cancel operations

### Scenario 4: MARKET Orders
- **Pre-populated orderbook** with liquidity (same as Scenario 2)
- Send MARKET orders (match at any price)
- **Measures**: Aggressive matching without price constraints
- **Timed**: Only the market orders (not the setup)

## Key Design Points

### Why Pre-populate?
- **Realistic**: Real orderbooks aren't empty
- **Isolation**: Separates setup cost from operation cost
- **Reusability**: Scenario 1 ADD timing can be used for other scenarios
- **Accuracy**: Only times the specific operation being measured

### Timing Method
- **CUDA Events**: GPU-side timing (no CPU overhead)
- **Warm-up Run**: First run excluded (eliminates cold start)
- **Kernel-level**: Times entire kernel execution (can't time device functions separately)

### What's NOT Timed
- Memory allocation
- Data transfer (CPU→GPU)
- Kernel launch overhead (minimized via warm-up)
- Setup/pre-population for scenarios 2-4

## Expected Results

### Performance Hierarchy (Typical)
1. **CANCEL**: ~0.1-0.2 μs/op (fastest - simple lookup)
2. **ADD**: ~0.15-0.25 μs/op (fast - find slot + insert)
3. **LIMIT Match**: ~0.25-0.4 μs/op (moderate - matching + trades)
4. **MARKET**: ~0.3-0.5 μs/op (slowest - sweeps multiple levels)

### Sample Output
```
=== CANCEL Operations ===
Time per Operation: 0.128 μs
Throughput:         7,819,417 ops/sec

=== ADD Operations ===
Time per Operation: 0.152 μs
Throughput:         6,564,551 ops/sec

=== LIMIT Order Insert+Match ===
Time per Operation: 0.285 μs
Throughput:         3,514,376 ops/sec

=== MARKET Order Insert+Match ===
Time per Operation: 0.321 μs
Throughput:         3,112,840 ops/sec

Relative Performance:
  CANCEL: 1.00x (fastest)
  ADD: 1.19x
  LIMIT: 2.22x
  MARKET: 2.51x (slowest)
```

## Use Cases

### 1. Bottleneck Analysis
Identify which operations dominate your workload:
```bash
make run-ops-large
# If MATCH is slowest and your workload is match-heavy, optimize matching
```

### 2. Optimization Validation
Compare before/after changes:
```bash
./benchmark_operations 1000 10000 > baseline.txt
# Make code changes
./benchmark_operations 1000 10000 > optimized.txt
diff baseline.txt optimized.txt
```

### 3. Hardware Comparison
Compare different GPUs:
```bash
# Run on each GPU
./benchmark_operations 100 1000
```

### 4. Regression Testing
Ensure changes don't slow down operations:
```bash
# Add to CI/CD
./benchmark_operations 100 1000 | grep "Time per Operation"
```

## Interpreting Results

### Metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| Time (ms) | Total execution time | Lower |
| μs/op | Microseconds per operation | Lower |
| ops/sec | Operations per second | Higher |
| Relative | Ratio to fastest operation | Lower |

### What Influences Performance?

**ADD Operations:**
- Number of existing orders (affects empty slot search)
- Memory layout (cache effects)

**MATCH Operations:**
- Number of price levels to traverse
- Number of trades generated
- Parallel reduction efficiency (GPU block size)

**CANCEL Operations:**
- Number of orders (affects linear search by ID)
- *Should be fastest* - if not, investigate

**MARKET Operations:**
- Depth of orderbook (how many levels to sweep)
- Similar to MATCH but typically more matches

## Common Patterns

### CANCEL Fastest
✓ Expected - simple operation with minimal state changes

### MARKET Slowest
✓ Expected - may traverse multiple price levels

### LIMIT Slower Than ADD
✓ Expected - includes matching overhead

### ADD Unexpectedly Slow
⚠ Investigate:
- Are orderbooks getting full? (affects empty slot search)
- Memory fragmentation?

### CANCEL Unexpectedly Slow
⚠ Investigate:
- Linear search overhead (consider hash table)
- Many cancelled orders?

## Advanced Usage

### Profile a Specific Operation
```bash
nsys profile --stats=true ./benchmark_operations 100 1000
# Look for kernel launch times by scenario
```

### Scale Testing
```bash
for books in 10 100 1000; do
  echo "=== $books books ==="
  ./benchmark_operations $books 1000 | grep "Time per Operation"
done
```

### Export Results
```bash
./benchmark_operations 1000 10000 > results_$(date +%Y%m%d).txt
```

## Limitations

### Can't Time Individual Device Functions
CUDA events work at kernel-launch level, not within device code.
- Can't separately time `add_order_device()`, `cancel_order_device()`, etc.
- Can only time entire kernel execution

### Operations Aren't Perfectly Isolated
- LIMIT orders include both matching AND insertion (if remainder)
- Can't separate the two within one kernel launch

### Hardware Dependent
- Absolute times vary by GPU
- Focus on relative performance ratios

### Workload Dependent
- Results depend on orderbook depth, message mix
- Provided scenarios are representative but not exhaustive

## Troubleshooting

### Build Errors
```bash
# Check CUDA is available
nvcc --version

# Adjust GPU architecture in Makefile
# Edit benchmarks/Makefile, line: GPU_ARCH = sm_XX
make clean
make operations
```

### Runtime Errors
```bash
# Out of memory
./benchmark_operations 10 100  # Try smaller workload

# Incorrect results
./benchmark_operations --help  # Check parameters
```

### Unexpected Performance
1. Check GPU is being used: `nvidia-smi`
2. Ensure optimization flags: `make` uses `-O3`
3. Close other GPU applications
4. Run multiple times to verify consistency

## Summary

This benchmark provides:
- ✓ Isolated timing for each operation type
- ✓ GPU-side measurement (CUDA events)
- ✓ Pre-populated realistic orderbook states
- ✓ Warm-up runs to eliminate cold starts
- ✓ Clear comparison metrics

Use it to:
- Identify performance bottlenecks
- Validate optimizations
- Compare hardware
- Understand operation costs

For more details, see `benchmarks/README.md`.

