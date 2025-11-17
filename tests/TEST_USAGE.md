# Test Suite Usage Guide

## Quick Start

```bash
# Run all tests with default settings
./test_suite

# Show help
./test_suite --help
```

## 🎯 Specify Message Sizes

Run functional tests with custom message counts:

```bash
# Test with 100 messages only
./test_suite 100

# Test with multiple message sizes
./test_suite 100 500 1000

# Test with large sizes
./test_suite 1000 5000 10000 50000
```

## 🔧 Control Which Tests Run

### Run Specific Test Levels

```bash
# Only unit tests (fast)
./test_suite --unit-only

# Only integration tests
./test_suite --integration-only

# Only functional tests with custom sizes
./test_suite --functional-only 1000 10000
```

### Skip Test Levels

```bash
# Skip unit tests
./test_suite --skip-unit 100 1000

# Skip functional tests (run only unit + integration)
./test_suite --skip-functional

# Run only integration tests
./test_suite --skip-unit --skip-functional
```

## 📊 Control Orderbook Configuration

### Number of Parallel Orderbooks

```bash
# Test with 100 parallel orderbooks (default: 1)
./test_suite --num-books 100 1000

# Test massive parallelism (1000 orderbooks!)
./test_suite --num-books 1000 10000

# Realistic trading scenario (100 books, 10k msgs each)
./test_suite --functional-only --num-books 100 10000
```

### Orderbook Size (Orders per Side)

```bash
# Set max orders per side (default: 1000)
./test_suite --max-orders 500 1000 5000

# Large orderbooks with many messages
./test_suite --max-orders 2000 10000 50000
```

## 🚀 Common Usage Patterns

### Quick Smoke Test
```bash
# Fast test with small sizes
./test_suite 100
```

### Standard Testing
```bash
# Default: all tests, standard sizes
./test_suite
```

### Performance Testing (Single Orderbook)
```bash
# Focus on functional tests with large sizes
./test_suite --functional-only 1000 5000 10000 50000 100000
```

### Parallel Performance Testing (MAIN USE CASE!)
```bash
# Test parallel scalability with 100 orderbooks
./test_suite --functional-only --num-books 100 1000 10000

# Realistic HFT scenario: 1000 parallel orderbooks
./test_suite --functional-only --num-books 1000 10000

# Massive scale test: 10,000 orderbooks
./test_suite --functional-only --num-books 10000 1000
```

### Development Testing
```bash
# Skip slow functional tests during development
./test_suite --skip-functional
```

### HPC Resource-Constrained
```bash
# Test with smaller orderbooks to fit memory
./test_suite --max-orders 100 500 1000
```

## 📋 Examples

### Example 1: Quick Validation
```bash
$ ./test_suite 100
Test Configuration:
  Unit tests: YES
  Integration tests: YES
  Functional tests: YES
  Number of orderbooks: 1
  Message sizes: 100
  Max orders per side: 1000
```

### Example 2: Parallel Performance Test
```bash
$ ./test_suite --functional-only --num-books 100 1000
Test Configuration:
  Unit tests: NO
  Integration tests: NO
  Functional tests: YES
  Number of orderbooks: 100
  Message sizes: 1000
  Max orders per side: 1000

Output:
  Testing with 1000 random messages/book, 100 orderbook(s)...
  Orders per side: 200, Max trades: 100
  CPU time: 125000 μs
  GPU time: 8500 μs
  Speedup (parallel): 1470.59x   ← GPU processes 100 books in parallel!
  Throughput: 11764.7 msgs/ms
  ✓ PASS: CPU == GPU
```

### Example 3: Massive Scale Test
```bash
$ ./test_suite --functional-only --num-books 1000 10000
Test Configuration:
  Unit tests: NO
  Integration tests: NO
  Functional tests: YES
  Number of orderbooks: 1000
  Message sizes: 10000
  Max orders per side: 1000

This tests: 1000 orderbooks × 10000 messages = 10 MILLION messages!
```

### Example 4: Memory-Constrained Testing
```bash
$ ./test_suite --max-orders 200 100 500 1000
Test Configuration:
  Unit tests: YES
  Integration tests: YES
  Functional tests: YES
  Number of orderbooks: 1
  Message sizes: 100, 500, 1000
  Max orders per side: 200
```

## 🎓 Default Behavior

When run without arguments:
- ✅ Runs all test levels (unit, integration, functional)
- ✅ Message sizes: 100, 500, 1000, 5000, 10000
- ✅ Max orders per side: 1000

## 📊 Test Levels Explained

### Unit Tests (Fixed Size)
- 3 tests
- 100 orders/side, 50 trades
- Tests: Add, Cancel, Simple Match
- **Duration:** ~1 second

### Integration Tests (Fixed Size)
- 7 tests
- 100 orders/side, 50 trades
- Tests: Various known scenarios
- **Duration:** ~5 seconds

### Functional Tests (Variable Size)
- Number of tests = number of message sizes specified
- Orderbook size scales with messages or uses --max-orders
- Random data, CPU vs GPU comparison
- **Duration:** Varies (seconds to minutes)

## 🔍 Memory Usage & Orderbook Sizing

### Automatic Sizing Formula

The test suite uses **10% sizing** (matching benchmarks and realistic market depth):

```
orders_per_side = max(100, num_messages / 10)   // 10% ratio (realistic)
trades = max(100, num_messages / 10)            // 10% of messages
```

**Why 10%?** Matches real market behavior and benchmark configuration. Most messages match and don't stay in the book.

### Memory Usage Table

| Messages | Orders/Side | Trades | Memory/Book | 100 Books | 1000 Books | 10000 Books |
|----------|-------------|--------|-------------|-----------|------------|-------------|
| 100      | 100         | 100    | ~15 KB      | ~1.5 MB   | ~15 MB     | ~150 MB     |
| 500      | 100         | 100    | ~15 KB      | ~1.5 MB   | ~15 MB     | ~150 MB     |
| 1000     | 100         | 100    | ~15 KB      | ~1.5 MB   | ~15 MB     | ~150 MB     |
| 5000     | 500         | 500    | ~75 KB      | ~7.5 MB   | ~75 MB     | ~750 MB     |
| 10000    | 1000        | 1000   | ~150 KB     | ~15 MB    | ~150 MB    | ~1.5 GB     |
| 50000    | 5000        | 5000   | ~750 KB     | ~75 MB    | ~750 MB    | ~7.5 GB     |

**Note:** Total workload = `num_books × messages_per_book` messages processed in parallel!

### Utilization Monitoring

The test suite monitors orderbook utilization:

- ✅ **<75% used:** Good, orderbook sized appropriately
- ⚠️ **75-90% used:** High utilization, working well
- 🚨 **>90% used:** FULL! Orders may be dropped - use `--max-orders` to increase size

## 🐛 Troubleshooting

### Orders Being Dropped (>90% Utilization)

**With 10% default sizing, orders CAN be dropped in extreme cases.**

If you see this warning:
```
⚠️  WARNING: Orderbook >90% full! Orders may have been dropped!
```

**Cause:** Unusual message pattern (too many LIMIT orders that don't match).

**Solution:** Increase orderbook size:
```bash
# Increase to 20% or 50% of messages
./test_suite --max-orders 2000 10000  # 20% for 10k messages

# Or go 1:1 for guaranteed safety
./test_suite --max-orders 10000 10000
```

### Test Fails with "CPU != GPU"

**Possible causes:**
1. Orders dropped due to full orderbook → Check utilization warning
2. Actual bug in CPU/GPU logic

**Solution:**
```bash
# First try increasing orderbook size
./test_suite --max-orders 5000 10000

# If still fails with large orderbook, likely a real bug
```

### Out of Memory
```bash
# Reduce orderbook size or number of books
./test_suite --max-orders 100 --num-books 10 1000
```

### Too Slow
```bash
# Test fewer/smaller message sizes
./test_suite 100 500

# Or skip unit/integration tests
./test_suite --functional-only 1000
```

### Compilation Error After Changes
```bash
make -f Makefile_tests clean
make -f Makefile_tests
./test_suite
```

