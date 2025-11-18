# Complete Usage Guide - Test Suite & Benchmark

## 🧪 Test Suite

### Build
```bash
cd tests
make -f Makefile_tests clean
make -f Makefile_tests
```

### Command Format
```bash
./test_suite [OPTIONS] [MESSAGE_SIZES...]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MESSAGE_SIZES` | Numbers | 100, 500, 1000, 5000, 10000 | Number of messages to test |
| `--num-books N` | Option | 1 | Number of orderbooks (parallel) |
| `--max-orders N` | Option | 1000 | Max orders per side |
| `--unit-only` | Flag | - | Run only unit tests |
| `--integration-only` | Flag | - | Run only integration tests |
| `--functional-only` | Flag | - | Run only functional tests |
| `--skip-unit` | Flag | - | Skip unit tests |
| `--skip-functional` | Flag | - | Skip functional tests |

**Note:** Test suite uses **hardcoded 256 threads** per block (optimal for most cases)

### Examples

#### 1. Quick Test
```bash
./test_suite 100
# Tests: 100 messages, 1 orderbook, 256 threads
```

#### 2. Multiple Message Sizes
```bash
./test_suite 100 500 1000 5000
# Tests each size sequentially
```

#### 3. Parallel Orderbooks (Main Use Case!)
```bash
./test_suite --functional-only --num-books 100 1000
# 100 orderbooks processing 1000 messages each IN PARALLEL
# Shows GPU parallelism benefits
```

#### 4. Massive Scale
```bash
./test_suite --functional-only --num-books 1000 10000
# 1000 orderbooks × 10,000 messages = 10 MILLION messages!
```

#### 5. Custom Orderbook Size
```bash
./test_suite --max-orders 500 100 1000
# Smaller orderbooks (500 orders/side)
```

#### 6. Development - Fast Tests
```bash
./test_suite --skip-functional
# Skip slow functional tests, run unit + integration only
```

---

## 🚀 Benchmark

### Build
```bash
cd benchmarks
make clean
make
```

### Command Format
```bash
./benchmark [num_books] [messages] [orders] [trades] [block_size]
```

### Parameters

| Position | Parameter | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| 1 | `num_books` | Number | 100 | Number of orderbooks to process in parallel |
| 2 | `messages` | Number | 1000 | Messages per orderbook |
| 3 | `orders` | Number | 100 | Max orders per side |
| 4 | `trades` | Number | 100 | Max trades |
| 5 | `block_size` | Number | 256 | GPU threads per block (32/64/128/256/512/1024) |

### Examples

#### 1. Default Configuration
```bash
./benchmark
# 100 books, 1000 messages, 100 orders, 100 trades, 256 threads
```

#### 2. Large Scale Test
```bash
./benchmark 1000 10000
# 1000 orderbooks, 10,000 messages each
# Uses defaults: 100 orders, 100 trades, 256 threads
```

#### 3. Custom Orderbook Size
```bash
./benchmark 100 5000 500 500
# 100 books, 5000 messages, 500 orders, 500 trades
# Uses default 256 threads
```

#### 4. Test Different Block Sizes
```bash
# Small block size
./benchmark 100 1000 100 100 64
# 64 threads per block

# Medium block size
./benchmark 100 1000 100 100 128
# 128 threads per block

# Optimal (default)
./benchmark 100 1000 100 100 256
# 256 threads per block ✅

# Large block size
./benchmark 100 1000 100 100 512
# 512 threads per block

# Maximum
./benchmark 100 1000 100 100 1024
# 1024 threads per block
```

#### 5. Block Size Comparison Script
```bash
#!/bin/bash
# compare_block_sizes.sh

echo "Testing different block sizes..."
echo "================================"

for BLOCK_SIZE in 64 128 256 512 1024
do
    echo ""
    echo "Block Size: $BLOCK_SIZE threads"
    echo "--------------------------------"
    ./benchmark 100 1000 100 100 $BLOCK_SIZE | grep -E "GPU Time|Speedup|Throughput"
done
```

#### 6. Scaling Test
```bash
# Test how performance scales with number of orderbooks
for N in 10 100 500 1000
do
    echo "Testing $N orderbooks..."
    ./benchmark $N 1000
done
```

#### 7. Help
```bash
./benchmark --help
# Shows full usage information
```

---

## 📊 Comparing Test Suite vs Benchmark

| Feature | Test Suite | Benchmark |
|---------|------------|-----------|
| **Purpose** | Correctness testing (CPU == GPU) | Performance measurement (CPU vs GPU) |
| **Block Size** | Hardcoded 256 | Configurable (32-1024) |
| **Parallelism** | --num-books flag | First argument |
| **Output** | ✓ PASS/FAIL | Time & Speedup |
| **Message Sizes** | Multiple sizes per run | Single size per run |
| **Use When** | Verifying correctness | Measuring performance |

---

## 🎯 Recommended Testing Workflow

### Step 1: Verify Correctness
```bash
cd tests
make -f Makefile_tests
./test_suite --functional-only 1 1000
# Should show: ✓ PASS: CPU == GPU for both tests
```

### Step 2: Test Parallel Scaling
```bash
./test_suite --functional-only --num-books 100 1000
# Should show significant speedup (>10x)
```

### Step 3: Benchmark Performance
```bash
cd ../benchmarks
make
./benchmark 100 1000
# Compare CPU vs GPU times
```

### Step 4: Test Block Size Scaling (Optional)
```bash
./benchmark 100 1000 100 100 64
./benchmark 100 1000 100 100 128
./benchmark 100 1000 100 100 256   # Should be fastest
./benchmark 100 1000 100 100 512
./benchmark 100 1000 100 100 1024
```

---

## 📈 Expected Results

### Test Suite
```
TEST: Functional Test: Random Test (1 messages)
  ✓ PASS: CPU == GPU

TEST: Functional Test: Random Test (1000 messages)
  ✓ PASS: CPU == GPU
  
TEST SUMMARY
✓ Passed: 2
✗ Failed: 0
```

### Benchmark (100 books, 1000 messages, 256 threads)
```
=== Comparison ===
CPU Time: 250.5 ms
GPU Time: 18.3 ms
Speedup: 13.7x
Throughput: 5464.5 messages/ms
```

### Block Size Scaling (Expected Pattern)
```
Block Size: 64  threads → Speedup: ~5-6×
Block Size: 128 threads → Speedup: ~7-8×
Block Size: 256 threads → Speedup: ~10-15× ✅ OPTIMAL
Block Size: 512 threads → Speedup: ~11-16× (marginal gain)
Block Size: 1024 threads → Speedup: ~12-17× (diminishing returns)
```

---

## 🔧 Troubleshooting

### Test Suite Fails
```bash
# If CPU != GPU:
./test_suite --max-orders 2000 1000  # Increase orderbook size
```

### Benchmark Crashes
```bash
# If out of memory:
./benchmark 10 1000     # Reduce number of books
./benchmark 100 1000 50 50  # Reduce orderbook size
```

### Block Size Issues
```bash
# Only use valid values: 32, 64, 128, 256, 512, 1024
./benchmark 100 1000 100 100 256  # ✅ Valid
./benchmark 100 1000 100 100 200  # ❌ Invalid - will error
```

---

## 💡 Pro Tips

1. **Test Suite:** Use `--num-books` to see GPU parallel benefits
2. **Benchmark:** 256 threads is optimal for most orderbook sizes
3. **Large Scale:** Start with small numbers and scale up
4. **Block Size:** Test 256 first, then try 128/512 if curious
5. **Memory:** Larger orderbooks need more GPU memory

---

## 📝 Quick Reference

### Most Common Commands

**Correctness Test:**
```bash
cd tests && ./test_suite --functional-only 1 1000
```

**Performance Test:**
```bash
cd benchmarks && ./benchmark 100 1000
```

**Parallel Scaling Test:**
```bash
cd tests && ./test_suite --functional-only --num-books 100 1000
```

**Block Size Test:**
```bash
cd benchmarks && ./benchmark 100 1000 100 100 512
```

---

## Summary Table

| What You Want | Command |
|---------------|---------|
| Quick correctness check | `./test_suite 100` |
| Full correctness test | `./test_suite` |
| Parallel GPU benefits | `./test_suite --num-books 100 1000` |
| Performance benchmark | `./benchmark` |
| Large scale test | `./benchmark 1000 10000` |
| Test block size scaling | `./benchmark 100 1000 100 100 512` |
| Fast dev testing | `./test_suite --skip-functional` |

**All commands assume you've built the executables first!**

