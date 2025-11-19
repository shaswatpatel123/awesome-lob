# Testing Guide for CUDA Orderbook

## 📋 Test Suite Overview

The test suite verifies all core matching engine functionality:

### Test Cases

1. **Test 1: Add Order** - Verifies orders are added correctly to the orderbook
2. **Test 2: Cancel Order** - Tests partial and full order cancellation
3. **Test 3: Simple Match** - Tests basic order matching (complete fill)
4. **Test 4: Partial Match** - Tests matching with remaining quantity
5. **Test 5: Price-Time Priority** - Verifies correct priority (price first, then time)
6. **Test 6: Market Order** - Tests aggressive market order execution

---

## 🚀 Quick Start (Cloud GPU)

### Option 1: Automated Build and Test (Recommended)

```bash
# On your cloud GPU machine
cd /path/to/refector/tests

# Make script executable
chmod +x build_and_test.sh

# Run everything
./build_and_test.sh
```

That's it! The script will:
- ✅ Build the CUDA orderbook library
- ✅ Compile the test program
- ✅ Run all 6 tests
- ✅ Show pass/fail summary

---

### Option 2: Manual Build and Test

```bash
# Step 1: Build library
cd /path/to/refector
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Step 2: Compile test
cd ../tests
nvcc -arch=sm_52 \
     -I../include \
     -L../build \
     -lcuda_orderbook \
     test_matching.cu \
     -o test_matching

# Note: Adjust -arch=sm_XX for your GPU:
# - GTX TITAN X (Maxwell): sm_52
# - T4: sm_75
# - V100: sm_70
# - A100: sm_80
# - RTX 30xx: sm_86
# - RTX 40xx: sm_89

# Step 3: Run tests
export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH
./test_matching
```

---

## 📊 Expected Output

### Successful Run

```
=========================================================
CUDA ORDERBOOK MATCHING ENGINE TEST SUITE
=========================================================

Using GPU: Tesla T4
Compute Capability: 7.5

============================================================
TEST 1: Add Order
============================================================

=== Orderbook State ===

Asks (Sell Orders):
     Price  Quantity   OrderID  TraderID
----------------------------------------
(empty)

Bids (Buy Orders):
     Price  Quantity   OrderID  TraderID
----------------------------------------
    100000       100      1001         1

=== Trades ===
(no trades)

✅ PASS: Order added correctly

[... more tests ...]

============================================================
TEST SUMMARY
============================================================

Tests Passed: 6/6

🎉 ALL TESTS PASSED! 🎉
Matching engine is working correctly!
```

### Failed Test Example

If a test fails, you'll see detailed output:

```
============================================================
TEST 3: Simple Match
============================================================

=== Orderbook State ===
[... orderbook state ...]

❌ FAIL: No trade generated

============================================================
TEST SUMMARY
============================================================

Tests Passed: 5/6

❌ SOME TESTS FAILED
Please review failed tests above
```

---

## 🔧 Troubleshooting

### Error: "CUDA Error: no kernel image is available"

**Problem:** Wrong CUDA architecture specified

**Solution:** Find your GPU architecture:
```bash
# Check GPU model
nvidia-smi --query-gpu=name --format=csv,noheader

# Common architectures:
# GTX TITAN X (Maxwell) → sm_52
# Tesla T4 → sm_75
# Tesla V100 → sm_70
# A100 → sm_80
# RTX 3090 → sm_86
```

Recompile with correct architecture:
```bash
nvcc -arch=sm_52 ...  # Replace 52 with your architecture (52, 60, 61, 70, 75, 80, 86, 89)
```

### Error: "cannot find -lcuda_orderbook"

**Problem:** Library not built or wrong path

**Solution:**
```bash
# Rebuild library
cd ../build
make clean
make -j$(nproc)

# Verify library exists
ls -lh libcuda_orderbook.a
```

### Error: "undefined reference to" during linking

**Problem:** Library and test out of sync

**Solution:** Clean rebuild everything:
```bash
cd /path/to/refector
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ../tests
# Recompile test
```

### Compilation Warning: "Wno-deprecated-gpu-targets"

**Safe to ignore** - Just means you're using an older GPU architecture.

---

## 🧪 Adding Your Own Tests

### Test Template

```cpp
bool test_my_feature() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST: My Feature" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    // Create messages
    Message msg;
    msg.type = Message::LIMIT;
    msg.side = Message::BID;
    msg.quantity = 100;
    msg.price = 100000;
    msg.order_id = 1001;
    msg.trader_id = 1;
    msg.time_sec = 34200;
    msg.time_ns = 0;
    
    // Process
    ob.process_messages(&msg, 1);
    ob.print();
    
    // Verify
    const Order* bids = ob.get_bids();
    if (bids[0].price != 100000) {
        std::cout << "❌ FAIL: ..." << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: ..." << std::endl;
    return true;
}

// Add to main():
total++; if (test_my_feature()) passed++;
```

---

## 📈 Performance Testing

### Batch Performance Test

Create `test_performance.cu`:

```cpp
#include "types.h"
#include "kernels.cuh"
#include <iostream>
#include <chrono>

using namespace cuda_orderbook;

int main() {
    int num_books = 10000;  // Test with 10k orderbooks
    int n_orders = 100;
    int n_trades = 100;
    
    // Setup batch
    OrderbookBatch batch;
    batch.num_books = num_books;
    batch.n_orders_per_book = n_orders;
    batch.n_trades_per_book = n_trades;
    
    // Allocate
    cudaMalloc(&batch.d_asks, num_books * n_orders * sizeof(Order));
    cudaMalloc(&batch.d_bids, num_books * n_orders * sizeof(Order));
    cudaMalloc(&batch.d_trades, num_books * n_trades * sizeof(Trade));
    
    // Initialize
    dim3 grid(num_books);
    dim3 block(256);
    
    auto start = std::chrono::high_resolution_clock::now();
    init_orderbooks_kernel<<<grid, block>>>(batch, num_books);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Initialized " << num_books << " orderbooks in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Throughput: " << (num_books * 1000000.0 / duration.count()) 
              << " orderbooks/sec" << std::endl;
    
    // Cleanup
    cudaFree(batch.d_asks);
    cudaFree(batch.d_bids);
    cudaFree(batch.d_trades);
    
    return 0;
}
```

Compile and run:
```bash
nvcc -arch=sm_52 -I../include -L../build -lcuda_orderbook test_performance.cu -o test_perf
./test_perf
```

---

## ✅ What Tests Verify

### Correctness
- ✅ Orders added to correct side (ask/bid)
- ✅ Cancellations reduce quantity correctly
- ✅ Matches generate trades
- ✅ Trade IDs (passive/aggressive) are correct
- ✅ Price-time priority enforced
- ✅ Partial fills handled
- ✅ Market orders execute aggressively
- ✅ Orderbook state cleaned up (zero-quantity removed)

### JAX Compatibility
All tests match expected JAX behavior. To validate against actual JAX:

1. Export JAX orderbook state (see TEAM2_MATCHING_COMPLETE.md)
2. Load in C++ test
3. Compare array contents

---

## 📞 Getting Help

If tests fail:

1. **Check GPU architecture** - Most common issue
2. **Clean rebuild** - `rm -rf build && mkdir build && ...`
3. **Review test output** - Shows exactly what failed
4. **Check CUDA device** - `nvidia-smi` to verify GPU accessible
5. **Library path** - Ensure `LD_LIBRARY_PATH` set correctly

---

## 🎯 Success Criteria

**All 6 tests must pass** before moving to integration!

✅ Test 1: Add Order  
✅ Test 2: Cancel Order  
✅ Test 3: Simple Match  
✅ Test 4: Partial Match  
✅ Test 5: Price-Time Priority  
✅ Test 6: Market Order  

When all pass: **Matching engine is verified! 🎉**

