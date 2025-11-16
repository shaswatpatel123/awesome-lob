# Comprehensive Test Suite

This test suite validates the CUDA Limit Order Book implementation by comparing CPU and GPU results.

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `data_generator.h` | Header for synthetic data generation |
| `data_generator.cpp` | Implementation of data generators |
| `test_suite.cu` | Main test suite with all tests |
| `Makefile_tests` | Build system for tests |
| `TEST_SUITE_README.md` | This file |

---

## 🎯 Test Organization

Tests are organized **incrementally** from simple to complex:

### **Level 1: Unit Tests** (Individual Operations)
- ✓ `unit_test_add_order` - Tests adding a single order
- ✓ `unit_test_cancel_order` - Tests canceling an order
- ✓ `unit_test_simple_match` - Tests basic order matching

### **Level 2: Integration Tests** (Known Scenarios)
- ✓ Partial Fill - Order partially matched
- ✓ No Match - Orders don't cross (spread exists)
- ✓ Price Improvement - Buyer gets better price than limit
- ✓ Cancel Test - Cancel partial quantity
- ✓ Market Order - Aggressive matching through levels
- ✓ Price-Time Priority - Same price, different times
- ✓ Multi-Level Book - Build complex orderbook

### **Level 3: Functional Tests** (CPU vs GPU Comparison)
- ✓ Small (100 messages) - Random data
- ✓ Medium (500 messages) - Random data
- ✓ Large (1000 messages) - Random data

**All tests verify:** CPU result == GPU result (EXACTLY!)

---

## 🚀 Quick Start

### Build and Run

```bash
cd tests

# Build the test suite
make -f Makefile_tests

# Run all tests
make -f Makefile_tests run
```

### Expected Output

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

------------------------------------------------------------
TEST: Unit Test: Cancel Order
------------------------------------------------------------
  ✓ PASS: Order cancelled correctly, CPU == GPU

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

## 📊 Synthetic Data Generated

### Simple Scenarios (Hardcoded)

1. **Perfect Match**
   ```
   SELL 100 @ $101.00
   BUY 100 @ $101.00
   → Expected: 1 trade, empty book
   ```

2. **Partial Fill**
   ```
   SELL 200 @ $101.00
   BUY 100 @ $101.00
   → Expected: 1 trade, 100 remaining
   ```

3. **No Match**
   ```
   SELL 100 @ $102.00
   BUY 100 @ $100.00
   → Expected: No trades, spread exists
   ```

4. **Price Improvement**
   ```
   SELL 50 @ $101.00
   SELL 30 @ $102.00
   BUY 100 @ $99.00
   BUY 60 @ $101.50  ← Crosses spread!
   → Expected: 1 trade @ $101.00 (better price!)
   ```

5. **Cancel**
   ```
   SELL 100 @ $101.00
   BUY 50 @ $99.00
   CANCEL 30 from SELL order
   → Expected: Reduced to 70 units
   ```

6. **Market Order**
   ```
   SELL 50 @ $101.00
   SELL 30 @ $102.00
   SELL 20 @ $103.00
   MARKET BUY 80
   → Expected: 2 trades, consumes first two levels
   ```

7. **Price-Time Priority**
   ```
   SELL 50 @ $101.00 at time T
   SELL 30 @ $101.00 at time T+1
   SELL 40 @ $101.00 at time T+2
   BUY 60 @ $101.00
   → Expected: Matches first order fully (50), then 10 from second
   ```

### Random Data

Generated with configurable parameters:
- 70% LIMIT orders (35% BUY, 35% SELL)
- 20% CANCEL orders
- 10% MARKET orders
- Prices: $95.00 - $105.00
- Quantities: 10-100 units
- Reproducible (seeded)

---

## 🔧 Customization

### Modify Data Generation

Edit `data_generator.cpp`:

```cpp
// Change default configuration
DataGenConfig config;
config.limit_order_pct = 0.80f;  // 80% limits
config.cancel_pct = 0.15f;        // 15% cancels
config.market_pct = 0.05f;        // 5% markets
config.mid_price = 100000;        // $100.00
config.price_range = 10000;       // ±$100 range

auto messages = generate_random_messages(1000, config);
```

### Add New Test Scenario

In `data_generator.cpp`:

```cpp
std::vector<Message> generate_my_scenario() {
    std::vector<Message> messages;
    
    // Add your messages
    messages.push_back(create_message(...));
    
    return messages;
}
```

In `test_suite.cu`:

```cpp
integration_test_scenario(stats, "My Test", generate_my_scenario);
```

### Change GPU Architecture

```bash
make -f Makefile_tests GPU_ARCH=sm_86
```

---

## 📈 What Tests Verify

### 1. **Correctness** ✓
- CPU and GPU produce **identical results**
- All orderbook operations work correctly
- Matching follows price-time priority
- Edge cases handled properly

### 2. **Consistency** ✓
- Same input → same output
- Reproducible with seeded random data
- No race conditions or non-determinism

### 3. **Completeness** ✓
- All message types tested (LIMIT, CANCEL, MARKET)
- All operations tested (add, cancel, match)
- Various scenarios covered

---

## 🐛 Debugging Failed Tests

If a test fails:

1. **Check which test failed:**
   ```
   ✗ FAIL: Integration: Price Improvement
   ```

2. **Look at the comparison output:**
   ```
   ✗ Asks mismatch at index 0
   CPU: price=101000 qty=50 id=1001
   GPU: price=102000 qty=30 id=1002
   ```

3. **Run that specific scenario manually:**
   ```cpp
   auto messages = generate_price_improvement();
   print_messages(messages, "Debug");
   // Process and inspect results
   ```

4. **Compare step-by-step:**
   - Process messages one at a time
   - Print orderbook state after each
   - Find where CPU and GPU diverge

---

## 📊 Performance Notes

**Single Orderbook Tests:**
- CPU and GPU times similar (small data, sequential processing)
- GPU overhead dominates (memory transfers, kernel launch)

**Multiple Orderbooks (in benchmarks):**
- GPU significantly faster (parallel processing)
- Speedup = N (number of orderbooks)

**This test suite focuses on CORRECTNESS, not performance!**

---

## ✅ Success Criteria

All tests should **PASS** with:
- `✓ Passed: 13` (all tests)
- `✗ Failed: 0` (no failures)
- All comparisons: `CPU == GPU`

If any test fails:
- 🐛 There's a bug in either CPU or GPU implementation
- 🔍 Use debugging steps above to isolate issue
- ⚠️ **Do not proceed** until all tests pass!

---

## 🎓 Next Steps

After all tests pass:

1. ✅ **Correctness verified** - CPU and GPU match
2. 📊 **Run benchmarks** - Test performance (see `benchmarks/`)
3. 🚀 **Scale up** - Test with larger datasets
4. 🔬 **Profile** - Use `nsys`/`ncu` to optimize
5. 🏭 **Deploy** - Use in production

---

## 📝 Files Summary

```
tests/
├── data_generator.h          ← Data generation API
├── data_generator.cpp        ← Synthetic data implementation
├── test_suite.cu             ← Main test suite
├── Makefile_tests            ← Build system
├── TEST_SUITE_README.md      ← This file
└── test_matching.cu          ← Old tests (still useful)
```

---

## 🎉 Congratulations!

You now have:
- ✅ Synthetic data generator (reproducible!)
- ✅ Comprehensive test suite (incremental!)
- ✅ CPU vs GPU verification (automatic!)
- ✅ Easy to run and extend

**Happy testing!** 🚀

