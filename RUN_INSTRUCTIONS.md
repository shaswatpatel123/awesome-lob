# 🚀 COMPLETE RUN INSTRUCTIONS

**Follow these steps IN ORDER to build and test the CUDA Orderbook system.**

---

## ⚠️ Prerequisites

Before starting, ensure you have:
- ✅ CUDA Toolkit installed (`nvcc --version` should work)
- ✅ NVIDIA GPU available
- ✅ You're in the project root directory: `awesome-lob/`

---

## 📋 STEP-BY-STEP INSTRUCTIONS

### **STEP 1: Clean Everything (Start Fresh)**

```bash
# Navigate to project root
cd /Users/kvlnraju/Desktop/courses/semester_3/GPU/Project/awesome-lob

# Clean any previous builds
cd build
make clean 2>/dev/null || true
cd ..

cd tests
make -f Makefile_tests clean 2>/dev/null || true
cd ..

cd benchmarks
make clean 2>/dev/null || true
cd ..

echo "✓ Clean complete"
```

---

### **STEP 2: Build the Core Library**

```bash
# Navigate to build directory
cd build

# Configure with CMake
cmake ..

# Build the library
make

# Verify build succeeded
ls libcuda_orderbook.a

echo "✓ Core library built successfully"
cd ..
```

**Expected output:**
```
[ 33%] Building CUDA object CMakeFiles/cuda_orderbook.dir/src/kernels.cu.o
[ 66%] Building CUDA object CMakeFiles/cuda_orderbook.dir/src/operations.cu.o
[100%] Linking CUDA static library libcuda_orderbook.a
```

---

### **STEP 3: Build the Test Suite**

```bash
# Navigate to tests directory
cd tests

# Build test suite
make -f Makefile_tests

# Verify test binary exists
ls test_suite

echo "✓ Test suite built successfully"
```

**Expected output:**
```
Compiling test suite...
Build complete: test_suite
```

---

### **STEP 4: Run the Test Suite** ⭐ **MOST IMPORTANT**

```bash
# Still in tests directory
# Run all tests (13 tests total)
./test_suite

# Or use make run
make -f Makefile_tests run
```

**Expected output:**
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

------------------------------------------------------------
TEST: Unit Test: Simple Match
------------------------------------------------------------
  ✓ PASS: Perfect match executed, CPU == GPU

============================================================
LEVEL 2: INTEGRATION TESTS (Known Scenarios)
============================================================

------------------------------------------------------------
TEST: Integration: Partial Fill
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: No Match
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: Price Improvement
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: Cancel Test
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: Market Order
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: Price-Time Priority
------------------------------------------------------------
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Integration: Multi-Level Book
------------------------------------------------------------
  ✓ PASS: CPU == GPU

============================================================
LEVEL 3: FUNCTIONAL TESTS (Random Data, CPU vs GPU)
============================================================

------------------------------------------------------------
TEST: Functional Test: Random Small (100 messages)
------------------------------------------------------------
  Testing with 100 random messages...
  CPU time: 125 μs
  GPU time: 98 μs
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Functional Test: Random Medium (500 messages)
------------------------------------------------------------
  Testing with 500 random messages...
  CPU time: 612 μs
  GPU time: 487 μs
  ✓ PASS: CPU == GPU

------------------------------------------------------------
TEST: Functional Test: Random Large (1000 messages)
------------------------------------------------------------
  Testing with 1000 random messages...
  CPU time: 1234 μs
  GPU time: 976 μs
  ✓ PASS: CPU == GPU

============================================================
TEST SUMMARY
============================================================
Total tests: 13
✓ Passed: 13
✗ Failed: 0

🎉 ALL TESTS PASSED!
============================================================
```

**⚠️ CRITICAL:** If ANY test fails, **STOP HERE** and debug before proceeding!

---

### **STEP 5: Build Benchmarks (Optional, After Tests Pass)**

```bash
# Navigate to benchmarks directory
cd ../benchmarks

# Build benchmark
make

# Verify benchmark binary exists
ls benchmark_cpu_vs_gpu

echo "✓ Benchmarks built successfully"
```

---

### **STEP 6: Run Benchmarks (Optional)**

```bash
# Still in benchmarks directory

# Run small workload
make run-small

# Run medium workload
make run-medium

# Run large workload (if you have time and GPU memory)
make run-large
```

**Expected output:**
```
Running benchmark with default parameters...
============================================================
CPU vs GPU Orderbook Benchmark
============================================================

Configuration:
  Number of orderbooks: 100
  Messages per orderbook: 1000
  Orders per side: 100
  Max trades: 100

[... timing results ...]

Performance Summary:
  CPU throughput: 50,000 ops/sec
  GPU throughput: 500,000 ops/sec
  Speedup: 10.0x
```

---

## 🎯 QUICK COMMAND SUMMARY

**Copy-paste this entire block to run everything:**

```bash
#!/bin/bash

echo "========================================="
echo "CUDA Orderbook - Complete Build & Test"
echo "========================================="

# Step 1: Clean
echo -e "\n[1/6] Cleaning previous builds..."
cd /Users/kvlnraju/Desktop/courses/semester_3/GPU/Project/awesome-lob
cd build && make clean 2>/dev/null; cd ..
cd tests && make -f Makefile_tests clean 2>/dev/null; cd ..
cd benchmarks && make clean 2>/dev/null; cd ..

# Step 2: Build core library
echo -e "\n[2/6] Building core library..."
cd build
cmake .. && make
if [ $? -ne 0 ]; then
    echo "❌ Core library build FAILED"
    exit 1
fi
cd ..

# Step 3: Build test suite
echo -e "\n[3/6] Building test suite..."
cd tests
make -f Makefile_tests
if [ $? -ne 0 ]; then
    echo "❌ Test suite build FAILED"
    exit 1
fi

# Step 4: Run tests
echo -e "\n[4/6] Running test suite..."
./test_suite
if [ $? -ne 0 ]; then
    echo "❌ TESTS FAILED - Fix before proceeding!"
    exit 1
fi
cd ..

# Step 5: Build benchmarks
echo -e "\n[5/6] Building benchmarks..."
cd benchmarks
make
if [ $? -ne 0 ]; then
    echo "❌ Benchmark build FAILED"
    exit 1
fi

# Step 6: Run small benchmark
echo -e "\n[6/6] Running small benchmark..."
make run-small
cd ..

echo -e "\n========================================="
echo "✅ ALL BUILDS AND TESTS SUCCESSFUL!"
echo "========================================="
```

Save this as `run_all.sh` and execute:

```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 📊 WHAT TO EXPECT

### ✅ Success Path

If everything works:

1. **Core library builds** → You'll see `libcuda_orderbook.a`
2. **Tests build** → You'll see `test_suite` binary
3. **All 13 tests PASS** → You'll see `✓ Passed: 13, ✗ Failed: 0`
4. **Benchmarks build** → You'll see `benchmark_cpu_vs_gpu` binary
5. **Benchmarks run** → You'll see performance results

### ❌ If Something Fails

**Build failures:**
```
error: identifier "cudaMalloc" is undefined
```
→ **Fix:** CUDA toolkit not installed or not in PATH

**Test failures:**
```
✗ FAIL: CPU != GPU
```
→ **Fix:** There's a bug! Check which test failed and debug

**Missing files:**
```
fatal error: types.h: No such file or directory
```
→ **Fix:** Check you're in correct directory, files exist

---

## 🐛 DEBUGGING TIPS

### If build fails:

```bash
# Check CUDA is installed
nvcc --version

# Check GPU is available
nvidia-smi

# Check you're in correct directory
pwd
# Should show: .../awesome-lob
```

### If tests fail:

```bash
# Run test suite with verbose output
cd tests
./test_suite | tee test_output.txt

# Check which test failed
grep "FAIL" test_output.txt

# Run just that scenario
# (edit test_suite.cu to comment out other tests)
```

---

## 📁 DIRECTORY STRUCTURE

After successful build, you should have:

```
awesome-lob/
├── build/
│   └── libcuda_orderbook.a          ← Core library ✓
├── tests/
│   ├── test_suite                    ← Test binary ✓
│   ├── data_generator.cpp            ← Synthetic data ✓
│   └── test_suite.cu                 ← Tests ✓
├── benchmarks/
│   └── benchmark_cpu_vs_gpu          ← Benchmark binary ✓
└── RUN_INSTRUCTIONS.md               ← This file
```

---

## ⏱️ ESTIMATED TIME

- Build core library: **30 seconds**
- Build test suite: **20 seconds**
- Run all tests: **5-10 seconds**
- Build benchmarks: **15 seconds**
- Run benchmarks: **10-60 seconds** (depends on workload)

**Total: ~2-3 minutes** for complete build and test cycle

---

## 🎓 WHAT EACH STEP DOES

| Step | What It Does | Output |
|------|-------------|--------|
| 1. Clean | Remove old builds | Clean slate |
| 2. Build library | Compile CUDA kernels + CPU code | `libcuda_orderbook.a` |
| 3. Build tests | Compile test suite | `test_suite` binary |
| 4. Run tests | Verify CPU == GPU | Pass/Fail for 13 tests |
| 5. Build benchmarks | Compile benchmark code | `benchmark_cpu_vs_gpu` |
| 6. Run benchmarks | Measure performance | Timing results |

---

## ✅ SUCCESS CRITERIA

Before moving forward, ensure:

- ✅ Core library built without errors
- ✅ Test suite built without errors
- ✅ **All 13 tests PASSED** (most important!)
- ✅ Benchmarks built without errors
- ✅ No CUDA errors during execution

**If all tests pass → Your implementation is correct!** 🎉

---

## 🚀 QUICK START (TL;DR)

```bash
cd /Users/kvlnraju/Desktop/courses/semester_3/GPU/Project/awesome-lob

# Build everything
cd build && cmake .. && make && cd ..
cd tests && make -f Makefile_tests && cd ..
cd benchmarks && make && cd ..

# Run tests (MOST IMPORTANT!)
cd tests && ./test_suite
```

If you see `🎉 ALL TESTS PASSED!` → You're good to go!

---

## 📞 HELP

If stuck:
1. Check error message carefully
2. Verify prerequisites (CUDA, GPU)
3. Read `tests/TEST_SUITE_README.md` for details
4. Check `TESTING_COMPLETE.md` for test explanations

**Most important:** Make sure all tests pass before doing anything else!

