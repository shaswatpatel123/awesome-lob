# 🔨 BUILD VERIFICATION GUIDE

**Use this to verify your code compiles BEFORE running tests.**

---

## ⚡ FASTEST WAY - Automated Build Check

```bash
cd /Users/kvlnraju/Desktop/courses/semester_3/GPU/Project/awesome-lob
./build_only.sh
```

**This will:**
1. ✓ Clean previous builds
2. ✓ Build core library (CUDA + CPU)
3. ✓ Build test suite
4. ✓ Verify output files exist

**Time:** ~1 minute

**Output if successful:**
```
✅ BUILD VERIFICATION SUCCESSFUL!

All components compiled without errors:
  ✓ Core library (libcuda_orderbook.a)
  ✓ Test suite (test_suite)

Your code compiles correctly! 🎉
```

---

## 📋 MANUAL STEP-BY-STEP (If You Want Control)

### **Step 1: Build Core Library Only**

```bash
cd build
cmake ..
make
```

**What to look for:**
- ✅ **Success:** You see `libcuda_orderbook.a` file
- ❌ **Failure:** Compilation errors, missing files

**Common errors:**
```
error: identifier "cudaMalloc" is undefined
→ CUDA toolkit not installed or not in PATH

fatal error: types.h: No such file or directory
→ Include path wrong or file missing
```

---

### **Step 2: Build Test Suite Only**

```bash
cd tests
make -f Makefile_tests
```

**What to look for:**
- ✅ **Success:** You see `test_suite` binary
- ❌ **Failure:** Linker errors, missing dependencies

**Common errors:**
```
undefined reference to 'process_message_cpu'
→ CPU implementation not linked

cannot find -lcuda_orderbook
→ Core library not built (do Step 1 first)
```

---

### **Step 3: Verify Files Exist**

```bash
# Check core library
ls -lh build/libcuda_orderbook.a

# Check test binary
ls -lh tests/test_suite
```

**Both files should exist!**

---

## 🎯 WHAT EACH STEP CHECKS

### **Step 1 (Core Library):**
Tests if your CUDA and CPU code compiles:
- ✓ `src/kernels.cu` - CUDA kernels
- ✓ `src/operations.cu` - CUDA device functions
- ✓ Syntax errors caught here
- ✓ CUDA compilation issues caught here

### **Step 2 (Test Suite):**
Tests if everything links together:
- ✓ CPU implementation compiles
- ✓ Test code compiles
- ✓ Data generator compiles
- ✓ All components link correctly

---

## ✅ SUCCESS CRITERIA

After running `./build_only.sh`, you should have:

```
build/
  └── libcuda_orderbook.a    ← Core library (CUDA + CPU kernels)

tests/
  └── test_suite             ← Test executable
```

**If both files exist** → ✅ **BUILD SUCCESSFUL**  
**If either missing** → ❌ **BUILD FAILED**

---

## ❌ COMMON BUILD FAILURES

### 1. **CUDA Not Found**
```
CMake Error: Could not find CUDA toolkit
```
**Fix:**
```bash
# Check CUDA installed
nvcc --version

# If not installed, install CUDA toolkit
# If installed, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

---

### 2. **GPU Architecture Mismatch**
```
nvcc fatal: Unsupported gpu architecture 'compute_86'
```
**Fix:** Edit `CMakeLists.txt` line 23:
```cmake
# Change this:
set(CMAKE_CUDA_ARCHITECTURES "75;86")

# To match your GPU (check with nvidia-smi):
# RTX 2080: "75"
# RTX 3080: "86"
# Tesla V100: "70"
```

---

### 3. **Missing Files**
```
fatal error: data_generator.h: No such file or directory
```
**Fix:** Check all required files exist:
```bash
ls tests/data_generator.h
ls tests/data_generator.cpp
ls tests/test_suite.cu
```

---

### 4. **Linker Errors**
```
undefined reference to `process_messages_sequential_kernel'
```
**Fix:** This means CUDA device code isn't linking. Check `CMakeLists.txt` has:
```cmake
set_target_properties(cuda_orderbook PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

---

## 🔍 DEBUGGING BUILD FAILURES

### Read Error Messages Carefully!

**Error messages tell you:**
1. **Which file** has the problem
2. **Which line** has the error
3. **What** the error is

**Example:**
```
src/operations.cu:254:5: error: expected ';' before 'qtm_remaining'
  254 |     qtm_remaining = max(0, qtm_remaining - passive_order.quantity)
      |     ^~~~~~~~~~~~~
```
**Problem:** Line 254 in `operations.cu`, missing semicolon on previous line

---

### Check Prerequisites

Before building, verify:
```bash
# CUDA installed?
nvcc --version

# GPU available?
nvidia-smi

# CMake installed?
cmake --version

# In correct directory?
pwd
# Should show: .../awesome-lob
```

---

## 🚀 AFTER SUCCESSFUL BUILD

Once `./build_only.sh` succeeds, you have 3 options:

### **Option 1: Run Tests (Recommended)**
```bash
cd tests
./test_suite
```
This verifies your code is **correct** (not just compiling).

### **Option 2: Run Full Suite**
```bash
./run_all.sh
```
Runs build + tests + benchmarks.

### **Option 3: Manual Test Run**
```bash
cd tests
./test_suite | tee test_output.txt
```
Saves output to file for review.

---

## 📊 BUILD vs RUN

| Script | What It Does | Time | Use When |
|--------|-------------|------|----------|
| `build_only.sh` | Compile only | ~1 min | First time, after code changes |
| `run_all.sh` | Build + Test + Benchmark | ~3 min | Full verification |
| `cd tests && ./test_suite` | Run tests only | ~10 sec | After successful build |

**Recommended workflow:**
1. First time: `./build_only.sh` (verify it compiles)
2. If success: `cd tests && ./test_suite` (verify correctness)
3. If tests pass: `./run_all.sh` (full suite)

---

## 💡 QUICK REFERENCE

```bash
# JUST BUILD (No testing)
./build_only.sh

# If build succeeds, THEN test
cd tests && ./test_suite

# Or do everything at once
./run_all.sh
```

---

## ⚠️ IMPORTANT NOTES

1. **Build first, test later**
   - If code doesn't compile, tests can't run
   - Fix compilation errors before worrying about test failures

2. **Clean before rebuild**
   - After code changes: `cd build && make clean && cd ..`
   - Then rebuild: `./build_only.sh`

3. **Check both outputs**
   - Core library: `build/libcuda_orderbook.a`
   - Test binary: `tests/test_suite`
   - Both must exist!

---

## ✅ SUCCESS CHECKLIST

After running `./build_only.sh`:

- [ ] Script completed without errors
- [ ] Saw "✅ BUILD VERIFICATION SUCCESSFUL!"
- [ ] File exists: `build/libcuda_orderbook.a`
- [ ] File exists: `tests/test_suite`
- [ ] No compilation errors in output

If all checked → **Ready to run tests!**

---

## 🎉 NEXT STEP

```bash
cd tests
./test_suite
```

This will verify your code is **correct**, not just **compiled**.

Goal: See `✓ Passed: 13, ✗ Failed: 0` 🎯

