# GPU Architecture Auto-Detection

This project now includes **automatic GPU architecture detection**! The build system will automatically detect your GPU's compute capability and configure the build accordingly.

## 🎯 How It Works

The auto-detection system uses multiple methods to determine your GPU architecture:

1. **nvidia-smi** - Queries GPU name and compute capability
2. **CUDA Runtime API** - Compiles and runs a small program to query device properties
3. **Fallback** - Defaults to SM 52 (Maxwell) for maximum compatibility

## 🚀 Usage

### CMake Build (Recommended)

```bash
mkdir build && cd build
cmake ..  # Auto-detects GPU architecture
make -j$(nproc)
```

The CMake configuration will:
- ✅ Auto-detect your GPU architecture
- ✅ Display the detected architecture in the configuration output
- ✅ Build optimized code for your specific GPU

**Override auto-detection:**
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..  # Force specific architecture
cmake -DCMAKE_CUDA_ARCHITECTURES="52;75;86" ..  # Build for multiple architectures
```

### Makefile Builds

#### Test Suite
```bash
cd tests
make  # Auto-detects GPU architecture
```

#### Benchmarks
```bash
cd benchmarks
make  # Auto-detects GPU architecture
```

**Override auto-detection:**
```bash
make GPU_ARCH=sm_86  # Force specific architecture
```

### Build Scripts

All build scripts (`build_and_test.sh`, `run_all.sh`, etc.) now use auto-detection automatically.

## 📋 Supported GPU Architectures

The auto-detection supports all common GPU architectures:

| Architecture | Compute Capability | Example GPUs |
|-------------|-------------------|--------------|
| Maxwell | SM 52 | GTX TITAN X, GTX 9xx |
| Pascal | SM 60, 61 | GTX 10xx, P100 |
| Volta | SM 70 | V100 |
| Turing | SM 75 | RTX 20xx, T4, GTX 16xx |
| Ampere | SM 80, 86 | A100 (80), RTX 30xx (86) |
| Ada Lovelace | SM 89 | RTX 40xx |
| Blackwell | SM 100 | RTX 50xx, H100 |

## 🔧 Manual Detection

If you want to manually check your GPU architecture:

```bash
# Method 1: Use the detection script
./detect_gpu_arch.sh

# Method 2: Use nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Method 3: Use deviceQuery (if CUDA samples installed)
deviceQuery | grep "Compute Capability"
```

## 🛠️ Troubleshooting

### Auto-detection fails

If auto-detection fails, the system will:
- **CMake**: Build for multiple architectures (52;60;61;70;75;80;86;89) for compatibility
- **Makefiles**: Default to SM 52 (widely compatible)
- **Scripts**: Default to SM 52

You can always override by specifying the architecture manually.

### No GPU detected

If no GPU is detected (e.g., building on a system without GPU):
- The system will use safe defaults
- You can still build by specifying an architecture manually
- The code will compile but won't run without a compatible GPU

### Wrong architecture detected

If the wrong architecture is detected:
1. Check your GPU model: `nvidia-smi --query-gpu=name --format=csv`
2. Manually override: `cmake -DCMAKE_CUDA_ARCHITECTURES="XX" ..`
3. Report the issue with your GPU model for script improvement

## 📝 Implementation Details

### Detection Script: `detect_gpu_arch.sh`

The detection script (`detect_gpu_arch.sh`) uses multiple detection methods:

1. **GPU Name Mapping** - Maps common GPU names to compute capabilities
2. **nvidia-smi Query** - Directly queries compute capability
3. **CUDA Runtime** - Compiles a small CUDA program to query device properties

### Integration Points

- **CMakeLists.txt**: Runs detection script at configure time
- **Makefiles**: Runs detection script at make time
- **Build Scripts**: Uses detection script for compilation

## 🎉 Benefits

✅ **No manual configuration needed** - Just run `cmake ..` or `make`  
✅ **Works on any GPU** - Automatically adapts to your hardware  
✅ **Optimized builds** - Code compiled specifically for your GPU  
✅ **Easy override** - Still allows manual specification when needed  
✅ **Safe defaults** - Falls back gracefully if detection fails  

## 📚 Related Files

- `detect_gpu_arch.sh` - Main detection script
- `CMakeLists.txt` - CMake auto-detection integration
- `tests/Makefile_tests` - Test suite auto-detection
- `benchmarks/Makefile` - Benchmark auto-detection
- `tests/build_and_test.sh` - Build script with auto-detection

