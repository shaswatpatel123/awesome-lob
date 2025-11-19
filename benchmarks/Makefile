# Makefile for CPU vs GPU Orderbook Benchmarks

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3 -I../include --compiler-options -Wall -rdc=true

# GPU Architecture - Auto-detect or manually specify
# - If GPU_ARCH is provided (sm_XX or XX) we'll target only that arch
# - Otherwise we detect via nvidia-smi; if detection fails we build a fat binary
#   that supports a broad set of architectures so the executable runs on any GPU.

# Auto-detect GPU architecture if not specified
ifeq ($(GPU_ARCH),)
    # Try to detect from nvidia-smi compute capability
    COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    ifeq ($(COMPUTE_CAP),)
        # Fallback: try to detect from GPU name
        GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        ifneq ($(GPU_NAME),)
            ifneq ($(findstring T4,$(GPU_NAME)),)
                COMPUTE_CAP := 75
            else ifneq ($(findstring V100,$(GPU_NAME)),)
                COMPUTE_CAP := 70
            else ifneq ($(findstring A100,$(GPU_NAME)),)
                COMPUTE_CAP := 80
            else ifneq ($(findstring RTX 30,$(GPU_NAME)),)
                COMPUTE_CAP := 86
            else ifneq ($(findstring RTX 40,$(GPU_NAME)),)
                COMPUTE_CAP := 89
            endif
        endif
    endif
    
    # Map compute capability to architecture (support sm_52, sm_75, sm_80)
    ifeq ($(COMPUTE_CAP),52)
        GPU_ARCH := sm_52
    else ifeq ($(COMPUTE_CAP),75)
        GPU_ARCH := sm_75
    else ifeq ($(COMPUTE_CAP),80)
        GPU_ARCH := sm_80
    else ifeq ($(COMPUTE_CAP),86)
        GPU_ARCH := sm_80
    else ifeq ($(COMPUTE_CAP),70)
        GPU_ARCH := sm_75
    else
        # Leave GPU_ARCH empty - we'll fall back to a fat binary
    endif
endif

# Determine which architectures to target
DEFAULT_ARCH_LIST := 52 60 70 75 80 86 89
GPU_ARCH_LIST :=

ifneq ($(GPU_ARCH),)
    GPU_ARCH_SANITIZED := $(patsubst sm_%,%,$(GPU_ARCH))
    ifeq ($(GPU_ARCH_SANITIZED),)
        GPU_ARCH_SANITIZED := $(GPU_ARCH)
    endif
    GPU_ARCH_LIST := $(GPU_ARCH_SANITIZED)
else ifneq ($(COMPUTE_CAP),)
    GPU_ARCH_LIST := $(COMPUTE_CAP)
else
    GPU_ARCH_LIST := $(DEFAULT_ARCH_LIST)
endif

NVCC_ARCH_FLAGS := $(foreach arch,$(GPU_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))
NVCC_FLAGS += $(NVCC_ARCH_FLAGS)

# Source files
CPU_SOURCES = ../src/orderbook_cpu.cpp
CUDA_SOURCES = ../src/kernels.cu ../src/operations.cu ../src/utils.cu
BENCHMARK_SOURCE = benchmark_cpu_vs_gpu.cu

# Output
BENCHMARK_BIN = benchmark_cpu_vs_gpu

# Targets
.PHONY: all clean run help

all: $(BENCHMARK_BIN)

$(BENCHMARK_BIN): $(BENCHMARK_SOURCE) $(CPU_SOURCES) $(CUDA_SOURCES)
	@echo "Compiling benchmark..."
	@echo "Generating code for SM architectures: $(GPU_ARCH_LIST)"
	$(NVCC) $(NVCC_FLAGS) \
		$(BENCHMARK_SOURCE) \
		$(CPU_SOURCES) \
		$(CUDA_SOURCES) \
		-o $(BENCHMARK_BIN)
	@echo "Build complete: $(BENCHMARK_BIN)"

clean:
	@echo "Cleaning..."
	rm -f $(BENCHMARK_BIN)
	@echo "Clean complete"

run: $(BENCHMARK_BIN)
	@echo "Running benchmark with default parameters..."
	./$(BENCHMARK_BIN)

run-small: $(BENCHMARK_BIN)
	@echo "Running small workload (10 books, 100 messages)..."
	./$(BENCHMARK_BIN) 10 100 50 50

run-medium: $(BENCHMARK_BIN)
	@echo "Running medium workload (100 books, 1000 messages)..."
	./$(BENCHMARK_BIN) 100 1000 100 100

run-large: $(BENCHMARK_BIN)
	@echo "Running large workload (1000 books, 10000 messages)..."
	./$(BENCHMARK_BIN) 1000 10000 200 200

profile: $(BENCHMARK_BIN)
	@echo "Profiling with nvprof..."
	nvprof ./$(BENCHMARK_BIN) 100 1000 100 100

profile-nsys: $(BENCHMARK_BIN)
	@echo "Profiling with Nsight Systems..."
	nsys profile --stats=true ./$(BENCHMARK_BIN) 100 1000 100 100

profile-ncu: $(BENCHMARK_BIN)
	@echo "Profiling with Nsight Compute..."
	ncu --set full ./$(BENCHMARK_BIN) 100 1000 100 100

help:
	@echo "Available targets:"
	@echo "  make              - Build the benchmark"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make run          - Run with default parameters"
	@echo "  make run-small    - Run small workload"
	@echo "  make run-medium   - Run medium workload"
	@echo "  make run-large    - Run large workload"
	@echo "  make profile      - Profile with nvprof"
	@echo "  make profile-nsys - Profile with Nsight Systems"
	@echo "  make profile-ncu  - Profile with Nsight Compute"
	@echo ""
	@echo "Custom parameters:"
	@echo "  ./$(BENCHMARK_BIN) <num_books> <messages_per_book> <orders_per_side> <max_trades> [block_size]"
	@echo "  block_size: Threads per block (default: 256, must be power of 2: 32, 64, 128, 256, 512, 1024)"
	@echo ""
	@echo "GPU Architectures:"
	@echo "  Targeted SM versions: $(GPU_ARCH_LIST)"
	@echo "  Override with: make GPU_ARCH=sm_80 or GPU_ARCH_LIST=\"70 80\""

