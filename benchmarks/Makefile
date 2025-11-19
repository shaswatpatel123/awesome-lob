# Makefile for CPU vs GPU Orderbook Benchmarks

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3 -I../include --compiler-options -Wall -rdc=true --use_fast_math -Wno-deprecated-gpu-targets

# GPU Architecture (adjust for your GPU)
# Common options:
#   A100: sm_80
#   Tesla V100: sm_70
#   RTX 2080/2080 Ti: sm_75
#   RTX 3080/3090: sm_86
#   RTX 4080/4090: sm_89
#   GTX TITAN X: sm_52
GPU_ARCH = sm_80

# Source files
CPU_SOURCES = ../src/orderbook_cpu.cpp
CUDA_SOURCES = ../src/kernels.cu ../src/operations.cu ../src/utils.cu
BENCHMARK_SOURCE = benchmark_cpu_vs_gpu.cu
OPERATIONS_BENCHMARK_SOURCE = benchmark_operations.cu

# Output
BENCHMARK_BIN = benchmark_cpu_vs_gpu
OPERATIONS_BIN = benchmark_operations
OPERATIONS_CPU_BIN = benchmark_operations_cpu

# C++ compiler settings
CXX = g++
CXX_FLAGS = -std=c++14 -I../include -Wall

# Targets
.PHONY: all clean run help operations operations-cpu

all: $(BENCHMARK_BIN) $(OPERATIONS_BIN) $(OPERATIONS_CPU_BIN)

$(BENCHMARK_BIN): $(BENCHMARK_SOURCE) $(CPU_SOURCES) $(CUDA_SOURCES)
	@echo "Compiling CPU vs GPU benchmark..."
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) \
		$(BENCHMARK_SOURCE) \
		$(CPU_SOURCES) \
		$(CUDA_SOURCES) \
		-o $(BENCHMARK_BIN)
	@echo "Build complete: $(BENCHMARK_BIN)"

$(OPERATIONS_BIN): $(OPERATIONS_BENCHMARK_SOURCE) $(CUDA_SOURCES)
	@echo "Compiling operations timing benchmark..."
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) \
		$(OPERATIONS_BENCHMARK_SOURCE) \
		$(CUDA_SOURCES) \
		-o $(OPERATIONS_BIN)
	@echo "Build complete: $(OPERATIONS_BIN)"

$(OPERATIONS_CPU_BIN): benchmark_operations_cpu.cpp $(CPU_SOURCES)
	@echo "Compiling CPU operations timing benchmark..."
	$(CXX) $(CXX_FLAGS) -O3 \
		benchmark_operations_cpu.cpp \
		$(CPU_SOURCES) \
		-o $(OPERATIONS_CPU_BIN)
	@echo "Build complete: $(OPERATIONS_CPU_BIN)"

operations: $(OPERATIONS_BIN)

operations-cpu: $(OPERATIONS_CPU_BIN)

clean:
	@echo "Cleaning..."
	rm -f $(BENCHMARK_BIN) $(OPERATIONS_BIN) $(OPERATIONS_CPU_BIN)
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

run-ops: $(OPERATIONS_BIN)
	@echo "Running operation timing benchmark with default parameters..."
	./$(OPERATIONS_BIN)

run-ops-small: $(OPERATIONS_BIN)
	@echo "Running operation timing benchmark (small: 10 books, 100 msgs)..."
	./$(OPERATIONS_BIN) 10 100

run-ops-medium: $(OPERATIONS_BIN)
	@echo "Running operation timing benchmark (medium: 100 books, 1000 msgs)..."
	./$(OPERATIONS_BIN) 100 1000

run-ops-large: $(OPERATIONS_BIN)
	@echo "Running operation timing benchmark (large: 1000 books, 10000 msgs)..."
	./$(OPERATIONS_BIN) 1000 10000

run-ops-cpu: $(OPERATIONS_CPU_BIN)
	@echo "Running CPU operation timing benchmark with default parameters..."
	./$(OPERATIONS_CPU_BIN)

run-ops-cpu-small: $(OPERATIONS_CPU_BIN)
	@echo "Running CPU operation timing benchmark (small: 10 books, 100 msgs)..."
	./$(OPERATIONS_CPU_BIN) 10 100

run-ops-cpu-medium: $(OPERATIONS_CPU_BIN)
	@echo "Running CPU operation timing benchmark (medium: 100 books, 1000 msgs)..."
	./$(OPERATIONS_CPU_BIN) 100 1000

run-ops-cpu-large: $(OPERATIONS_CPU_BIN)
	@echo "Running CPU operation timing benchmark (large: 1000 books, 10000 msgs)..."
	./$(OPERATIONS_CPU_BIN) 1000 10000

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
	@echo ""
	@echo "BUILD:"
	@echo "  make              - Build all benchmarks"
	@echo "  make operations   - Build GPU operations timing benchmark"
	@echo "  make operations-cpu - Build CPU operations timing benchmark"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "CPU vs GPU BENCHMARK:"
	@echo "  make run          - Run with default parameters"
	@echo "  make run-small    - Run small workload"
	@echo "  make run-medium   - Run medium workload"
	@echo "  make run-large    - Run large workload"
	@echo ""
	@echo "GPU OPERATION TIMING BENCHMARK:"
	@echo "  make run-ops        - Run GPU operation timing (default: 100 books, 1000 msgs)"
	@echo "  make run-ops-small  - Run GPU operation timing (10 books, 100 msgs)"
	@echo "  make run-ops-medium - Run GPU operation timing (100 books, 1000 msgs)"
	@echo "  make run-ops-large  - Run GPU operation timing (1000 books, 10000 msgs)"
	@echo ""
	@echo "CPU OPERATION TIMING BENCHMARK:"
	@echo "  make run-ops-cpu        - Run CPU operation timing (default: 100 books, 1000 msgs)"
	@echo "  make run-ops-cpu-small  - Run CPU operation timing (10 books, 100 msgs)"
	@echo "  make run-ops-cpu-medium - Run CPU operation timing (100 books, 1000 msgs)"
	@echo "  make run-ops-cpu-large  - Run CPU operation timing (1000 books, 10000 msgs)"
	@echo ""
	@echo "PROFILING:"
	@echo "  make profile      - Profile with nvprof"
	@echo "  make profile-nsys - Profile with Nsight Systems"
	@echo "  make profile-ncu  - Profile with Nsight Compute"
	@echo ""
	@echo "Custom parameters:"
	@echo "  ./$(BENCHMARK_BIN) <num_books> <msgs_per_book> <orders_per_side> <max_trades>"
	@echo "  ./$(OPERATIONS_BIN) <num_books> <msgs_per_book>"
	@echo "  ./$(OPERATIONS_CPU_BIN) <num_books> <msgs_per_book>"
	@echo ""
	@echo "GPU Architecture:"
	@echo "  Current: $(GPU_ARCH)"
	@echo "  Change with: make GPU_ARCH=sm_86"

