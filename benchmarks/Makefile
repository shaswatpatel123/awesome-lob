# Makefile for CPU vs GPU Orderbook Benchmarks

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3 -I../include --compiler-options -Wall -rdc=true

# GPU Architecture (adjust for your GPU)
# Common options:
#   Tesla V100: sm_70
#   Tesla T4, RTX 2080/2080 Ti: sm_75
#   RTX 3080/3090: sm_86
#   RTX 4080/4090: sm_89
GPU_ARCH = sm_75

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
	$(NVCC) $(NVCC_FLAGS) -arch=$(GPU_ARCH) \
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
	@echo "  ./$(BENCHMARK_BIN) <num_books> <messages_per_book> <orders_per_side> <max_trades>"
	@echo ""
	@echo "GPU Architecture:"
	@echo "  Current: $(GPU_ARCH)"
	@echo "  Change with: make GPU_ARCH=sm_86"

