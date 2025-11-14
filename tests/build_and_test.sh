#!/bin/bash

# Build and Test Script for CUDA Orderbook
# Run this on your cloud GPU machine

set -e  # Exit on error

echo "========================================="
echo "CUDA Orderbook - Build and Test"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the tests directory
if [ ! -f "test_matching.cu" ]; then
    echo -e "${RED}Error: test_matching.cu not found!${NC}"
    echo "Please run this script from the tests/ directory"
    exit 1
fi

# Step 1: Build the library
echo -e "${YELLOW}Step 1: Building CUDA orderbook library...${NC}"
cd ..
if [ ! -d "build" ]; then
    mkdir build
fi
cd build

# Clean previous build
rm -f CMakeCache.txt
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Library built successfully${NC}"
else
    echo -e "${RED}✗ Library build failed${NC}"
    exit 1
fi

# Step 2: Compile test program
echo ""
echo -e "${YELLOW}Step 2: Compiling test program...${NC}"
cd ../tests

# Detect CUDA architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Detected GPU: $GPU_NAME"
    
    # Set architecture based on common GPUs
    if [[ $GPU_NAME == *"T4"* ]]; then
        ARCH="75"
    elif [[ $GPU_NAME == *"V100"* ]]; then
        ARCH="70"
    elif [[ $GPU_NAME == *"A100"* ]] || [[ $GPU_NAME == *"RTX 30"* ]]; then
        ARCH="80"
    elif [[ $GPU_NAME == *"RTX 40"* ]]; then
        ARCH="89"
    else
        ARCH="75"  # Default
        echo -e "${YELLOW}Warning: Unknown GPU, using default architecture 75${NC}"
    fi
    echo "Using CUDA architecture: $ARCH"
else
    ARCH="75"
    echo -e "${YELLOW}Warning: nvidia-smi not found, using default architecture 75${NC}"
fi

nvcc -arch=sm_$ARCH \
     -I../include \
     -L../build \
     -lcuda_orderbook \
     test_matching.cu \
     -o test_matching

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test program compiled successfully${NC}"
else
    echo -e "${RED}✗ Test compilation failed${NC}"
    exit 1
fi

# Step 3: Run tests
echo ""
echo -e "${YELLOW}Step 3: Running tests...${NC}"
echo ""

# Set library path
export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH

./test_matching

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  ✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}=========================================${NC}"
else
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  ✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}=========================================${NC}"
fi

exit $TEST_RESULT

