#!/bin/bash
# GPU Architecture Auto-Detection Script
# Detects the compute capability of the first available GPU
# Returns SM version (e.g., "52", "75", "86") or "52" as safe default

# Method 1: Try to get compute capability directly from CUDA
if command -v nvidia-smi &> /dev/null; then
    # Get GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
    
    if [ -n "$GPU_NAME" ]; then
        # Map GPU names to compute capabilities
        case "$GPU_NAME" in
            *"TITAN X"*|*"Maxwell"*|*"GTX 9"*|*"GTX TITAN"*)
                echo "52"
                exit 0
                ;;
            *"GTX 10"*|*"P100"*|*"Pascal"*)
                # Check for specific models
                if [[ "$GPU_NAME" == *"GTX 1080"* ]] || [[ "$GPU_NAME" == *"GTX 1070"* ]] || [[ "$GPU_NAME" == *"GTX 1060"* ]]; then
                    echo "61"
                else
                    echo "60"
                fi
                exit 0
                ;;
            *"V100"*|*"Volta"*)
                echo "70"
                exit 0
                ;;
            *"T4"*|*"RTX 20"*|*"Turing"*|*"GTX 16"*)
                echo "75"
                exit 0
                ;;
            *"A100"*|*"A6000"*|*"A40"*)
                echo "80"
                exit 0
                ;;
            *"RTX 30"*|*"RTX A"*|*"Ampere"*)
                echo "86"
                exit 0
                ;;
            *"RTX 40"*|*"Ada"*|*"H100"*)
                echo "89"
                exit 0
                ;;
            *"RTX 50"*|*"Blackwell"*)
                echo "100"
                exit 0
                ;;
        esac
    fi
    
    # Try to get compute capability directly
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ] && [[ "$COMPUTE_CAP" =~ ^[0-9]+$ ]]; then
        # Convert 7.5 to 75, 8.6 to 86, etc.
        if [[ "$COMPUTE_CAP" == *"."* ]]; then
            COMPUTE_CAP=$(echo "$COMPUTE_CAP" | tr -d '.')
        fi
        echo "$COMPUTE_CAP"
        exit 0
    fi
fi

# Method 2: Try deviceQuery if available (CUDA samples)
if command -v deviceQuery &> /dev/null; then
    COMPUTE_CAP=$(deviceQuery 2>/dev/null | grep "Compute Capability" | head -1 | awk '{print $3}' | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        echo "$COMPUTE_CAP"
        exit 0
    fi
fi

# Method 3: Try to compile and run a small CUDA program
TEMP_DIR=$(mktemp -d)
cat > "$TEMP_DIR/detect_arch.cu" << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int major = prop.major;
        int minor = prop.minor;
        printf("%d%d\n", major, minor);
        return 0;
    }
    return 1;
}
EOF

if command -v nvcc &> /dev/null; then
    if nvcc -o "$TEMP_DIR/detect_arch" "$TEMP_DIR/detect_arch.cu" -lcudart 2>/dev/null; then
        if [ -f "$TEMP_DIR/detect_arch" ]; then
            COMPUTE_CAP=$("$TEMP_DIR/detect_arch" 2>/dev/null)
            rm -rf "$TEMP_DIR"
            if [ -n "$COMPUTE_CAP" ] && [[ "$COMPUTE_CAP" =~ ^[0-9]+$ ]]; then
                echo "$COMPUTE_CAP"
                exit 0
            fi
        fi
    fi
fi

# Cleanup
rm -rf "$TEMP_DIR" 2>/dev/null

# Fallback: Return safe default (SM 52 - Maxwell, widely compatible)
echo "52"

