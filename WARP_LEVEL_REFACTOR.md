# Warp-Level Parallelism Refactoring

## Overview
Refactored from **1 LOB per block** to **1 LOB per warp** (32 threads).

## Architecture Changes

### Core Principle
- **Before**: 1 orderbook = 1 thread block (up to 1024 threads)
- **After**: 1 orderbook = 1 warp (32 threads)
- **Benefits**: Better GPU occupancy, reduced shared memory usage, finer-grained parallelism

### Kernel Launch Configuration
```cuda
// OLD: 1 block per LOB
dim3 grid(num_books);
dim3 block(256);

// NEW: Multiple warps per block, 4 warps = 128 threads per block
int warps_per_block = 4;
int books_per_block = warps_per_block;
int num_blocks = (num_books + books_per_block - 1) / books_per_block;
dim3 grid(num_blocks);
dim3 block(128);  // 4 warps × 32 threads
```

## Key Implementation Changes

### 1. Thread Indexing (kernels.cu)
```cuda
// Helper functions added
__device__ inline int get_warp_id() {
    return threadIdx.x / 32;
}

__device__ inline int get_lane_id() {
    return threadIdx.x % 32;
}

__device__ inline int get_book_idx(int num_books) {
    int warps_per_block = blockDim.x / 32;
    int book_idx = blockIdx.x * warps_per_block + get_warp_id();
    return (book_idx < num_books) ? book_idx : -1;
}
```

### 2. Synchronization Changes (operations.cu)

**Block-level (OLD)**:
```cuda
__syncthreads();  // Synchronize all threads in block
__shared__ int shared_data[];  // Block-level shared memory
```

**Warp-level (NEW)**:
```cuda
__syncwarp();  // Implicit - warp executes in lockstep
int data = __shfl_sync(0xFFFFFFFF, value, src_lane);  // Warp shuffle
```

### 3. Parallel Reductions

**Block-level reduction (OLD)**:
```cuda
extern __shared__ BestOrderInfo shared_best[];
// Tree reduction in shared memory
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    // Reduce...
}
```

**Warp-level reduction (NEW)**:
```cuda
// Shuffle-based reduction
for (int offset = 32/2; offset > 0; offset /= 2) {
    int32_t other = __shfl_down_sync(0xFFFFFFFF, local_value, offset);
    local_value = min(local_value, other);
}
// Broadcast result to all lanes
result = __shfl_sync(0xFFFFFFFF, result, 0);
```

### 4. Message Broadcasting

**OLD (shared memory)**:
```cuda
__shared__ Message shared_msg;
if (threadIdx.x == 0) shared_msg = message;
__syncthreads();
```

**NEW (warp shuffle)**:
```cuda
Message msg;
if (laneId == 0) msg = message;
// Broadcast each field to all lanes
msg.type = __shfl_sync(0xFFFFFFFF, msg.type, 0);
msg.side = __shfl_sync(0xFFFFFFFF, msg.side, 0);
msg.quantity = __shfl_sync(0xFFFFFFFF, msg.quantity, 0);
// ... continue for all fields
```

## Modified Files

### Core Implementation
1. **src/kernels.cu**: Warp-level kernel wrappers
   - Added warp indexing helpers
   - Changed all kernels to use `get_book_idx()`
   - Removed shared memory allocations

2. **src/operations.cu**: Device functions using warp primitives
   - Renamed functions: `*_device` → `*_warp`
   - Added `laneId` parameter to all functions
   - Replaced block-level with warp-level reductions
   - Used `__shfl_sync()` for communication

3. **src/utils.cu**: Launch configuration helper
   - Added `calculate_launch_config()` function
   - Updated `init_orderbooks_device()` to use proper config

### Headers
4. **include/kernels.cuh**: Updated documentation

### Tests
5. **tests/test_suite.cu**: Updated all kernel launches
6. **tests/test_matching.cu**: Updated kernel launches

## Performance Implications

### Advantages
- **Higher Occupancy**: More warps per SM (4×)
- **No Shared Memory**: Faster kernel launch, more blocks resident
- **Better Scaling**: Can process 4× LOBs per block
- **Reduced Divergence**: Warp naturally handles branching

### Considerations
- Each LOB now uses exactly 32 threads (vs potentially 256+)
- Sequential operations (add/cancel/match) still on lane 0 only
- Parallel search operations benefit from full warp participation

## Usage Example

```cuda
// Initialize batch
OrderbookBatch batch;
allocate_orderbook_batch(batch, num_books, n_orders, n_trades);

// Calculate launch config
dim3 grid, block;
calculate_launch_config(num_books, grid, block);

// Launch kernel (no shared memory needed)
init_orderbooks_kernel<<<grid, block>>>(batch, num_books);

// Process messages
process_messages_sequential_kernel<<<grid, block>>>(
    batch, messages, num_messages_per_book, num_books
);
```

## Testing

All existing tests remain valid. Launch configurations automatically adjusted to warp-level parallelism.

```bash
cd tests
make -f Makefile_tests
./test_suite
./test_matching
```

## Summary

This refactoring maintains identical functionality while improving GPU resource utilization through warp-level parallelism. The code is cleaner, uses hardware more efficiently, and scales better to multiple LOBs.

