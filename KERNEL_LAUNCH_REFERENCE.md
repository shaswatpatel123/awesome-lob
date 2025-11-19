# Warp-Level Kernel Launch Quick Reference

## Configuration Constants

```cuda
constexpr int WARP_SIZE = 32;           // Hardware constant
constexpr int WARPS_PER_BLOCK = 4;      // Tunable (2, 4, 8)
constexpr int THREADS_PER_BLOCK = 128;  // WARP_SIZE × WARPS_PER_BLOCK
```

## Launch Configuration Formula

```cuda
int num_books;  // Number of LOBs to process

// Calculate grid/block dimensions
int books_per_block = WARPS_PER_BLOCK;
int num_blocks = (num_books + books_per_block - 1) / books_per_block;

dim3 grid(num_blocks);
dim3 block(THREADS_PER_BLOCK);
```

## Helper Function

```cuda
#include "utils.cuh"

dim3 grid, block;
calculate_launch_config(num_books, grid, block);
```

## Kernel Launch Examples

### 1. Initialize Orderbooks
```cuda
OrderbookBatch batch;
allocate_orderbook_batch(batch, num_books, n_orders, n_trades);

dim3 grid, block;
calculate_launch_config(num_books, grid, block);

init_orderbooks_kernel<<<grid, block>>>(batch, num_books);
cudaDeviceSynchronize();
```

### 2. Process Messages (Main Kernel)
```cuda
Message* d_messages;  // Device pointer
int num_messages_per_book;

// Same grid/block as initialization
process_messages_sequential_kernel<<<grid, block>>>(
    batch, 
    d_messages, 
    num_messages_per_book, 
    num_books
);
cudaDeviceSynchronize();
```

### 3. Query Operations
```cuda
int32_t* d_best_asks;
int32_t* d_best_bids;

get_best_bid_ask_kernel<<<grid, block>>>(
    batch, 
    d_best_asks, 
    d_best_bids, 
    num_books
);
```

### 4. Single Orderbook Operations
```cuda
// For num_books = 1
init_orderbooks_kernel<<<1, 128>>>(batch, 1);

process_messages_sequential_kernel<<<1, 128>>>(
    batch, messages, num_messages, 1
);
```

## Memory Requirements

```cuda
// Per LOB
size_t orders_per_side = n_orders_per_book * sizeof(Order);
size_t trades_size = n_trades_per_book * sizeof(Trade);

// Total batch
size_t total_orders = num_books * orders_per_side;
size_t total_trades = num_books * trades_size;

// Example: 1000 LOBs, 500 orders/side, 100 trades
// Orders: 1000 × 500 × 24 bytes = 12 MB per side
// Trades: 1000 × 100 × 24 bytes = 2.4 MB
// Total: ~26.4 MB
```

## Occupancy Considerations

### Optimal Configuration (4 warps/block)
```
Threads per block: 128
Warps per block: 4
LOBs per block: 4
Registers per thread: ~64 (depends on kernel)
Shared memory: 0 bytes

Theoretical occupancy: 100% on modern GPUs
Active blocks per SM: 8-32 (GPU dependent)
```

### Alternative Configurations

#### High Concurrency (8 warps/block)
```cuda
constexpr int WARPS_PER_BLOCK = 8;  // 256 threads
// Use for: Many small LOBs, high GPU utilization
```

#### Cache Optimized (2 warps/block)  
```cuda
constexpr int WARPS_PER_BLOCK = 2;  // 64 threads
// Use for: Large LOBs (1000+ orders), better cache hit rate
```

## Performance Guidelines

### Scalability
| LOBs | Blocks | GPU Utilization |
|------|--------|----------------|
| 1    | 1      | ~1% (testing)  |
| 10   | 3      | ~10%           |
| 100  | 25     | ~80%           |
| 1000 | 250    | 100%           |

### Message Throughput
```
Per warp: ~100-500K messages/sec (depends on message mix)
Per GPU: 100M+ messages/sec (1000s of LOBs)
```

## Common Patterns

### Batch Processing Loop
```cuda
for (int batch = 0; batch < num_batches; batch++) {
    // Copy messages for this batch
    cudaMemcpy(d_messages, h_messages[batch], 
               msg_size, cudaMemcpyHostToDevice);
    
    // Process
    process_messages_sequential_kernel<<<grid, block>>>(
        batch, d_messages, num_messages, num_books
    );
    
    // Copy results
    cudaMemcpy(h_trades, d_trades, 
               trades_size, cudaMemcpyDeviceToHost);
}
```

### Streaming for Overlap
```cuda
cudaStream_t stream[N_STREAMS];
for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamCreate(&stream[i]);
}

for (int batch = 0; batch < num_batches; batch++) {
    int s = batch % N_STREAMS;
    
    // Async copy H→D
    cudaMemcpyAsync(d_messages[s], h_messages[batch], 
                    size, cudaMemcpyHostToDevice, stream[s]);
    
    // Process
    process_messages_sequential_kernel<<<grid, block, 0, stream[s]>>>(
        batch, d_messages[s], num_messages, num_books
    );
    
    // Async copy D→H
    cudaMemcpyAsync(h_results[batch], d_results[s], 
                    size, cudaMemcpyDeviceToHost, stream[s]);
}
```

## Error Checking

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_batch, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaDeviceSynchronize());
```

## Testing Snippet

```cuda
// Minimal example
int main() {
    int num_books = 10;
    int n_orders = 100;
    int n_trades = 50;
    
    // Allocate
    OrderbookBatch batch;
    allocate_orderbook_batch(batch, num_books, n_orders, n_trades);
    allocate_host_orderbook_batch(batch, num_books, n_orders, n_trades);
    
    // Initialize
    dim3 grid, block;
    calculate_launch_config(num_books, grid, block);
    init_orderbooks_kernel<<<grid, block>>>(batch, num_books);
    
    // Create and process messages
    Message msg = {Message::LIMIT, Message::BID, 100, 99000, 1, 1, 0, 0};
    Message* d_msg;
    cudaMalloc(&d_msg, sizeof(Message));
    cudaMemcpy(d_msg, &msg, sizeof(Message), cudaMemcpyHostToDevice);
    
    process_messages_sequential_kernel<<<grid, block>>>(
        batch, d_msg, 1, num_books
    );
    
    // Copy back and verify
    copy_to_host(batch);
    print_orderbook(batch, 0);
    
    // Cleanup
    free_orderbook_batch(batch);
    free_host_orderbook_batch(batch);
    cudaFree(d_msg);
    
    return 0;
}
```

## Summary

**Key Points**:
- Always use `calculate_launch_config()` for correct dimensions
- 4 warps/block is optimal for most workloads
- No shared memory needed (0 bytes)
- Same grid/block config for all kernels
- Test with 1 LOB first, then scale up

**Common Mistake**:
```cuda
// ❌ Wrong: Using old block-level config
kernel<<<num_books, 256>>>(...)

// ✓ Correct: Using warp-level config
calculate_launch_config(num_books, grid, block);
kernel<<<grid, block>>>(...)
```

