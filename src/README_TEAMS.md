# Source Implementation Guide

This directory contains the implementation files for the CUDA orderbook.

## File Organization

### Team 1 Files
- **utils.cu** (Days 3-4)
  - Memory allocation/deallocation functions
  - CUDA error checking implementations
  - Data transfer utilities
  - Initialization functions
  
- **operations.cu** (Days 3-5 for add/cancel, Team 2 adds matching Days 3-8)
  - Device functions for orderbook operations
  - `add_order_device()` - Add order to orderbook
  - `cancel_order_device()` - Cancel/remove order
  - `remove_zero_neg_quant_device()` - Clean up invalid orders
  
- **kernels.cu** (Days 4-5 for basic kernels, all teams add more Days 4-8)
  - `add_order_batch_kernel()` - Batch add operations
  - `cancel_order_batch_kernel()` - Batch cancel operations
  - `init_orderbooks_kernel()` - Initialize orderbooks

### Team 2 Files
- **operations.cu** (continued, Days 3-8)
  - `get_top_ask_order_idx()` - Find best ask with price-time priority
  - `get_top_bid_order_idx()` - Find best bid with price-time priority  
  - `match_against_asks_device()` - Match buy order against asks
  - `match_against_bids_device()` - Match sell order against bids
  - `match_single_order_device()` - Execute one match, create trade

- **kernels.cu** (continued, Days 6-8)
  - `match_order_batch_kernel()` - Batch matching operations
  - `process_messages_sequential_kernel()` - **MAIN KERNEL** - Process message arrays

### Team 3 Files
- **queries.cu** (Days 2-4)
  - `get_best_ask_device()` - Find minimum ask price
  - `get_best_bid_device()` - Find maximum bid price
  - `get_volume_at_price_device()` - Sum volume at price level
  - `get_L2_state_device()` - Extract L2 orderbook snapshot

- **kernels.cu** (continued, Days 4-6)
  - `get_best_bid_ask_kernel()` - Query best prices
  - `get_volume_at_price_kernel()` - Query volume
  - `get_L2_state_kernel()` - Extract L2 states
  - `get_best_bid_ask_with_qty_kernel()` - Best prices with quantities

- **orderbook.cu** (Days 5-8)
  - `CudaOrderbook` class implementation
  - Constructor/destructor with memory management
  - Host API methods wrapping kernel calls
  - Data transfer between host and device

## Implementation Template

### Device Function Template
```cuda
#include "types.h"
#include "utils.cuh"

namespace cuda_orderbook {

__device__ void your_function_device(
    Order* orderside,
    const Message& msg,
    int n_orders
) {
    // Your implementation here
    // Remember: This runs on GPU, accessible from kernels
}

} // namespace cuda_orderbook
```

### Kernel Template
```cuda
#include "kernels.cuh"
#include "types.h"
#include "utils.cuh"

namespace cuda_orderbook {

__global__ void your_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    // Get this block's orderbook index
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get pointers to this orderbook's data
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    Trade* trades = batch.get_trades(book_idx);
    
    // Get this orderbook's message
    const Message& msg = messages[book_idx];
    
    // Your implementation here
    // Use threadIdx.x for parallel work within the orderbook
}

} // namespace cuda_orderbook
```

### Host Function Template (utils.cu, orderbook.cu)
```cpp
#include "utils.cuh"
#include "types.h"

namespace cuda_orderbook {

bool your_host_function(/* params */) {
    // This runs on CPU, can call CUDA APIs
    
    // Example: Allocate memory
    void* d_ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, size));
    
    // Example: Launch kernel
    dim3 grid(num_books);
    dim3 block(256);
    your_kernel<<<grid, block>>>(args);
    CHECK_KERNEL_ERROR();
    
    // Example: Copy data
    CHECK_CUDA_ERROR(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
    
    return true;
}

} // namespace cuda_orderbook
```

## Adding Your Implementation

1. Create your `.cu` file in this directory
2. Include necessary headers:
   ```cpp
   #include "types.h"      // Data structures
   #include "kernels.cuh"  // Kernel declarations
   #include "utils.cuh"    // Utility functions
   ```
3. Wrap in namespace:
   ```cpp
   namespace cuda_orderbook {
   // Your code
   }
   ```
4. Update `CMakeLists.txt` to include your file
5. Build and test!

## Coordination

- **Team 1 → Team 2:** Team 2 needs `add_order_device` and `cancel_order_device` from operations.cu
- **Team 2 → Team 3:** Team 3 needs all kernels working to implement API
- **Team 3 → All:** Provides API for testing

## Reference Implementation

All implementations should match behavior of:
`gymnax_exchange/jaxob/JaxOrderBookArrays.py`

Compare outputs with JAX for validation!

