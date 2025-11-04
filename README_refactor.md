# CUDA C++ Orderbook Refactor

## Overview

Convert the JAX orderbook implementation to pure CUDA C++ optimized for processing thousands of independent orderbooks in parallel on GPU. This is a significant project requiring creation of a complete C++/CUDA library from scratch.

## Architecture

### Directory Structure

```
gymnax_exchange/cuda_orderbook/
├── include/
│   ├── orderbook.cuh          # Main orderbook class
│   ├── types.h                # Data structures
│   ├── kernels.cuh            # CUDA kernel declarations
│   └── utils.cuh              # Helper functions
├── src/
│   ├── orderbook.cu           # Orderbook implementation
│   ├── kernels.cu             # CUDA kernel implementations
│   ├── operations.cu          # Add/cancel/match operations
│   ├── queries.cu             # Query operations (best bid/ask, L2)
│   └── utils.cu               # Utilities
├── examples/
│   ├── single_orderbook.cpp   # Example: single orderbook usage
│   └── batch_orderbooks.cpp   # Example: parallel batch processing
├── tests/
│   ├── test_operations.cpp    # Unit tests
│   └── test_batch.cpp         # Batch processing tests
├── CMakeLists.txt             # Build configuration
└── README.md                  # C++ library documentation
```

## Implementation Plan

### Phase 1: Core Data Structures (types.h)

Define C++ equivalents of JAX data structures:

```cpp
// Order format: [price, quantity, order_id, trader_id, time_sec, time_ns]
struct Order {
    int32_t price;
    int32_t quantity;
    int32_t order_id;
    int32_t trader_id;
    int32_t time_sec;
    int32_t time_ns;
};

// Message format: [type, side, quantity, price, trader_id, order_id, time_sec, time_ns]
struct Message {
    int32_t type;      // 1=limit, 2=cancel, 3=delete, 4=market
    int32_t side;      // -1=ask, 1=bid
    int32_t quantity;
    int32_t price;
    int32_t trader_id;
    int32_t order_id;
    int32_t time_sec;
    int32_t time_ns;
};

// Orderbook state for one book
struct OrderbookState {
    Order* asks;       // Device pointer
    Order* bids;       // Device pointer
    Order* trades;     // Device pointer
    int32_t n_orders;
    int32_t n_trades;
};

// Batch of orderbooks (main use case)
struct OrderbookBatch {
    Order* d_asks;     // Flattened array: [book0_orders, book1_orders, ...]
    Order* d_bids;
    Order* d_trades;
    int32_t num_books;
    int32_t n_orders_per_book;
    int32_t n_trades_per_book;
};
```

**Key Constants:**

- `INITID = -9000` (for L2 snapshot orders)
- `MAX_INT = 2147483647` (empty/sentinel value)

### Phase 2: CUDA Kernels for Parallel Orderbook Operations (kernels.cu)

Implement parallel kernels where each thread block handles one orderbook:

**2.1 Add Order Kernel**

```cpp
__global__ void add_order_batch_kernel(
    Order* asks, Order* bids, 
    Message* messages,
    int num_books, int n_orders
);
```

- Maps JAX `add_order` (JaxOrderBookArrays.py:32-37)
- Each block processes one orderbook
- Finds first empty slot (price == -1)
- Writes order atomically

**2.2 Cancel Order Kernel**

```cpp
__global__ void cancel_order_batch_kernel(
    Order* side_array,
    Message* messages, 
    int num_books, int n_orders
);
```

- Maps JAX `cancel_order` (JaxOrderBookArrays.py:52-65)
- Finds order by ID or by price for INITID orders
- Reduces quantity, removes if <= 0

**2.3 Match Order Kernel**

```cpp
__global__ void match_order_batch_kernel(
    Order* asks, Order* bids, Order* trades,
    Message* messages,
    int num_books, int n_orders, int n_trades
);
```

- Maps JAX `_match_against_bid_orders` and `_match_against_ask_orders` (lines 115-130)
- Iteratively matches against best price with price-time priority
- Uses while loop per thread block (acceptable since orderbooks are independent)
- Generates trade records

**2.4 Get Best Bid/Ask Kernel**

```cpp
__global__ void get_best_bid_ask_kernel(
    Order* asks, Order* bids,
    int32_t* best_asks, int32_t* best_bids,
    int num_books, int n_orders
);
```

- Parallel reduction within each block to find min ask / max bid
- Maps JAX `get_best_ask` (line 430) and `get_best_bid` (line 436)

**2.5 Get L2 State Kernel**

```cpp
__global__ void get_L2_state_kernel(
    Order* asks, Order* bids,
    int32_t* l2_states,
    int num_books, int n_orders, int n_levels
);
```

- Maps JAX `get_L2_state` (lines 525-545)
- Extract unique price levels (parallel sort/unique per block)
- Aggregate volumes at each level using parallel reduction

**2.6 Process Message Array Kernel**

```cpp
__global__ void process_messages_sequential_kernel(
    Order* asks, Order* bids, Order* trades,
    Message* messages,
    int num_books, int n_messages_per_book,
    int n_orders, int n_trades
);
```

- Maps JAX `scan_through_entire_array` (lines 265-267)
- Each thread block processes ALL messages for ONE orderbook sequentially
- Calls add/cancel/match based on message type using switch statement
- This is the main entry point mimicking `process_orders_array`

### Phase 3: Host API (orderbook.cuh / orderbook.cu)

C++ class interface for managing orderbooks:

```cpp
class CudaOrderbook {
public:
    // Constructor
    CudaOrderbook(int num_books, int n_orders, int n_trades);
    ~CudaOrderbook();
    
    // Initialization
    void init();  // Initialize empty orderbooks
    void reset_from_l2(int32_t* h_l2_data, int book_idx);  // From L2 snapshot
    void reset_from_arrays(Order* h_asks, Order* h_bids, int book_idx);  // From arrays
    
    // Core operations
    void process_messages(Message* h_messages, int num_messages, int book_idx);
    void process_messages_batch(Message* h_messages, int* num_messages_per_book);
    
    // Queries
    void get_best_bid_ask(int32_t* h_best_bids, int32_t* h_best_asks);
    void get_l2_state(int32_t* h_l2_states, int n_levels);
    void get_volume_at_price(int book_idx, int side, int price, int32_t* volume);
    
    // State access
    void get_orderbook_state(int book_idx, Order* h_asks, Order* h_bids);
    void get_trades(int book_idx, Order* h_trades);
    
private:
    OrderbookBatch batch_;
    cudaStream_t stream_;
    
    // Internal helpers
    void allocate_device_memory();
    void free_device_memory();
};
```

Maps to JAX `OrderBook` class (jorderbook.py:21-239) methods.

### Phase 4: Sequential Operations Per Orderbook (operations.cu)

Device functions callable from kernels:

```cpp
__device__ void add_order_device(
    Order* orderside, const Message& msg, int n_orders
);

__device__ void cancel_order_device(
    Order* orderside, const Message& msg, int n_orders
);

__device__ void match_against_asks_device(
    Order* asks, Order* bids, Order* trades,
    const Message& msg, int n_orders, int n_trades
);

__device__ void match_against_bids_device(
    Order* asks, Order* bids, Order* trades,
    const Message& msg, int n_orders, int n_trades
);

__device__ int get_top_ask_order_idx(Order* asks, int n_orders);
__device__ int get_top_bid_order_idx(Order* bids, int n_orders);
```

These are the core matching engine logic, executed within each thread block processing an orderbook.

### Phase 5: Query Operations (queries.cu)

Implement device functions for querying orderbook state:

```cpp
__device__ int32_t get_best_ask_device(Order* asks, int n_orders);
__device__ int32_t get_best_bid_device(Order* bids, int n_orders);
__device__ int32_t get_volume_at_price_device(Order* side, int price, int n_orders);
__device__ void get_L2_state_device(Order* asks, Order* bids, int32_t* l2_out, int n_orders, int n_levels);
```

### Phase 6: Build System (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.18)
project(CudaOrderbook CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_orderbook STATIC
    src/orderbook.cu
    src/kernels.cu
    src/operations.cu
    src/queries.cu
    src/utils.cu
)

target_include_directories(cuda_orderbook PUBLIC include)
target_link_libraries(cuda_orderbook CUDA::cudart)

# Examples
add_executable(single_example examples/single_orderbook.cpp)
target_link_libraries(single_example cuda_orderbook)

add_executable(batch_example examples/batch_orderbooks.cpp)
target_link_libraries(batch_example cuda_orderbook)

# Tests (optional: use Google Test)
# add_executable(tests tests/test_operations.cpp tests/test_batch.cpp)
# target_link_libraries(tests cuda_orderbook gtest gtest_main)
```

### Phase 7: Examples

**single_orderbook.cpp**: Demonstrates processing messages on a single orderbook

**batch_orderbooks.cpp**: Demonstrates parallel processing of 10,000 orderbooks (main use case)

Example usage:

```cpp
int main() {
    // Create 10,000 orderbooks
    CudaOrderbook orderbooks(10000, 100, 100);
    orderbooks.init();
    
    // Prepare messages (one per orderbook)
    Message* messages = new Message[10000];
    // ... fill messages ...
    
    // Process in parallel
    orderbooks.process_messages_batch(messages, nullptr);
    
    // Query results
    int32_t* best_bids = new int32_t[10000];
    int32_t* best_asks = new int32_t[10000];
    orderbooks.get_best_bid_ask(best_bids, best_asks);
    
    return 0;
}
```

### Phase 8: Testing & Validation

Create test cases comparing CUDA output to JAX reference implementation:

1. Add order tests
2. Cancel order tests  
3. Match order tests (limit/market)
4. Sequential message processing
5. Batch processing correctness
6. L2 state extraction
7. Edge cases (empty book, full book, INITID orders)

## Key Design Decisions

### Parallelization Strategy

- **Across orderbooks**: Each thread block processes one complete orderbook
- **Within orderbook**: Sequential message processing (unavoidable dependency)
- **Optimize for**: 1,000-10,000+ independent orderbooks

### Memory Management

- **Coalesced access**: Structure orderbook arrays as AoS within each book
- **Pitched allocation**: Use `cudaMallocPitch` for 2D order arrays
- **Streams**: Use CUDA streams for overlapping compute and data transfer

### Performance Optimizations

- Shared memory for orderbook state within thread block
- Warp-level primitives for reductions (best bid/ask)
- Bank conflict avoidance in shared memory accesses
- Register pressure minimization

## Implementation Effort

Estimated complexity: **4-6 weeks full-time for experienced CUDA developer**

- Phase 1 (Structures): 1-2 days
- Phase 2 (Kernels): 1-2 weeks (core complexity)
- Phase 3 (API): 3-5 days
- Phase 4-5 (Operations/Queries): 1 week
- Phase 6 (Build): 1 day
- Phase 7 (Examples): 2-3 days
- Phase 8 (Testing): 1 week

## Files to Create

New directory: `gymnax_exchange/cuda_orderbook/`

Core files (11 new files):

- `include/types.h`
- `include/orderbook.cuh`
- `include/kernels.cuh`
- `include/utils.cuh`
- `src/orderbook.cu`
- `src/kernels.cu`
- `src/operations.cu`
- `src/queries.cu`
- `src/utils.cu`
- `CMakeLists.txt`
- `README.md`

Example files (2):

- `examples/single_orderbook.cpp`
- `examples/batch_orderbooks.cpp`

Test files (2):

- `tests/test_operations.cpp`
- `tests/test_batch.cpp`

Total: 15 new files, ~5000-8000 lines of CUDA/C++ code

## Next Steps

After plan approval:

1. Create directory structure
2. Implement Phase 1 (data structures)
3. Implement Phase 2 (CUDA kernels) - iterative development
4. Implement Phase 3 (host API)
5. Create examples and validate against JAX
6. Performance profiling and optimization
