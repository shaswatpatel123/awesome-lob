# CUDA Orderbook - Team Setup Guide

## 🚀 Foundation Complete! (Team 1 - Day 2)

The foundation infrastructure is ready for parallel development:

### ✅ Completed
- Directory structure (`include/`, `src/`, `examples/`, `tests/`)
- `include/types.h` - All data structures
- `include/kernels.cuh` - All kernel declarations
- `include/utils.cuh` - Utility function declarations
- `CMakeLists.txt` - Build system

### 📁 Directory Structure
```
refector/
├── include/
│   ├── types.h          ✅ Core data structures (Order, Message, OrderbookBatch)
│   ├── kernels.cuh      ✅ Kernel function declarations
│   └── utils.cuh        ✅ Utility function declarations
├── src/                 ⏳ Implementation files (teams to add)
│   ├── utils.cu         → Team 1 (Days 3-4)
│   ├── operations.cu    → Team 1 & 2 (Days 3-8)
│   ├── kernels.cu       → All teams (Days 4-8)
│   ├── queries.cu       → Team 3 (Days 2-4)
│   └── orderbook.cu     → Team 3 (Days 5-8)
├── examples/            ⏳ Example programs
│   ├── single_orderbook.cpp   → Team 3 (Days 7-8)
│   └── batch_orderbooks.cpp   → Team 3 (Days 7-8)
├── tests/               ⏳ Test programs
│   ├── test_operations.cpp    → Team 3 (Days 7-10)
│   └── test_batch.cpp         → Team 3 (Days 7-10)
└── CMakeLists.txt       ✅ Build configuration
```

---

## 🔥 Quick Start for Teams

### Team 1: Infrastructure & Add/Cancel Operations (YOU)
**Status:** Foundation complete, starting Days 3-5 work

**Next Steps:**
1. Implement `src/utils.cu` (memory allocation, CUDA helpers)
2. Implement `src/operations.cu` (add_order_device, cancel_order_device)
3. Implement `src/kernels.cu` (add_order_batch_kernel, cancel_order_batch_kernel)

**Key Files:**
- `include/utils.cuh` - Function declarations ready
- `include/kernels.cuh` - add_order_batch_kernel, cancel_order_batch_kernel declared

---

### Team 2: Matching Engine
**Status:** Can start preparation work now!

**Days 1-2 Tasks (NOW):**
1. Study JAX matching logic:
   - `gymnax_exchange/jaxob/JaxOrderBookArrays.py` lines 115-130
   - `_match_against_bid_orders` and `_match_against_ask_orders`
2. Design price-time priority algorithm on paper
3. Write pseudocode for matching logic

**Days 3+ (After Team 1 commits operations.cu):**
- Implement matching device functions in `src/operations.cu`:
  - `get_top_ask_order_idx()` - Find best ask with time priority
  - `get_top_bid_order_idx()` - Find best bid with time priority
  - `match_against_asks_device()` - Match buy orders vs asks
  - `match_against_bids_device()` - Match sell orders vs bids

**Reference Files:**
- `include/types.h` - See Order, Message, Trade structures
- `include/kernels.cuh` - See match_order_batch_kernel, process_messages_sequential_kernel

---

### Team 3: Queries, API & Validation
**Status:** Can start query implementation now!

**Days 2-3 Tasks (NOW):**
1. Implement `src/queries.cu` - Device query functions:
   ```cuda
   __device__ int32_t get_best_ask_device(Order* asks, int n_orders);
   __device__ int32_t get_best_bid_device(Order* bids, int n_orders);
   __device__ int32_t get_volume_at_price_device(Order* side, int price, int n_orders);
   __device__ void get_L2_state_device(Order* asks, Order* bids, int32_t* l2_out, ...);
   ```

2. Start API design for `include/orderbook.cuh`:
   ```cpp
   class CudaOrderbook {
   public:
       CudaOrderbook(int num_books, int n_orders, int n_trades);
       void init();
       void process_messages_batch(Message* h_messages, int* num_messages_per_book);
       void get_best_bid_ask(int32_t* h_best_bids, int32_t* h_best_asks);
       // ... more methods
   };
   ```

**Reference Files:**
- `include/types.h` - See OrderbookBatch, L2State structures
- `include/kernels.cuh` - See query kernel declarations
- `include/utils.cuh` - Memory management functions to use

---

## 🛠️ Building the Project

### Prerequisites
- CUDA Toolkit 11.0+ (with nvcc compiler)
- CMake 3.18+
- C++17 compatible compiler (g++, clang, MSVC)

### Build Instructions

```bash
# From refector/ directory
mkdir build
cd build

# Configure (adjust CUDA architecture for your GPU)
cmake ..

# Or specify architecture explicitly:
# cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..  # RTX 20xx
# cmake -DCMAKE_CUDA_ARCHITECTURES="86" ..  # RTX 30xx

# Build (currently just headers, will compile as you add .cu files)
make -j$(nproc)

# As you add source files, uncomment them in CMakeLists.txt
# and rebuild with:
make clean
make -j$(nproc)
```

### Adding Your Source Files

When you create a `.cu` file:

1. Open `CMakeLists.txt`
2. Uncomment your file in `CUDA_ORDERBOOK_SOURCES`
3. If it's the first source file, change library type:
   ```cmake
   # Change from:
   add_library(cuda_orderbook INTERFACE)
   
   # To:
   add_library(cuda_orderbook STATIC ${CUDA_ORDERBOOK_SOURCES})
   target_include_directories(cuda_orderbook PUBLIC include)
   target_link_libraries(cuda_orderbook PUBLIC CUDA::cudart)
   ```
4. Rebuild: `cd build && make clean && make -j$(nproc)`

---

## 📋 Key Data Structures (from types.h)

### Order
```cpp
struct Order {
    int32_t price;      // -1 = empty slot
    int32_t quantity;   
    int32_t order_id;   
    int32_t trader_id;  
    int32_t time_sec;   
    int32_t time_ns;    
};
```

### Message
```cpp
struct Message {
    int32_t type;       // 1=limit, 2=cancel, 3=delete, 4=market
    int32_t side;       // -1=ask, 1=bid
    int32_t quantity;   
    int32_t price;      
    int32_t trader_id;  
    int32_t order_id;   
    int32_t time_sec;   
    int32_t time_ns;    
};
```

### OrderbookBatch
```cpp
struct OrderbookBatch {
    Order* d_asks;      // Device array (flattened)
    Order* d_bids;      
    Trade* d_trades;    
    int32_t num_books;  
    int32_t n_orders_per_book;
    int32_t n_trades_per_book;
    
    // Helper methods
    Order* get_asks(int book_idx);  // Get specific book's asks
    Order* get_bids(int book_idx);  // Get specific book's bids
    Trade* get_trades(int book_idx);
};
```

---

## 🔑 Key Constants

```cpp
constexpr int32_t INITID = -9000;      // L2 snapshot order ID
constexpr int32_t MAX_INT = 2147483647; // Empty sentinel
constexpr int32_t EMPTY_PRICE = -1;    // Empty order indicator
```

---

## 🎯 Parallelization Strategy

**Core Concept:** Each thread block processes ONE complete orderbook

```cuda
__global__ void some_kernel(OrderbookBatch batch, ..., int num_books) {
    int book_idx = blockIdx.x;  // Each block = one orderbook
    if (book_idx >= num_books) return;
    
    // Get pointers to this orderbook's data
    Order* my_asks = batch.get_asks(book_idx);
    Order* my_bids = batch.get_bids(book_idx);
    Trade* my_trades = batch.get_trades(book_idx);
    
    // Process this orderbook (can use all threads in block)
    // ...
}
```

**Launch Configuration:**
- Grid dimension: `num_books` (one block per orderbook)
- Block dimension: Typically 128-256 threads
- Each block works independently (no inter-block sync needed)

---

## 🐛 Debugging Tips

### CUDA Error Checking
Always use the provided macros:
```cpp
#include "utils.cuh"

// For CUDA API calls
CHECK_CUDA_ERROR(cudaMalloc(&ptr, size));

// After kernel launches
my_kernel<<<grid, block>>>(args);
CHECK_KERNEL_ERROR();
```

### Printing from Kernels
```cpp
__global__ void debug_kernel(...) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Debug: value = %d\n", some_value);
    }
}
```

---

## 📞 Communication

### Daily Standups: 9 AM (15 minutes)
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers?

### Critical Integration Points
- **Day 2 EOD:** Team 1 commits foundation (DONE!)
- **Day 4 EOD:** Team 1 commits basic operations → Team 2 pulls
- **Day 6 EOD:** First integration test (all teams)
- **Day 8 EOD:** Full system functional test
- **Day 10 EOD:** DEMO

### Git Workflow
```bash
# Create your branch
git checkout -b team1-infra   # or team2-matching, team3-api

# Work and commit regularly
git add src/your_file.cu
git commit -m "Implement add_order_device"

# Push for review
git push origin team1-infra

# Merge to dev after review
git checkout dev
git merge team1-infra
```

---

## 📚 Reference JAX Implementation

All CUDA implementations map to JAX functions in:
`gymnax_exchange/jaxob/JaxOrderBookArrays.py`

Key mappings:
- `add_order` (line 32) → `add_order_device` in operations.cu
- `cancel_order` (line 52) → `cancel_order_device` in operations.cu
- `_match_against_bid_orders` (line 115) → `match_against_bids_device`
- `get_best_ask` (line 430) → `get_best_ask_device` in queries.cu
- `get_L2_state` (line 525) → `get_L2_state_device` in queries.cu

---

## ✅ Success Criteria

By Day 10, we must have:
1. ✅ Add/cancel/match operations working
2. ✅ Batch processing of 1000+ orderbooks
3. ✅ Sequential message processing per orderbook
4. ✅ Best bid/ask queries
5. ✅ Basic L2 state extraction
6. ✅ Examples demonstrating usage
7. ✅ Tests validating against JAX

---

## 🚨 Important Notes

1. **Interface Freeze:** types.h and kernels.cuh are now frozen
   - Any changes require all-team approval
   - Interface stability is critical for parallel work

2. **Work Independently:** Teams should not block each other
   - Use stub implementations if needed
   - Mock dependencies until real implementations available

3. **Test Early:** Don't wait for full integration
   - Team 1: Test add/cancel in isolation
   - Team 2: Test matching with simple data
   - Team 3: Test queries with mock orderbooks

4. **Cut Scope if Needed:** Working prototype > perfect code
   - Must-haves are prioritized
   - Nice-to-haves can be dropped

---

## 🎉 Let's Ship This!

Foundation is ready. All teams can start parallel development NOW!

Questions? Check the plan file or ask in standup.

**Next Commit Deadline:** EOD Day 4 (Team 1 basic operations)

