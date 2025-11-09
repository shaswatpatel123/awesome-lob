# Examples

This directory will contain example programs demonstrating CUDA orderbook usage.

## Planned Examples (Team 3: Days 7-8)

### single_orderbook.cpp
Demonstrates basic usage with a single orderbook:
- Initialize one orderbook
- Add limit orders
- Cancel orders
- Match orders
- Query best bid/ask
- Extract L2 state

**Expected Usage:**
```bash
./single_example
# Output: Orderbook state after various operations
```

### batch_orderbooks.cpp
Demonstrates parallel batch processing (main use case):
- Initialize 1,000-10,000 orderbooks
- Process messages in parallel (one per orderbook)
- Query all orderbooks simultaneously
- Compare performance to sequential processing

**Expected Usage:**
```bash
./batch_example --num-books 10000 --num-messages 100
# Output: Performance metrics, throughput measurements
```

## Example Template

```cpp
#include "orderbook.cuh"
#include "types.h"
#include <iostream>
#include <chrono>

using namespace cuda_orderbook;

int main(int argc, char** argv) {
    // Parse arguments
    int num_books = 1000;
    int n_orders = 100;
    int n_trades = 100;
    
    // Create orderbooks
    CudaOrderbook orderbooks(num_books, n_orders, n_trades);
    
    // Initialize
    orderbooks.init();
    
    // Create messages
    Message* messages = new Message[num_books];
    // ... fill messages ...
    
    // Process
    auto start = std::chrono::high_resolution_clock::now();
    orderbooks.process_messages_batch(messages, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Query results
    int32_t* best_bids = new int32_t[num_books];
    int32_t* best_asks = new int32_t[num_books];
    orderbooks.get_best_bid_ask(best_bids, best_asks);
    
    // Print results
    std::cout << "Processed " << num_books << " orderbooks" << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time: " << duration.count() << " us" << std::endl;
    
    // Cleanup
    delete[] messages;
    delete[] best_bids;
    delete[] best_asks;
    
    return 0;
}
```

## Building Examples

After implementation, uncomment the example targets in CMakeLists.txt:
```cmake
add_executable(single_example examples/single_orderbook.cpp)
target_link_libraries(single_example cuda_orderbook)

add_executable(batch_example examples/batch_orderbooks.cpp)
target_link_libraries(batch_example cuda_orderbook)
```

Then build:
```bash
cd build
make single_example
make batch_example
./single_example
./batch_example
```

