# Tests

This directory will contain test programs to validate CUDA orderbook correctness.

## Planned Tests (Team 3: Days 7-10)

### test_operations.cpp
Unit tests for individual operations:
- ✓ Add order - verify order added correctly
- ✓ Cancel order - verify quantity reduced/removed
- ✓ Match limit order - verify matching and trade generation
- ✓ Match market order - verify aggressive matching
- ✓ Price-time priority - verify correct order selection
- ✓ Edge cases - empty book, full book, INITID orders

### test_batch.cpp
Integration tests comparing CUDA to JAX reference:
- ✓ Sequential message processing - match JAX output exactly
- ✓ Batch processing - verify all orderbooks correct
- ✓ L2 state extraction - compare to JAX L2 state
- ✓ Complex scenarios - multiple operations, interleaved messages
- ✓ Performance benchmarks - throughput measurements

## Test Strategy

### Validation Against JAX
All CUDA outputs must match JAX reference implementation:

```cpp
// 1. Create same initial state in CUDA and JAX
// 2. Apply same messages
// 3. Compare outputs:
//    - Orderbook state (all orders)
//    - Trade records
//    - Best bid/ask
//    - L2 state
```

### Test Data
Use realistic market data:
- Price levels: 99,000 - 101,000 (realistic spread)
- Quantities: 1 - 1,000
- Order IDs: Sequential or random
- Timestamps: Microsecond precision

## Test Template

```cpp
#include "orderbook.cuh"
#include "types.h"
#include <cassert>
#include <iostream>

using namespace cuda_orderbook;

// Helper function to compare orderbooks
bool compare_orderbooks(
    const Order* cuda_orders,
    const int32_t* jax_orders,  // From numpy array
    int n_orders
) {
    for (int i = 0; i < n_orders; i++) {
        if (cuda_orders[i].price != jax_orders[i * 6 + 0]) return false;
        if (cuda_orders[i].quantity != jax_orders[i * 6 + 1]) return false;
        // ... check other fields
    }
    return true;
}

// Test: Add Order
void test_add_order() {
    std::cout << "Testing add_order..." << std::endl;
    
    // Setup
    CudaOrderbook ob(1, 100, 100);
    ob.init();
    
    // Create message
    Message msg;
    msg.type = Message::LIMIT;
    msg.side = Message::BID;
    msg.quantity = 100;
    msg.price = 100000;
    msg.order_id = 1001;
    msg.trader_id = 1;
    msg.time_sec = 34200;
    msg.time_ns = 0;
    
    // Process
    ob.process_messages(&msg, 1, 0);
    
    // Verify
    Order* h_bids = new Order[100];
    ob.get_orderbook_state(0, nullptr, h_bids);
    
    assert(h_bids[0].price == 100000);
    assert(h_bids[0].quantity == 100);
    assert(h_bids[0].order_id == 1001);
    
    delete[] h_bids;
    std::cout << "✓ test_add_order passed" << std::endl;
}

// Test: Compare to JAX
void test_compare_to_jax() {
    std::cout << "Testing CUDA vs JAX..." << std::endl;
    
    // 1. Load JAX reference output (from numpy array or file)
    // int32_t* jax_asks = load_jax_output("jax_asks.npy");
    // int32_t* jax_bids = load_jax_output("jax_bids.npy");
    
    // 2. Run same operations in CUDA
    CudaOrderbook ob(1, 100, 100);
    // ... apply same messages ...
    
    // 3. Compare
    Order* cuda_asks = new Order[100];
    Order* cuda_bids = new Order[100];
    ob.get_orderbook_state(0, cuda_asks, cuda_bids);
    
    // assert(compare_orderbooks(cuda_asks, jax_asks, 100));
    // assert(compare_orderbooks(cuda_bids, jax_bids, 100));
    
    delete[] cuda_asks;
    delete[] cuda_bids;
    std::cout << "✓ test_compare_to_jax passed" << std::endl;
}

int main() {
    std::cout << "Running CUDA Orderbook Tests..." << std::endl;
    std::cout << "================================" << std::endl;
    
    test_add_order();
    test_compare_to_jax();
    // ... more tests ...
    
    std::cout << "================================" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

## Running Tests

```bash
cd build
make test_operations
make test_batch

# Run tests
./test_operations
./test_batch

# With verbose output
./test_batch --verbose

# Specific test
./test_operations --test add_order
```

## Validation Checklist

Before declaring DONE on Day 10:

- [ ] Add order works correctly
- [ ] Cancel order works correctly
- [ ] Match limit orders works correctly
- [ ] Match market orders works correctly
- [ ] Price-time priority is correct
- [ ] Batch processing all books correctly
- [ ] Sequential message processing matches JAX
- [ ] L2 state extraction matches JAX
- [ ] Best bid/ask queries correct
- [ ] Volume queries correct
- [ ] Edge cases handled (empty, full, INITID)
- [ ] Performance meets requirements (>1000 books/sec)

## Performance Targets

- **Throughput:** Process 10,000 orderbooks with 100 messages each in < 1 second
- **Latency:** Single orderbook processing < 100 microseconds
- **Scalability:** Linear scaling from 1,000 to 100,000 orderbooks

## Integration with JAX Testing

### Export JAX Test Data
```python
import numpy as np
from gymnax_exchange.jaxob.jorderbook import OrderBook

# Create test case
ob = OrderBook(nOrders=100, nTrades=100)
state = ob.init()

# Apply messages
msgs = jnp.array([...])  # Test messages
state = ob.process_orders_array(state, msgs)

# Save for CUDA testing
np.save('test_asks.npy', state.asks)
np.save('test_bids.npy', state.bids)
np.save('test_trades.npy', state.trades)
```

### Load in C++ Tests
```cpp
// Use cnpy library or write simple numpy loader
// Compare CUDA output to loaded JAX output
```

