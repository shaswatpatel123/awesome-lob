# CPU Implementation of Limit Order Book

## Overview

The CPU implementation provides a sequential baseline for performance comparison against the GPU-accelerated version. This implementation processes market messages in a strictly sequential manner, implementing the core functionality of a limit order book matching engine with price-time priority.

## Purpose

This implementation serves three critical purposes:

1. **Performance Baseline**: Establishes single-threaded performance metrics for comparison
2. **Correctness Validation**: Provides ground truth results for GPU implementation testing
3. **Reference Implementation**: Demonstrates order book semantics in straightforward, readable code

## Architecture

### Core Data Structures

The implementation uses three primary structures stored in contiguous memory arrays:

#### 1. Order Structure
```cpp
struct Order {
    int32_t price;      // Price level (-1 = empty slot)
    int32_t quantity;   // Order size
    int32_t order_id;   // Unique identifier
    int32_t trader_id;  // Trader identifier
    int32_t time_sec;   // Timestamp (seconds)
    int32_t time_ns;    // Timestamp (nanoseconds)
}
```

- **Asks Array**: Sell orders stored in fixed-size array
- **Bids Array**: Buy orders stored in fixed-size array  
- **Empty Slots**: Marked with `EMPTY_PRICE = -1`

#### 2. Trade Structure
```cpp
struct Trade {
    int32_t price;             // Execution price
    int32_t quantity;          // Executed quantity
    int32_t passive_order_id;  // Resting order
    int32_t aggressive_order_id; // Incoming order
    int32_t time_sec;          // Execution time (seconds)
    int32_t time_ns;           // Execution time (nanoseconds)
}
```

- Records complete audit trail of all executions
- Maintains passive/aggressive order distinction

#### 3. Message Structure
```cpp
struct Message {
    int32_t type;      // 1=LIMIT, 2=CANCEL, 3=DELETE, 4=MARKET
    int32_t side;      // -1=ASK (sell), 1=BID (buy)
    int32_t quantity;  // Order quantity
    int32_t price;     // Limit price
    int32_t trader_id; // Trader ID
    int32_t order_id;  // Order ID
    int32_t time_sec;  // Timestamp (seconds)
    int32_t time_ns;   // Timestamp (nanoseconds)
}
```

### Memory Management

**OrderbookCPU Class**:
- Dynamic allocation using C++ `new[]` for orders and trades
- RAII pattern ensures automatic cleanup
- Fixed capacity determined at initialization

**OrderbookBatchCPU Class**:
- Manages multiple independent orderbooks
- Enables batch processing for fair GPU comparison
- Each orderbook processes sequentially

## Core Algorithms

### 1. Order Addition

**Function**: `add_order_cpu(Order* orderside, const Message& msg, int n_orders)`

**Algorithm**:
1. Linear scan to find first empty slot (price == -1)
2. Insert order at empty position
3. Clean up any zero-quantity orders

**Time Complexity**: O(n)

```cpp
// Pseudo-code
for i in 0..n_orders:
    if orderside[i].price == EMPTY_PRICE:
        orderside[i] = msg
        break
remove_zero_quantity_orders()
```

### 2. Order Cancellation

**Function**: `cancel_order_cpu(Order* orderside, const Message& msg, int n_orders)`

**Algorithm**:
1. Search by `order_id` (primary lookup)
2. If not found, search by `price` (for INITID snapshot orders)
3. Reduce quantity by cancel amount
4. Remove order if quantity reaches zero

**Time Complexity**: O(n) search + O(n) cleanup = O(n)

```cpp
// Pseudo-code
idx = find_order_by_id(msg.order_id)
if idx == -1:
    idx = find_order_by_price(msg.price)  // Fallback for INITID
orderside[idx].quantity -= msg.quantity
remove_zero_quantity_orders()
```

### 3. Price-Time Priority Matching

**Functions**: `get_top_ask_order_idx_cpu()`, `get_top_bid_order_idx_cpu()`

**Algorithm for Asks** (find best sell order):
1. Linear scan through all ask orders
2. Find **lowest price** (best ask)
3. Break ties using **earliest timestamp**

**Algorithm for Bids** (find best buy order):
1. Linear scan through all bid orders
2. Find **highest price** (best bid)
3. Break ties using **earliest timestamp**

**Time Complexity**: O(n) per call

```cpp
// Pseudo-code for best ask
best_idx = -1
min_price = MAX_INT
min_time = MAX_INT

for i in 0..n_orders:
    if asks[i].price < min_price:
        best_idx = i
        min_price = asks[i].price
        min_time = asks[i].timestamp
    else if asks[i].price == min_price and asks[i].timestamp < min_time:
        best_idx = i
        min_time = asks[i].timestamp

return best_idx
```

### 4. Order Matching Engine

**Functions**: `match_against_asks_cpu()`, `match_against_bids_cpu()`

**Algorithm for Buying (match against asks)**:
1. Find best ask using price-time priority
2. Check if ask price ≤ buyer's limit price
3. If yes, execute match at ask price
4. Update quantities, record trade
5. Repeat until quantity exhausted or no valid matches

**Algorithm for Selling (match against bids)**:
1. Find best bid using price-time priority
2. Check if bid price ≥ seller's limit price
3. If yes, execute match at bid price
4. Update quantities, record trade
5. Repeat until quantity exhausted or no valid matches

**Time Complexity**: O(k × (n + m)) where:
- k = number of matches
- n = orders per side
- m = max trades (for recording)

```cpp
// Pseudo-code for aggressive buy order
quantity_remaining = msg.quantity
while quantity_remaining > 0:
    best_ask = get_top_ask()  // O(n)
    
    if best_ask == null or best_ask.price > limit_price:
        break  // No more matches
    
    matched_qty = min(quantity_remaining, best_ask.quantity)
    record_trade(matched_qty, best_ask.price)  // O(m)
    
    best_ask.quantity -= matched_qty
    quantity_remaining -= matched_qty
    
    if best_ask.quantity == 0:
        remove_order(best_ask)
```

### 5. Message Processing Dispatcher

**Function**: `process_message_cpu()`

Routes messages to appropriate handlers based on type:

#### LIMIT Orders
1. Calculate total matchable quantity on opposite side
2. Perform aggressive matching
3. Add any remaining quantity to book

**Time Complexity**: O(n) + O(k×(n+m)) + O(n) = O(k×(n+m))

```cpp
// Pseudo-code for limit buy order
matchable_qty = count_asks_at_or_below(limit_price)  // O(n)
match_against_asks(msg)                              // O(k×(n+m))
remaining = msg.quantity - matchable_qty
if remaining > 0:
    add_order_to_bids(remaining)                     // O(n)
```

#### CANCEL/DELETE Orders
- Route to `cancel_order_cpu()` on appropriate side

**Time Complexity**: O(n)

#### MARKET Orders
- Set price to MAX_INT (buy) or 0 (sell)
- Match aggressively at any price
- Do not add remainder to book

**Time Complexity**: O(k×(n+m))

## Performance Analysis

### Time Complexity Summary

| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| Add Order | O(1) | O(n) | O(n) |
| Cancel Order | O(1) | O(n) | O(n) |
| Find Best Price | - | O(n) | O(n) |
| Match Single Order | O(m) | O(m) | O(m) |
| Process Limit Order | O(n) | O(k×(n+m)) | O(k×(n+m)) |
| Process N Messages | - | O(N×k×(n+m)) | O(N×k×(n+m)) |

Where:
- **n**: Maximum orders per side
- **m**: Maximum trades to record
- **k**: Number of matches per order
- **N**: Number of messages

### Space Complexity

- **Per Orderbook**: O(n) for asks + O(n) for bids + O(m) for trades = O(n + m)
- **Batch of B orderbooks**: O(B × (n + m))

### Performance Bottlenecks

1. **Best Price Selection**: O(n) linear scan every match
   - No sorted priority queue maintained
   - No heap or tree structure for fast retrieval

2. **Order Lookup for Cancellation**: O(n) linear search
   - No hash table for O(1) order_id lookup
   - No index maintained

3. **Empty Slot Search**: O(n) linear scan per addition
   - No free list of available slots
   - Fragmentation after cancellations

4. **Sequential Message Processing**
   - Messages processed one-at-a-time
   - No parallelism across messages
   - No parallelism across orderbooks in batch

5. **Cache Inefficiency**
   - Orders scattered in array (no locality)
   - Repeated full scans miss cache
   - No prefetching optimization

## Design Trade-offs

### Simplicity Over Performance

**Design Choice**: Straightforward algorithms with no complex data structures

**Advantages**:
- Easy to understand and verify correctness
- Minimal code complexity (~600 lines)
- Direct financial market semantics
- Serves as ground truth for GPU validation
- No synchronization or concurrency issues

**Disadvantages**:
- O(n) operations dominate for large orderbooks
- No algorithmic optimizations (heaps, indexes)
- Poor scalability with order count
- Sequential bottleneck prevents parallelism

### Array-Based Storage

**Design Choice**: Fixed-size arrays vs. dynamic containers (std::map, std::priority_queue)

**Advantages**:
- Matches GPU memory model (fixed allocations)
- Enables direct CPU-GPU result comparison
- Contiguous memory (theoretically cache-friendly)
- Predictable memory footprint
- No heap fragmentation

**Disadvantages**:
- Fixed capacity limits (orderbook can fill up)
- Memory waste after cancellations (sparse arrays)
- No automatic resizing
- Empty slot search still O(n)

### No Sorted Structures

**Design Choice**: Unsorted order arrays vs. maintaining sorted state

**Why Unsorted**:
- Matches GPU implementation constraints
- Avoids complex insertion/deletion in sorted array
- Simplifies cancel operations (no shift required)
- Fair comparison baseline

**Cost**:
- Every match requires O(n) best price scan
- No O(log n) heap operations
- Significant performance penalty at scale

## Batch Processing

### Single Orderbook Processing

```cpp
void process_messages_sequential_cpu(
    OrderbookCPU& book,
    const Message* messages,
    int num_messages
)
```

Processes messages sequentially through one orderbook.

### Batch Processing

```cpp
void process_messages_batch_cpu(
    OrderbookBatchCPU& batch,
    const Message* messages,
    int num_messages_per_book
)
```

Processes multiple independent orderbooks sequentially:
- Each orderbook processed in order (book 0, book 1, ...)
- Enables comparison with GPU parallel processing
- No inter-orderbook parallelism exploited

## Utility Functions

### Testing and Validation

**`copy_orderbook_cpu(src, dst)`**
- Efficient memory copy using `memcpy()`
- Copies asks, bids, and trades arrays

**`compare_orderbooks_cpu(book1, book2)`**
- Byte-by-byte comparison using `memcmp()`
- Validates CPU vs GPU results
- Returns true if orderbooks are identical

**`print_orderbook_cpu(book, max_orders)`**
- Human-readable orderbook state
- Displays top N asks, bids, and trades
- Useful for debugging and visualization

## Usage Example

```cpp
#include "orderbook_cpu.h"

// Create and allocate orderbook
cuda_orderbook::OrderbookCPU book;
book.allocate(1000, 5000);  // 1000 orders/side, 5000 trades

// Prepare messages
cuda_orderbook::Message msgs[3];
msgs[0] = {1, 1, 100, 9950, 1, 1001, 0, 0};  // Buy limit 100@9950
msgs[1] = {1, -1, 50, 10050, 2, 1002, 0, 1}; // Sell limit 50@10050
msgs[2] = {2, 1, 20, 9950, 1, 1001, 0, 2};   // Cancel 20 from order 1001

// Process messages
cuda_orderbook::process_messages_sequential_cpu(book, msgs, 3);

// Print results
cuda_orderbook::print_orderbook_cpu(book, 10);

// Cleanup automatic via destructor
```

## Files

- **Header**: `include/orderbook_cpu.h` - Interface definitions
- **Implementation**: `src/orderbook_cpu.cpp` - Core algorithms (~600 lines)
- **Types**: `include/types.h` - Shared data structures

## Benchmark Integration

The CPU implementation is integrated into the benchmarking framework:

```bash
cd benchmarks
make
./benchmark_cpu_vs_gpu --cpu-only
```

See `benchmarks/README.md` for full benchmarking guide.

## Comparison with GPU Implementation

| Aspect | CPU Implementation | GPU Implementation |
|--------|-------------------|-------------------|
| **Parallelism** | None (sequential) | Massive (thousands of threads) |
| **Best Price** | O(n) linear scan | O(log n) parallel reduction |
| **Order Lookup** | O(n) linear search | O(1) hash table |
| **Message Processing** | Sequential | Parallel across books |
| **Memory** | Host RAM | Device VRAM |
| **Complexity** | Simple algorithms | Complex synchronization |

The CPU implementation's O(n) operations become GPU's parallel reductions, and sequential processing becomes parallel batch execution.

## Key Insights

1. **Linear Operations Dominate**: O(n) scans for best price and cancellations are the primary bottleneck
2. **No Index Structures**: Absence of hash tables and priority queues causes poor scalability
3. **Sequential Bottleneck**: Single-threaded processing cannot exploit modern CPU parallelism
4. **Correctness First**: Design prioritizes clarity and correctness over performance
5. **Fair Baseline**: Provides realistic comparison point for GPU acceleration benefits

## Performance Expectations

For typical parameters:
- **Orders per side**: 1,000 - 10,000
- **Messages**: 10,000 - 100,000
- **Matches per message**: 1-5 average

**Expected CPU throughput**: 10K - 100K messages/second
**Expected GPU speedup**: 10x - 100x (depending on batch size and workload)

The performance gap widens with:
- Larger orderbook size (n increases)
- More matches per message (k increases)
- Larger batch sizes (more parallelism available)

## Conclusion

The CPU implementation establishes a sequential baseline that demonstrates:
- **What**: Complete limit order book functionality with price-time priority
- **How**: Straightforward algorithms using simple data structures
- **Why**: Provides correctness validation and performance comparison

Its O(n) operations and sequential processing represent traditional single-threaded orderbook performance—precisely the bottlenecks that GPU parallelization addresses. This makes it an ideal reference for evaluating the benefits of GPU acceleration in financial market simulation workloads.

## References

- Main Project README: `../README.md`
- GPU Implementation: `../src/kernels.cu`, `../src/operations.cu`
- Benchmark Guide: `../benchmarks/README.md`
- Type Definitions: `../include/types.h`

