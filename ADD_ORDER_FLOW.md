# ADD ORDER MESSAGE FLOW

## 📊 VISUAL FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER SUBMITS ADD ORDER                       │
│   Message{ type=LIMIT, side=BID/ASK, price, qty, order_id }    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: KERNEL LAUNCH (GPU Entry Point)            │
│                                                                 │
│  process_messages_sequential_kernel<<<grid, block>>>()         │
│  📍 src/kernels.cu:186-225                                      │
│                                                                 │
│  • One block per orderbook (blockIdx.x = book_idx)             │
│  • Only thread 0 processes (threadIdx.x == 0)                  │
│  • Messages processed sequentially within each block           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: MESSAGE ROUTING (Dispatch by Type/Side)         │
│                                                                 │
│  process_message_device(asks, bids, trades, msg, ...)         │
│  📍 src/operations.cu:382-467                                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ if (msg.type == LIMIT)        # Line 402                 │  │
│  │     if (msg.side == BID)       # Line 428                │  │
│  │         ⟶ BUY LIMIT ORDER FLOW                           │  │
│  │     elif (msg.side == ASK)     # Line 404                │  │
│  │         ⟶ SELL LIMIT ORDER FLOW                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│    BUY LIMIT ORDER      │   │   SELL LIMIT ORDER      │
│   (msg.side == BID)     │   │   (msg.side == ASK)     │
│  Line 428-451           │   │  Line 404-427           │
└───────┬─────────────────┘   └─────────────┬───────────┘
        │                                   │
        │                                   │
        ▼                                   ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  STEP 3A: TRY MATCHING  │   │  STEP 3A: TRY MATCHING  │
│  Match against ASKS     │   │  Match against BIDS     │
│                         │   │                         │
│  match_against_asks()   │   │  match_against_bids()   │
│  📍 Line 291-324        │   │  📍 Line 333-369        │
└───────┬─────────────────┘   └─────────────┬───────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3B: ITERATIVE MATCHING (while loop)              │
│                                                                 │
│  while (qtm_remaining > 0 && matching orders exist) {          │
│      1. get_top_ask_order_idx() or get_top_bid_order_idx()     │
│         📍 Line 144-212 (finds best price-time order)          │
│                                                                 │
│      2. match_single_order_device(...)                         │
│         📍 Line 232-280                                         │
│         • Calculate matched quantity                            │
│         • Create Trade record                                   │
│         • Update passive order quantity                         │
│         • Clean up if fully matched                             │
│  }                                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        STEP 4: ADD REMAINDER TO BOOK (if any left)              │
│                                                                 │
│  Calculate: remaining = msg.quantity - matchable_qty           │
│  📍 Line 419-420 (ASK) or Line 443-444 (BID)                   │
│                                                                 │
│  if (remaining > 0) {                                          │
│      add_order_device(asks/bids, remaining_msg, n_orders)     │
│      📍 Line 49-79                                              │
│  }                                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: ADD ORDER TO BOOK ARRAY                    │
│                                                                 │
│  add_order_device(orderside, msg, n_orders)                    │
│  📍 src/operations.cu:49-79                                     │
│                                                                 │
│  1. Find first empty slot (price == EMPTY_PRICE)  # Line 54-61│
│     for (i = 0; i < n_orders; i++) {                           │
│         if (orderside[i].price == EMPTY_PRICE)                 │
│             empty_idx = i                                       │
│     }                                                           │
│                                                                 │
│  2. Insert order at empty_idx                      # Line 70-75│
│     orderside[empty_idx].price      = msg.price                │
│     orderside[empty_idx].quantity   = max(0, msg.quantity)     │
│     orderside[empty_idx].order_id   = msg.order_id             │
│     orderside[empty_idx].trader_id  = msg.trader_id            │
│     orderside[empty_idx].time_sec   = msg.time_sec             │
│     orderside[empty_idx].time_ns    = msg.time_ns              │
│                                                                 │
│  3. Clean up zero/negative quantities              # Line 78   │
│     remove_zero_neg_quant_device(orderside, n_orders)          │
│     📍 Line 24-36                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ORDER ADDED ✓                               │
│                                                                 │
│  Final State:                                                   │
│  • Order in asks[] or bids[] array                             │
│  • Trades generated (if matched)                               │
│  • Only unmatched remainder added to book                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 DETAILED CODE POINTERS

### **ENTRY POINT: GPU Kernel**

```54:79:src/operations.cu
// Find first empty slot (price == -1)
int empty_idx = -1;
for (int i = 0; i < n_orders; i++) {
    if (orderside[i].price == EMPTY_PRICE) {
        empty_idx = i;
        break;
    }
}

if (empty_idx == -1) {
    // Orderbook full - cannot add
    // In production, might want to handle this differently
    return;
}

// Add the order
orderside[empty_idx].price = msg.price;
orderside[empty_idx].quantity = max(0, msg.quantity);
orderside[empty_idx].order_id = msg.order_id;
orderside[empty_idx].trader_id = msg.trader_id;
orderside[empty_idx].time_sec = msg.time_sec;
orderside[empty_idx].time_ns = msg.time_ns;

// Clean up any orders with zero/negative quantity
remove_zero_neg_quant_device(orderside, n_orders);
```

**Key**: Finds first empty slot, inserts order, cleans up zeros

---

### **MESSAGE DISPATCHER**

```402:451:src/operations.cu
else if (msg.type == Message::LIMIT) {
    // Limit order - need to track remaining quantity after matching
    if (msg.side == Message::ASK) {
        // Sell limit: match against bids, then add remainder
        
        // Count initial bid volume at or above our price
        int32_t matchable_qty = 0;
        for (int i = 0; i < n_orders; i++) {
            if (bids[i].price != EMPTY_PRICE && bids[i].price >= msg.price) {
                matchable_qty += bids[i].quantity;
            }
        }
        
        // Match against bids
        match_against_bids_device(asks, bids, trades, msg, n_orders, n_trades);
        
        // Calculate remaining quantity (what wasn't matched)
        int32_t remaining = msg.quantity - matchable_qty;
        if (remaining < 0) remaining = 0;
        
        // Only add if there's remaining quantity
        if (remaining > 0) {
            Message remaining_msg = msg;
            remaining_msg.quantity = remaining;
            add_order_device(asks, remaining_msg, n_orders);
        }
    } else if (msg.side == Message::BID) {
        // Buy limit: match against asks, then add remainder
        
        // Count initial ask volume at or below our price
        int32_t matchable_qty = 0;
        for (int i = 0; i < n_orders; i++) {
            if (asks[i].price != EMPTY_PRICE && asks[i].price <= msg.price) {
                matchable_qty += asks[i].quantity;
            }
        }
        
        // Match against asks
        match_against_asks_device(asks, bids, trades, msg, n_orders, n_trades);
        
        // Calculate remaining quantity (what wasn't matched)
        int32_t remaining = msg.quantity - matchable_qty;
        if (remaining < 0) remaining = 0;
        
        // Only add if there's remaining quantity
        if (remaining > 0) {
            Message remaining_msg = msg;
            remaining_msg.quantity = remaining;
            add_order_device(bids, remaining_msg, n_orders);
        }
    }
}
```

**Key**: LIMIT orders try to match first, then add remainder

---

### **MATCHING ENGINE (Example: Buy Limit)**

```291:324:src/operations.cu
__device__ void match_against_asks_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Match iteratively against best asks
    while (qtm_remaining > 0) {
        // Find best ask (lowest price, earliest time)
        int top_ask_idx = get_top_ask_order_idx(asks, n_orders);
        
        // Check if we can match
        if (top_ask_idx == -1) break;  // No asks available
        if (asks[top_ask_idx].price == EMPTY_PRICE) break;  // No valid ask
        if (asks[top_ask_idx].price > limit_price) break;  // Price too high
        
        // Match against this ask
        match_single_order_device(
            top_ask_idx,
            asks,
            qtm_remaining,
            trades,
            n_trades,
            msg.order_id,
            msg.time_sec,
            msg.time_ns,
            n_orders
        );
    }
}
```

**Key**: Iteratively matches against best opposing orders until quantity exhausted or no more matches

---

### **SINGLE ORDER MATCH**

```232:280:src/operations.cu
__device__ void match_single_order_device(
    int top_order_idx,
    Order* orderside,
    int32_t& qtm_remaining,
    Trade* trades,
    int n_trades,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns,
    int n_orders
) {
    if (top_order_idx < 0 || top_order_idx >= n_orders) return;
    if (qtm_remaining <= 0) return;
    
    Order& passive_order = orderside[top_order_idx];
    if (passive_order.price == EMPTY_PRICE) return;
    
    // Calculate matched quantity
    int32_t matched_qty = min(qtm_remaining, passive_order.quantity);
    int32_t new_quantity = max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity to match
    qtm_remaining = max(0, qtm_remaining - passive_order.quantity);
    
    // Find empty trade slot and record trade
    for (int i = 0; i < n_trades; i++) {
        if (trades[i].price == EMPTY_PRICE) {
            trades[i].price = passive_order.price;
            trades[i].quantity = matched_qty;
            trades[i].passive_order_id = passive_order.order_id;
            trades[i].aggressive_order_id = aggressive_order_id;
            trades[i].time_sec = time_sec;
            trades[i].time_ns = time_ns;
            break;
        }
    }
    
    // Update passive order quantity
    passive_order.quantity = new_quantity;
    
    // Clean up if quantity is zero
    if (new_quantity <= 0) {
        passive_order.price = EMPTY_PRICE;
        passive_order.order_id = 0;
        passive_order.trader_id = 0;
        passive_order.time_sec = 0;
        passive_order.time_ns = 0;
    }
}
```

**Key**: Matches qty, creates trade record, updates/removes passive order

---

### **PRICE-TIME PRIORITY: Finding Best Order**

```101:142:src/operations.cu
__device__ int get_top_ask_order_idx(const Order* asks, int n_orders) {
    int best_idx = -1;
    int32_t min_price = MAX_INT;
    int32_t min_time_sec = MAX_INT;
    int32_t min_time_ns = MAX_INT;
    
    for (int i = 0; i < n_orders; i++) {
        if (asks[i].price == EMPTY_PRICE) continue;
        
        bool is_better = false;
        
        // Price priority: lower is better for asks
        if (asks[i].price < min_price) {
            is_better = true;
        }
        // Time priority: if same price, earlier is better
        else if (asks[i].price == min_price) {
            if (asks[i].time_sec < min_time_sec) {
                is_better = true;
            } else if (asks[i].time_sec == min_time_sec && 
                       asks[i].time_ns < min_time_ns) {
                is_better = true;
            }
        }
        
        if (is_better) {
            best_idx = i;
            min_price = asks[i].price;
            min_time_sec = asks[i].time_sec;
            min_time_ns = asks[i].time_ns;
        }
    }
    
    return best_idx;
}
```

**Key**: Implements price-time priority (best price, then earliest time)

---

## 🎯 EXECUTION FLOW SUMMARY

### **For a BUY LIMIT ORDER at price P, quantity Q:**

1. **Kernel Launch**: `process_messages_sequential_kernel<<<num_books, 256>>>`
   - Each block processes one orderbook
   - Only thread 0 processes messages sequentially

2. **Message Routing**: `process_message_device()` routes to LIMIT/BID handler
   - Lines 428-451 in `operations.cu`

3. **Try Matching**: `match_against_asks_device()`
   - Lines 291-324
   - While loop: find best ask, match, repeat
   - Stops when: qty exhausted OR no asks at price ≤ P

4. **Match Individual Orders**: `match_single_order_device()`
   - Lines 232-280
   - Creates trade records
   - Updates passive order (ask side)
   - Removes if fully matched

5. **Add Remainder**: `add_order_device()` on bids
   - Lines 49-79
   - Only if unmatched quantity remains
   - Finds empty slot, inserts order

---

## ⚠️ IMPORTANT NOTES

### **Sequential Processing Within Orderbook**
- Messages for each orderbook MUST be processed sequentially
- Why? Each message depends on previous state (price-time priority)
- Only thread 0 processes; others are idle (lines 206-224 in kernels.cu)

### **Parallel Processing Across Orderbooks**
- Different orderbooks processed in parallel (one per block)
- Test suite now supports: `./test_suite --num-books 1000`
- This is where GPU wins: 1000 orderbooks in parallel!

### **Memory Layout**
- Orders stored as arrays: `Order asks[n_orders]`, `Order bids[n_orders]`
- Empty slots marked with `price == EMPTY_PRICE (-1)`
- Linear search for empty slots (no fancy data structures on GPU)

### **Matching vs Adding**
- **LIMIT orders**: Try to match, then add remainder
- **MARKET orders**: Only match, never add to book
- **Price-time priority**: Best price first, then earliest timestamp

---

## 📂 KEY FILES

| File | Lines | Purpose |
|------|-------|---------|
| `src/kernels.cu` | 186-225 | Entry point: `process_messages_sequential_kernel` |
| `src/operations.cu` | 382-467 | Message dispatcher: `process_message_device` |
| `src/operations.cu` | 49-79 | Add order: `add_order_device` |
| `src/operations.cu` | 291-324 | Match asks: `match_against_asks_device` |
| `src/operations.cu` | 333-369 | Match bids: `match_against_bids_device` |
| `src/operations.cu` | 232-280 | Single match: `match_single_order_device` |
| `src/operations.cu` | 101-142 | Best ask: `get_top_ask_order_idx` |
| `src/operations.cu` | 171-212 | Best bid: `get_top_bid_order_idx` |

---

## 🚀 TRY IT YOURSELF

Run tests with detailed output:
```bash
cd tests
./test_suite --functional-only --num-books 1 100
```

This will show you exactly how 100 messages are processed on 1 orderbook.

