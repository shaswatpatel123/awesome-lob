/**
 * CUDA Kernels for Orderbook Operations
 * 
 * Each kernel operates on a batch of orderbooks in parallel.
 * Each thread block processes ONE complete orderbook.
 * 
 * Team 1: Basic operations (add, cancel, init)
 * Team 2: Matching operations (match, process_messages_sequential)
 * Team 3: Query operations (best bid/ask, L2 state)
 */

#include "kernels.cuh"
#include "types.h"
#include "utils.cuh"

// Forward declare device functions from operations.cu
namespace cuda_orderbook {
    __device__ void add_order_device(const OrderbookState& state, bool is_ask_side, const Message& msg);
    __device__ void cancel_order_device(const OrderbookState& state, bool is_ask_side, const Message& msg);
    __device__ void match_against_asks_device(const OrderbookState& state, const Message& msg);
    __device__ void match_against_bids_device(const OrderbookState& state, const Message& msg);
    __device__ void process_message_device(const OrderbookState& state, const Message& msg);
}

namespace cuda_orderbook {

// ============================================================================
// TEAM 1: BASIC OPERATION KERNELS
// ============================================================================

/**
 * Initialize orderbooks to empty state
 * Each thread block handles one orderbook
 * Threads within block parallelize across orders
 */
__global__ void init_orderbooks_kernel(
    OrderbookBatch batch,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's arrays
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    Trade* trades = batch.get_trades(book_idx);
    
    // Get price-aware structures
    OrderMetadata* ask_metadata = batch.get_ask_metadata(book_idx);
    OrderMetadata* bid_metadata = batch.get_bid_metadata(book_idx);
    PriceBucket* ask_buckets = batch.get_ask_buckets(book_idx);
    PriceBucket* bid_buckets = batch.get_bid_buckets(book_idx);
    PriceMapEntry* ask_price_map = batch.get_ask_price_map(book_idx);
    PriceMapEntry* bid_price_map = batch.get_bid_price_map(book_idx);
    OrderIDMapEntry* ask_order_id_map = batch.get_ask_order_id_map(book_idx);
    OrderIDMapEntry* bid_order_id_map = batch.get_bid_order_id_map(book_idx);
    BestPriceTracker* ask_tracker = batch.get_ask_tracker(book_idx);
    BestPriceTracker* bid_tracker = batch.get_bid_tracker(book_idx);
    
    int n_orders = batch.n_orders_per_book;
    int n_trades = batch.n_trades_per_book;
    int n_buckets = batch.n_price_buckets_per_book;
    int price_map_size = batch.price_map_size;
    int order_id_map_size = batch.order_id_map_size;
    
    // Parallelize initialization across threads
    // Each thread initializes multiple orders/trades
    for (int i = threadIdx.x; i < n_orders; i += blockDim.x) {
        asks[i].price = EMPTY_PRICE;
        asks[i].quantity = 0;
        asks[i].order_id = 0;
        asks[i].trader_id = 0;
        asks[i].time_sec = 0;
        asks[i].time_ns = 0;
        
        bids[i].price = EMPTY_PRICE;
        bids[i].quantity = 0;
        bids[i].order_id = 0;
        bids[i].trader_id = 0;
        bids[i].time_sec = 0;
        bids[i].time_ns = 0;
        
        if (ask_metadata) {
            ask_metadata[i] = OrderMetadata();
        }
        if (bid_metadata) {
            bid_metadata[i] = OrderMetadata();
        }
    }
    
    for (int i = threadIdx.x; i < n_trades; i += blockDim.x) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
    
    // Initialize price buckets
    for (int i = threadIdx.x; i < n_buckets; i += blockDim.x) {
        if (ask_buckets) {
            ask_buckets[i] = PriceBucket();
        }
        if (bid_buckets) {
            bid_buckets[i] = PriceBucket();
        }
    }
    
    // Initialize price maps
    for (int i = threadIdx.x; i < price_map_size; i += blockDim.x) {
        if (ask_price_map) {
            ask_price_map[i] = PriceMapEntry();
        }
        if (bid_price_map) {
            bid_price_map[i] = PriceMapEntry();
        }
    }
    
    // Initialize order-ID maps
    for (int i = threadIdx.x; i < order_id_map_size; i += blockDim.x) {
        if (ask_order_id_map) {
            ask_order_id_map[i] = OrderIDMapEntry();
        }
        if (bid_order_id_map) {
            bid_order_id_map[i] = OrderIDMapEntry();
        }
    }
    
    // Initialize best price trackers (only thread 0)
    if (threadIdx.x == 0) {
        if (ask_tracker) {
            *ask_tracker = BestPriceTracker();
        }
        if (bid_tracker) {
            *bid_tracker = BestPriceTracker();
        }
    }
}

/**
 * Add orders to orderbooks in batch
 * Each thread block processes one orderbook
 * Single message per orderbook
 */
__global__ void add_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's state
    OrderbookState state = batch.get_state(book_idx);
    const Message& msg = messages[book_idx];
    
    // Only thread 0 processes (add_order is sequential within orderbook)
    if (threadIdx.x == 0) {
        if (msg.side == Message::ASK) {
            add_order_device(state, true, msg);  // is_ask_side = true
        } else if (msg.side == Message::BID) {
            add_order_device(state, false, msg);  // is_ask_side = false
        }
    }
}

/**
 * Cancel orders from orderbooks in batch
 * Each thread block processes one orderbook
 * Single message per orderbook
 */
__global__ void cancel_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's state
    OrderbookState state = batch.get_state(book_idx);
    const Message& msg = messages[book_idx];
    
    // Only thread 0 processes
    if (threadIdx.x == 0) {
        if (msg.side == Message::ASK) {
            cancel_order_device(state, true, msg);  // is_ask_side = true
        } else if (msg.side == Message::BID) {
            cancel_order_device(state, false, msg);  // is_ask_side = false
        }
    }
}

// ============================================================================
// TEAM 2: MATCHING ENGINE KERNELS
// ============================================================================

/**
 * Match orders in batch (limit and market orders)
 * Each thread block processes one orderbook
 * Single message per orderbook
 */
__global__ void match_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's state
    OrderbookState state = batch.get_state(book_idx);
    const Message& msg = messages[book_idx];
    
    // Only thread 0 processes (matching is sequential within orderbook)
    if (threadIdx.x == 0) {
        // Match based on message side
        if (msg.side == Message::BID) {
            // Buy order: match against asks
            match_against_asks_device(state, msg);
        } else if (msg.side == Message::ASK) {
            // Sell order: match against bids
            match_against_bids_device(state, msg);
        }
    }
}

/**
 * Process array of messages sequentially for each orderbook in parallel
 * THIS IS THE MAIN KERNEL - Entry point for message processing
 * 
 * Each thread processes ALL messages for ONE orderbook sequentially
 * Multiple orderbooks processed in parallel (one per thread)
 * 
 * Maps to JAX scan_through_entire_array (JaxOrderBookArrays.py:265-267)
 * 
 * CRITICAL: This is the most important kernel for the system!
 */
__global__ void process_messages_sequential_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_messages_per_book,
    int num_books
) {
    // Calculate global thread index (one orderbook per thread)
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_books) return;
    
    int book_idx = thread_idx;
    
    // Get this orderbook's state
    OrderbookState state = batch.get_state(book_idx);
    
    // Get this orderbook's message array
    // Messages are laid out as: [book0_msgs, book1_msgs, ..., bookN_msgs]
    const Message* book_messages = messages + (book_idx * num_messages_per_book);
    
    // Process each message in sequence (sequential dependency within orderbook)
    for (int msg_idx = 0; msg_idx < num_messages_per_book; msg_idx++) {
        const Message& msg = book_messages[msg_idx];
        
        // Skip empty/invalid messages (price == -1 or quantity == 0)
        if (msg.quantity <= 0 || msg.type == 0) continue;
        
        // Process this message
        process_message_device(state, msg);
    }
}

// ============================================================================
// TEAM 3: QUERY KERNELS (Stubs for now - Team 3 will implement)
// ============================================================================

/**
 * Get best bid and ask for all orderbooks in batch (price-aware version)
 * Each thread block handles one orderbook
 * Uses O(1) best price tracker lookup
 */
__global__ void get_best_bid_ask_kernel(
    const OrderbookBatch batch,
    int32_t* best_asks,
    int32_t* best_bids,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's state
    OrderbookState state = batch.get_state(book_idx);
    
    // Only thread 0 processes (best prices are already tracked)
    if (threadIdx.x == 0) {
        if (state.ask_tracker && state.ask_tracker->has_best_ask()) {
            best_asks[book_idx] = state.ask_tracker->best_ask_price;
        } else {
            best_asks[book_idx] = -1;
        }
        
        if (state.bid_tracker && state.bid_tracker->has_best_bid()) {
            best_bids[book_idx] = state.bid_tracker->best_bid_price;
        } else {
            best_bids[book_idx] = -1;
        }
    }
}

/**
 * Get volume at specific price level for all orderbooks
 */
__global__ void get_volume_at_price_kernel(
    const OrderbookBatch batch,
    const int32_t* prices,
    const int32_t* sides,
    int32_t* volumes,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    int32_t target_price = prices[book_idx];
    int32_t side = sides[book_idx];  // 0=ask, 1=bid
    
    // Get appropriate side
    const Order* orders = (side == 0) ? 
        batch.get_asks(book_idx) : 
        batch.get_bids(book_idx);
    
    int n_orders = batch.n_orders_per_book;
    
    // Shared memory for reduction
    __shared__ int32_t shared_volume;
    if (threadIdx.x == 0) {
        shared_volume = 0;
    }
    __syncthreads();
    
    // Each thread sums its portion
    int32_t local_volume = 0;
    for (int i = threadIdx.x; i < n_orders; i += blockDim.x) {
        if (orders[i].price == target_price) {
            local_volume += orders[i].quantity;
        }
    }
    
    // Atomic add to shared memory
    atomicAdd(&shared_volume, local_volume);
    __syncthreads();
    
    // Thread 0 writes result
    if (threadIdx.x == 0) {
        volumes[book_idx] = shared_volume;
    }
}

/**
 * Extract L2 orderbook state (top N price levels with volumes)
 * Simplified version - Team 3 will enhance
 */
__global__ void get_L2_state_kernel(
    const OrderbookBatch batch,
    int32_t* l2_states,
    int n_levels,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's arrays
    const Order* asks = batch.get_asks(book_idx);
    const Order* bids = batch.get_bids(book_idx);
    int n_orders = batch.n_orders_per_book;
    
    // Output format: [ask_p1, ask_q1, bid_p1, bid_q1, ..., ask_pN, ask_qN, bid_pN, bid_qN]
    int32_t* book_l2 = l2_states + (book_idx * n_levels * 4);
    
    // Only thread 0 processes (simplification)
    if (threadIdx.x == 0) {
        // Initialize output to -1 (empty)
        for (int i = 0; i < n_levels * 4; i++) {
            book_l2[i] = -1;
        }
        
        // This is a simplified version
        // Team 3 will implement proper price level aggregation
        // For now, just extract first n_levels orders
        
        int ask_count = 0;
        int bid_count = 0;
        
        for (int i = 0; i < n_orders && (ask_count < n_levels || bid_count < n_levels); i++) {
            // Collect asks
            if (ask_count < n_levels && asks[i].price != EMPTY_PRICE) {
                book_l2[ask_count * 4 + 0] = asks[i].price;
                book_l2[ask_count * 4 + 1] = asks[i].quantity;
                ask_count++;
            }
            
            // Collect bids
            if (bid_count < n_levels && bids[i].price != EMPTY_PRICE) {
                book_l2[bid_count * 4 + 2] = bids[i].price;
                book_l2[bid_count * 4 + 3] = bids[i].quantity;
                bid_count++;
            }
        }
    }
}

/**
 * Get best bid and ask with quantities
 */
__global__ void get_best_bid_ask_with_qty_kernel(
    const OrderbookBatch batch,
    int32_t* best_asks_with_qty,
    int32_t* best_bids_with_qty,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get this orderbook's arrays
    const Order* asks = batch.get_asks(book_idx);
    const Order* bids = batch.get_bids(book_idx);
    int n_orders = batch.n_orders_per_book;
    
    // Find best ask and accumulate volume
    int32_t best_ask_price = MAX_INT;
    int32_t best_ask_qty = 0;
    
    for (int i = 0; i < n_orders; i++) {
        if (asks[i].price != EMPTY_PRICE) {
            if (asks[i].price < best_ask_price) {
                best_ask_price = asks[i].price;
                best_ask_qty = asks[i].quantity;
            } else if (asks[i].price == best_ask_price) {
                best_ask_qty += asks[i].quantity;
            }
        }
    }
    
    // Find best bid and accumulate volume
    int32_t best_bid_price = -1;
    int32_t best_bid_qty = 0;
    
    for (int i = 0; i < n_orders; i++) {
        if (bids[i].price != EMPTY_PRICE) {
            if (bids[i].price > best_bid_price) {
                best_bid_price = bids[i].price;
                best_bid_qty = bids[i].quantity;
            } else if (bids[i].price == best_bid_price) {
                best_bid_qty += bids[i].quantity;
            }
        }
    }
    
    // Write results
    if (threadIdx.x == 0) {
        best_asks_with_qty[book_idx * 2 + 0] = (best_ask_price == MAX_INT) ? -1 : best_ask_price;
        best_asks_with_qty[book_idx * 2 + 1] = best_ask_qty;
        best_bids_with_qty[book_idx * 2 + 0] = best_bid_price;
        best_bids_with_qty[book_idx * 2 + 1] = best_bid_qty;
    }
}

/**
 * Copy orderbooks from source to destination
 */
__global__ void copy_orderbooks_kernel(
    const OrderbookBatch src_batch,
    OrderbookBatch dst_batch,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Get source and destination arrays
    const Order* src_asks = src_batch.get_asks(book_idx);
    const Order* src_bids = src_batch.get_bids(book_idx);
    const Trade* src_trades = src_batch.get_trades(book_idx);
    
    Order* dst_asks = dst_batch.get_asks(book_idx);
    Order* dst_bids = dst_batch.get_bids(book_idx);
    Trade* dst_trades = dst_batch.get_trades(book_idx);
    
    int n_orders = src_batch.n_orders_per_book;
    int n_trades = src_batch.n_trades_per_book;
    
    // Parallelize copy across threads
    for (int i = threadIdx.x; i < n_orders; i += blockDim.x) {
        dst_asks[i] = src_asks[i];
        dst_bids[i] = src_bids[i];
    }
    
    for (int i = threadIdx.x; i < n_trades; i += blockDim.x) {
        dst_trades[i] = src_trades[i];
    }
}

/**
 * Reset trades array to empty
 */
__global__ void reset_trades_kernel(
    OrderbookBatch batch,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    Trade* trades = batch.get_trades(book_idx);
    int n_trades = batch.n_trades_per_book;
    
    // Parallelize across threads
    for (int i = threadIdx.x; i < n_trades; i += blockDim.x) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
}

} // namespace cuda_orderbook

