/**
 * CUDA Kernels for Orderbook Operations - WARP LEVEL
 * 
 * Each kernel operates on a batch of orderbooks in parallel.
 * Each WARP (32 threads) processes ONE complete orderbook.
 * 
 * Architecture:
 * - 1 LOB per warp (32 threads)
 * - Multiple warps per block for efficiency
 * - Warp-level primitives for synchronization and reduction
 */

#include "kernels.cuh"
#include "types.h"
#include "utils.cuh"

// Forward declare device functions from operations.cu
namespace cuda_orderbook {
    __device__ void add_order_warp(Order* orderside, const Message& msg, int n_orders, int laneId);
    __device__ void cancel_order_warp(Order* orderside, const Message& msg, int n_orders, int laneId);
    __device__ int32_t match_against_asks_warp(Order* asks, Order* bids, Trade* trades, const Message& msg, int n_orders, int n_trades, int laneId);
    __device__ int32_t match_against_bids_warp(Order* asks, Order* bids, Trade* trades, const Message& msg, int n_orders, int n_trades, int laneId);
    __device__ void process_message_warp(Order* asks, Order* bids, Trade* trades, const Message& msg, int n_orders, int n_trades, int laneId);
}

namespace cuda_orderbook {

// Warp size constant
constexpr int WARP_SIZE = 32;

// Helper: Get warp ID within block
__device__ inline int get_warp_id() {
    return threadIdx.x / WARP_SIZE;
}

// Helper: Get lane ID within warp
__device__ inline int get_lane_id() {
    return threadIdx.x % WARP_SIZE;
}

// Helper: Get global book index for this warp
__device__ inline int get_book_idx(int num_books) {
    int warps_per_block = blockDim.x / WARP_SIZE;
    int book_idx = blockIdx.x * warps_per_block + get_warp_id();
    return (book_idx < num_books) ? book_idx : -1;
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * Initialize orderbooks to empty state
 * Each warp handles one orderbook
 * Threads within warp parallelize across orders
 */
__global__ void init_orderbooks_kernel(
    OrderbookBatch batch,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's arrays
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    Trade* trades = batch.get_trades(book_idx);
    
    int n_orders = batch.n_orders_per_book;
    int n_trades = batch.n_trades_per_book;
    
    // Parallelize initialization across warp lanes
    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
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
    }
    
    for (int i = laneId; i < n_trades; i += WARP_SIZE) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
}

/**
 * Add orders to orderbooks in batch
 * Each warp processes one orderbook
 */
__global__ void add_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's data
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    const Message& msg = messages[book_idx];
    
    // All lanes participate, only lane 0 modifies state
    if (msg.side == Message::ASK) {
        add_order_warp(asks, msg, batch.n_orders_per_book, laneId);
    } else if (msg.side == Message::BID) {
        add_order_warp(bids, msg, batch.n_orders_per_book, laneId);
    }
}

/**
 * Cancel orders from orderbooks in batch
 * Each warp processes one orderbook
 */
__global__ void cancel_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's data
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    const Message& msg = messages[book_idx];
    
    // All lanes participate
    if (msg.side == Message::ASK) {
        cancel_order_warp(asks, msg, batch.n_orders_per_book, laneId);
    } else if (msg.side == Message::BID) {
        cancel_order_warp(bids, msg, batch.n_orders_per_book, laneId);
    }
}

// ============================================================================
// MATCHING ENGINE KERNELS
// ============================================================================

/**
 * Match orders in batch (limit and market orders)
 * Each warp processes one orderbook
 */
__global__ void match_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's data
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    Trade* trades = batch.get_trades(book_idx);
    const Message& msg = messages[book_idx];
    
    // Match based on message side
    if (msg.side == Message::BID) {
        // Buy order: match against asks
        match_against_asks_warp(asks, bids, trades, msg, 
                               batch.n_orders_per_book, 
                               batch.n_trades_per_book,
                               laneId);
    } else if (msg.side == Message::ASK) {
        // Sell order: match against bids
        match_against_bids_warp(asks, bids, trades, msg,
                               batch.n_orders_per_book,
                               batch.n_trades_per_book,
                               laneId);
    }
}

/**
 * Process array of messages sequentially for each orderbook in parallel
 * THIS IS THE MAIN KERNEL
 * 
 * Each warp processes ALL messages for ONE orderbook sequentially
 * Multiple orderbooks processed in parallel (one per warp)
 */
__global__ void process_messages_sequential_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_messages_per_book,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's arrays
    Order* asks = batch.get_asks(book_idx);
    Order* bids = batch.get_bids(book_idx);
    Trade* trades = batch.get_trades(book_idx);
    
    // Get this orderbook's message array
    const Message* book_messages = messages + (book_idx * num_messages_per_book);
    
    // Process each message in sequence
    // All lanes participate in warp-level operations
    for (int msg_idx = 0; msg_idx < num_messages_per_book; msg_idx++) {
        // Lane 0 loads the message, broadcast to all lanes
        Message msg;
        if (laneId == 0) {
            msg = book_messages[msg_idx];
        }
        
        // Broadcast message to all lanes using shuffle
        msg.type = __shfl_sync(0xFFFFFFFF, msg.type, 0);
        msg.side = __shfl_sync(0xFFFFFFFF, msg.side, 0);
        msg.quantity = __shfl_sync(0xFFFFFFFF, msg.quantity, 0);
        msg.price = __shfl_sync(0xFFFFFFFF, msg.price, 0);
        msg.trader_id = __shfl_sync(0xFFFFFFFF, msg.trader_id, 0);
        msg.order_id = __shfl_sync(0xFFFFFFFF, msg.order_id, 0);
        msg.time_sec = __shfl_sync(0xFFFFFFFF, msg.time_sec, 0);
        msg.time_ns = __shfl_sync(0xFFFFFFFF, msg.time_ns, 0);
        
        // Skip empty/invalid messages
        if (msg.quantity <= 0 || msg.type == 0) continue;
        
        // All lanes call this (required for warp-level operations)
        process_message_warp(
            asks, 
            bids, 
            trades, 
            msg,
            batch.n_orders_per_book,
            batch.n_trades_per_book,
            laneId
        );
    }
}

// ============================================================================
// QUERY KERNELS
// ============================================================================

/**
 * Get best bid and ask for all orderbooks in batch
 * Each warp handles one orderbook
 * Uses warp-level reduction to find min/max
 */
__global__ void get_best_bid_ask_kernel(
    const OrderbookBatch batch,
    int32_t* best_asks,
    int32_t* best_bids,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's arrays
    const Order* asks = batch.get_asks(book_idx);
    const Order* bids = batch.get_bids(book_idx);
    int n_orders = batch.n_orders_per_book;
    
    // Each lane processes multiple orders
    int32_t local_min_ask = MAX_INT;
    int32_t local_max_bid = -1;
    
    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
        if (asks[i].price != EMPTY_PRICE) {
            local_min_ask = min(local_min_ask, asks[i].price);
        }
        if (bids[i].price != EMPTY_PRICE) {
            local_max_bid = max(local_max_bid, bids[i].price);
        }
    }
    
    // Warp-level reduction for min ask
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        int32_t other_ask = __shfl_down_sync(0xFFFFFFFF, local_min_ask, offset);
        local_min_ask = min(local_min_ask, other_ask);
    }
    
    // Warp-level reduction for max bid
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        int32_t other_bid = __shfl_down_sync(0xFFFFFFFF, local_max_bid, offset);
        local_max_bid = max(local_max_bid, other_bid);
    }
    
    // Lane 0 writes results
    if (laneId == 0) {
        best_asks[book_idx] = (local_min_ask == MAX_INT) ? -1 : local_min_ask;
        best_bids[book_idx] = local_max_bid;
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
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    int32_t target_price = prices[book_idx];
    int32_t side = sides[book_idx];
    
    // Get appropriate side
    const Order* orders = (side == 0) ? 
        batch.get_asks(book_idx) : 
        batch.get_bids(book_idx);
    
    int n_orders = batch.n_orders_per_book;
    
    // Each lane sums its portion
    int32_t local_volume = 0;
    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
        if (orders[i].price == target_price) {
            local_volume += orders[i].quantity;
        }
    }
    
    // Warp-level reduction for sum
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_volume += __shfl_down_sync(0xFFFFFFFF, local_volume, offset);
    }
    
    // Lane 0 writes result
    if (laneId == 0) {
        volumes[book_idx] = local_volume;
    }
}

/**
 * Extract L2 orderbook state (top N price levels with volumes)
 */
__global__ void get_L2_state_kernel(
    const OrderbookBatch batch,
    int32_t* l2_states,
    int n_levels,
    int num_books
) {
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's arrays
    const Order* asks = batch.get_asks(book_idx);
    const Order* bids = batch.get_bids(book_idx);
    int n_orders = batch.n_orders_per_book;
    
    // Output format: [ask_p1, ask_q1, bid_p1, bid_q1, ...]
    int32_t* book_l2 = l2_states + (book_idx * n_levels * 4);
    
    // Lane 0 processes (simplified version)
    if (laneId == 0) {
        // Initialize to -1
        for (int i = 0; i < n_levels * 4; i++) {
            book_l2[i] = -1;
        }
        
        // Extract first n_levels orders
        int ask_count = 0;
        int bid_count = 0;
        
        for (int i = 0; i < n_orders && (ask_count < n_levels || bid_count < n_levels); i++) {
            if (ask_count < n_levels && asks[i].price != EMPTY_PRICE) {
                book_l2[ask_count * 4 + 0] = asks[i].price;
                book_l2[ask_count * 4 + 1] = asks[i].quantity;
                ask_count++;
            }
            
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
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get this orderbook's arrays
    const Order* asks = batch.get_asks(book_idx);
    const Order* bids = batch.get_bids(book_idx);
    int n_orders = batch.n_orders_per_book;
    
    // Lane 0 processes (sequential scan for now)
    if (laneId == 0) {
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
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    // Get source and destination arrays
    const Order* src_asks = src_batch.get_asks(book_idx);
    const Order* src_bids = src_batch.get_bids(book_idx);
    const Trade* src_trades = src_batch.get_trades(book_idx);
    
    Order* dst_asks = dst_batch.get_asks(book_idx);
    Order* dst_bids = dst_batch.get_bids(book_idx);
    Trade* dst_trades = dst_batch.get_trades(book_idx);
    
    int n_orders = src_batch.n_orders_per_book;
    int n_trades = src_batch.n_trades_per_book;
    
    // Parallelize copy across warp lanes
    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
        dst_asks[i] = src_asks[i];
        dst_bids[i] = src_bids[i];
    }
    
    for (int i = laneId; i < n_trades; i += WARP_SIZE) {
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
    int book_idx = get_book_idx(num_books);
    if (book_idx < 0) return;
    
    int laneId = get_lane_id();
    
    Trade* trades = batch.get_trades(book_idx);
    int n_trades = batch.n_trades_per_book;
    
    // Parallelize across warp lanes
    for (int i = laneId; i < n_trades; i += WARP_SIZE) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
}

} // namespace cuda_orderbook
