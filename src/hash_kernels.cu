/**
 * Hash-Accelerated Orderbook Kernels
 * 
 * CUDA kernels for parallel processing of multiple orderbooks
 * Each thread block processes one orderbook
 */

#include "types.h"
#include "simple_hash.cuh"
#include "cuco_wrapper.cuh"

// Forward declarations from hash_operations.cu
namespace cuda_orderbook {

// Device functions
__device__ void add_order_hash_device(
    Order* orderside, void* hash_map, const Message& msg,
    int n_orders, HashImplementation impl, bool* is_sorted
);

__device__ void cancel_order_hash_device(
    Order* orderside, void* hash_map, const Message& msg,
    int n_orders, HashImplementation impl, bool* is_sorted
);

__device__ int32_t get_best_ask_hash_device(
    Order* asks, int n_orders, bool* is_sorted
);

__device__ int32_t get_best_bid_hash_device(
    Order* bids, int n_orders, bool* is_sorted
);

__device__ void process_message_hash_device(
    HashOrderbookState* state,
    const Message& msg
);

__device__ void match_against_asks_hash_device(
    Order* asks, Order* bids, Trade* trades, const Message& msg,
    int n_orders, int n_trades,
    void* ask_hash_map, void* bid_hash_map, HashImplementation impl,
    bool* asks_sorted, bool* bids_sorted
);

__device__ void match_against_bids_hash_device(
    Order* asks, Order* bids, Trade* trades, const Message& msg,
    int n_orders, int n_trades,
    void* ask_hash_map, void* bid_hash_map, HashImplementation impl,
    bool* asks_sorted, bool* bids_sorted
);

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

/**
 * Initialize hash orderbooks
 * Each block initializes one orderbook
 */
__global__ void init_hash_orderbooks_kernel(
    HashOrderbookBatch batch
) {
    int book_idx = blockIdx.x;
    if (book_idx >= batch.num_books) return;
    
    HashOrderbookState* state = batch.get_state(book_idx);
    int tid = threadIdx.x;
    int n_orders = state->n_orders;
    int n_trades = state->n_trades;
    
    // Initialize orders to empty (parallel across threads)
    for (int i = tid; i < n_orders; i += blockDim.x) {
        state->asks[i].price = EMPTY_PRICE;
        state->asks[i].quantity = 0;
        state->asks[i].order_id = 0;
        state->asks[i].trader_id = 0;
        state->asks[i].time_sec = 0;
        state->asks[i].time_ns = 0;
        
        state->bids[i].price = EMPTY_PRICE;
        state->bids[i].quantity = 0;
        state->bids[i].order_id = 0;
        state->bids[i].trader_id = 0;
        state->bids[i].time_sec = 0;
        state->bids[i].time_ns = 0;
    }
    
    // Initialize trades to empty
    for (int i = tid; i < n_trades; i += blockDim.x) {
        state->trades[i].price = EMPTY_PRICE;
        state->trades[i].quantity = 0;
        state->trades[i].passive_order_id = 0;
        state->trades[i].aggressive_order_id = 0;
        state->trades[i].time_sec = 0;
        state->trades[i].time_ns = 0;
    }
    
    // Initialize sort state
    if (tid == 0) {
        state->asks_sorted = false;
        state->bids_sorted = false;
    }
    
    __syncthreads();
}

// ============================================================================
// BASIC OPERATION KERNELS
// ============================================================================

/**
 * Add orders to batch of orderbooks
 * One orderbook per block
 */
__global__ void add_order_hash_kernel(
    HashOrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Only first thread in block processes the message
    if (threadIdx.x == 0) {
        HashOrderbookState* state = batch.get_state(book_idx);
        const Message& msg = messages[book_idx];
        
        if (msg.side == Message::ASK) {
            add_order_hash_device(
                state->asks,
                state->ask_hash_map,
                msg,
                state->n_orders,
                state->hash_impl,
                &state->asks_sorted
            );
        } else if (msg.side == Message::BID) {
            add_order_hash_device(
                state->bids,
                state->bid_hash_map,
                msg,
                state->n_orders,
                state->hash_impl,
                &state->bids_sorted
            );
        }
    }
}

/**
 * Cancel orders from batch of orderbooks
 */
__global__ void cancel_order_hash_kernel(
    HashOrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    if (threadIdx.x == 0) {
        HashOrderbookState* state = batch.get_state(book_idx);
        const Message& msg = messages[book_idx];
        
        if (msg.side == Message::ASK) {
            cancel_order_hash_device(
                state->asks,
                state->ask_hash_map,
                msg,
                state->n_orders,
                state->hash_impl,
                &state->asks_sorted
            );
        } else if (msg.side == Message::BID) {
            cancel_order_hash_device(
                state->bids,
                state->bid_hash_map,
                msg,
                state->n_orders,
                state->hash_impl,
                &state->bids_sorted
            );
        }
    }
}

// ============================================================================
// MATCHING KERNELS
// ============================================================================

/**
 * Match orders in batch
 */
__global__ void match_order_hash_kernel(
    HashOrderbookBatch batch,
    const Message* messages,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    if (threadIdx.x == 0) {
        HashOrderbookState* state = batch.get_state(book_idx);
        const Message& msg = messages[book_idx];
        
        if (msg.side == Message::BID) {
            // Buy order: match against asks
            match_against_asks_hash_device(
                state->asks,
                state->bids,
                state->trades,
                msg,
                state->n_orders,
                state->n_trades,
                state->ask_hash_map,
                state->bid_hash_map,
                state->hash_impl,
                &state->asks_sorted,
                &state->bids_sorted
            );
        } else if (msg.side == Message::ASK) {
            // Sell order: match against bids
            match_against_bids_hash_device(
                state->asks,
                state->bids,
                state->trades,
                msg,
                state->n_orders,
                state->n_trades,
                state->ask_hash_map,
                state->bid_hash_map,
                state->hash_impl,
                &state->asks_sorted,
                &state->bids_sorted
            );
        }
    }
}

/**
 * Process array of messages sequentially for each orderbook
 * THIS IS THE MAIN KERNEL
 */
__global__ void process_messages_hash_kernel(
    HashOrderbookBatch batch,
    const Message* messages,
    int num_messages_per_book,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    // Only first thread processes messages sequentially
    if (threadIdx.x == 0) {
        HashOrderbookState* state = batch.get_state(book_idx);
        
        // Process all messages for this orderbook
        for (int msg_idx = 0; msg_idx < num_messages_per_book; msg_idx++) {
            int global_msg_idx = book_idx * num_messages_per_book + msg_idx;
            const Message& msg = messages[global_msg_idx];
            
            process_message_hash_device(state, msg);
        }
    }
}

// ============================================================================
// QUERY KERNELS
// ============================================================================

/**
 * Get best bid and ask for all orderbooks
 */
__global__ void get_best_bid_ask_hash_kernel(
    const HashOrderbookBatch batch,
    int32_t* best_asks,
    int32_t* best_bids,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    if (threadIdx.x == 0) {
        HashOrderbookState* state = batch.get_state(book_idx);
        
        best_asks[book_idx] = get_best_ask_hash_device(
            state->asks,
            state->n_orders,
            &state->asks_sorted
        );
        
        best_bids[book_idx] = get_best_bid_hash_device(
            state->bids,
            state->n_orders,
            &state->bids_sorted
        );
    }
}

/**
 * Get volume at specific price for all orderbooks
 */
__global__ void get_volume_at_price_hash_kernel(
    const HashOrderbookBatch batch,
    const int32_t* prices,
    const int32_t* sides,
    int32_t* volumes,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    int tid = threadIdx.x;
    HashOrderbookState* state = batch.get_state(book_idx);
    int32_t target_price = prices[book_idx];
    int32_t side = sides[book_idx];
    
    // Choose side
    Order* orderside = (side == 0) ? state->asks : state->bids;
    
    // Parallel reduction to sum volumes at price
    __shared__ int32_t partial_sums[256];
    int32_t thread_sum = 0;
    
    for (int i = tid; i < state->n_orders; i += blockDim.x) {
        if (orderside[i].price == target_price) {
            thread_sum += orderside[i].quantity;
        }
    }
    
    partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        volumes[book_idx] = partial_sums[0];
    }
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * Reset trades array
 */
__global__ void reset_trades_hash_kernel(
    HashOrderbookBatch batch,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    HashOrderbookState* state = batch.get_state(book_idx);
    int tid = threadIdx.x;
    
    for (int i = tid; i < state->n_trades; i += blockDim.x) {
        state->trades[i].price = EMPTY_PRICE;
        state->trades[i].quantity = 0;
        state->trades[i].passive_order_id = 0;
        state->trades[i].aggressive_order_id = 0;
        state->trades[i].time_sec = 0;
        state->trades[i].time_ns = 0;
    }
}

/**
 * Copy orderbook state
 */
__global__ void copy_hash_orderbooks_kernel(
    const HashOrderbookBatch src_batch,
    HashOrderbookBatch dst_batch,
    int num_books
) {
    int book_idx = blockIdx.x;
    if (book_idx >= num_books) return;
    
    HashOrderbookState* src_state = src_batch.get_state(book_idx);
    HashOrderbookState* dst_state = dst_batch.get_state(book_idx);
    int tid = threadIdx.x;
    
    // Copy orders
    for (int i = tid; i < src_state->n_orders; i += blockDim.x) {
        dst_state->asks[i] = src_state->asks[i];
        dst_state->bids[i] = src_state->bids[i];
    }
    
    // Copy trades
    for (int i = tid; i < src_state->n_trades; i += blockDim.x) {
        dst_state->trades[i] = src_state->trades[i];
    }
    
    // Copy metadata
    if (tid == 0) {
        dst_state->asks_sorted = src_state->asks_sorted;
        dst_state->bids_sorted = src_state->bids_sorted;
    }
    
    // Note: Hash maps are not deep copied (would need special handling)
}

} // namespace cuda_orderbook

