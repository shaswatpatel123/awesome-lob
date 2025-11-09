#ifndef CUDA_ORDERBOOK_KERNELS_H
#define CUDA_ORDERBOOK_KERNELS_H

#include "types.h"

namespace cuda_orderbook {

// ============================================================================
// BASIC OPERATIONS KERNELS (Team 1: Days 4-5)
// ============================================================================

/**
 * Add orders to orderbooks in batch
 * Each thread block processes one orderbook
 * Maps to JAX add_order (JaxOrderBookArrays.py:32-37)
 * 
 * @param batch Batch of orderbooks
 * @param messages Messages to process (one per orderbook)
 * @param num_books Number of orderbooks
 */
__global__ void add_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
);

/**
 * Cancel orders from orderbooks in batch
 * Each thread block processes one orderbook
 * Maps to JAX cancel_order (JaxOrderBookArrays.py:52-65)
 * 
 * @param batch Batch of orderbooks
 * @param messages Cancel messages (one per orderbook)
 * @param num_books Number of orderbooks
 */
__global__ void cancel_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
);

// ============================================================================
// MATCHING ENGINE KERNELS (Team 2: Days 6-8)
// ============================================================================

/**
 * Match orders in batch (limit and market orders)
 * Each thread block processes one orderbook
 * Maps to JAX _match_against_bid_orders and _match_against_ask_orders
 * (JaxOrderBookArrays.py:115-130)
 * 
 * @param batch Batch of orderbooks
 * @param messages Messages to match (one per orderbook)
 * @param num_books Number of orderbooks
 */
__global__ void match_order_batch_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_books
);

/**
 * Process array of messages sequentially for each orderbook in parallel
 * THIS IS THE MAIN KERNEL - Entry point for message processing
 * Each thread block processes ALL messages for ONE orderbook sequentially
 * Maps to JAX scan_through_entire_array (JaxOrderBookArrays.py:265-267)
 * 
 * @param batch Batch of orderbooks
 * @param messages Array of messages [book0_msgs, book1_msgs, ...]
 * @param num_messages_per_book Number of messages per orderbook
 * @param num_books Number of orderbooks
 */
__global__ void process_messages_sequential_kernel(
    OrderbookBatch batch,
    const Message* messages,
    int num_messages_per_book,
    int num_books
);

// ============================================================================
// QUERY KERNELS (Team 3: Days 4-6)
// ============================================================================

/**
 * Get best bid and ask for all orderbooks in batch
 * Parallel reduction within each thread block
 * Maps to JAX get_best_ask (line 430) and get_best_bid (line 436)
 * 
 * @param batch Batch of orderbooks
 * @param best_asks Output array of best ask prices (length: num_books)
 * @param best_bids Output array of best bid prices (length: num_books)
 * @param num_books Number of orderbooks
 */
__global__ void get_best_bid_ask_kernel(
    const OrderbookBatch batch,
    int32_t* best_asks,
    int32_t* best_bids,
    int num_books
);

/**
 * Get volume at specific price level for all orderbooks
 * 
 * @param batch Batch of orderbooks
 * @param prices Price levels to query (one per orderbook)
 * @param sides Side to query: 0=ask, 1=bid (one per orderbook)
 * @param volumes Output array of volumes (length: num_books)
 * @param num_books Number of orderbooks
 */
__global__ void get_volume_at_price_kernel(
    const OrderbookBatch batch,
    const int32_t* prices,
    const int32_t* sides,
    int32_t* volumes,
    int num_books
);

/**
 * Extract L2 orderbook state (top N price levels with volumes)
 * Maps to JAX get_L2_state (JaxOrderBookArrays.py:525-545)
 * 
 * @param batch Batch of orderbooks
 * @param l2_states Output array of L2 states [book0_l2, book1_l2, ...]
 *                  Format per book: [ask_p1, ask_q1, bid_p1, bid_q1, ..., ask_pN, ask_qN, bid_pN, bid_qN]
 * @param n_levels Number of price levels to extract per side
 * @param num_books Number of orderbooks
 */
__global__ void get_L2_state_kernel(
    const OrderbookBatch batch,
    int32_t* l2_states,
    int n_levels,
    int num_books
);

/**
 * Get best bid and ask with quantities for all orderbooks
 * Extended version that includes volume at best price
 * Maps to JAX get_best_bid_and_ask_inclQuants (JaxOrderBookArrays.py:357-363)
 * 
 * @param batch Batch of orderbooks
 * @param best_asks_with_qty Output [price, qty] pairs for asks (length: num_books * 2)
 * @param best_bids_with_qty Output [price, qty] pairs for bids (length: num_books * 2)
 * @param num_books Number of orderbooks
 */
__global__ void get_best_bid_ask_with_qty_kernel(
    const OrderbookBatch batch,
    int32_t* best_asks_with_qty,
    int32_t* best_bids_with_qty,
    int num_books
);

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * Initialize orderbooks to empty state
 * All prices set to EMPTY_PRICE (-1)
 * 
 * @param batch Batch of orderbooks to initialize
 * @param num_books Number of orderbooks
 */
__global__ void init_orderbooks_kernel(
    OrderbookBatch batch,
    int num_books
);

/**
 * Copy orderbook state from device to device (useful for checkpointing)
 * 
 * @param src_batch Source orderbook batch
 * @param dst_batch Destination orderbook batch
 * @param num_books Number of orderbooks
 */
__global__ void copy_orderbooks_kernel(
    const OrderbookBatch src_batch,
    OrderbookBatch dst_batch,
    int num_books
);

/**
 * Reset trades array to empty
 * 
 * @param batch Batch of orderbooks
 * @param num_books Number of orderbooks
 */
__global__ void reset_trades_kernel(
    OrderbookBatch batch,
    int num_books
);

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_KERNELS_H

