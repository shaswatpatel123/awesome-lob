#ifndef CUDA_ORDERBOOK_CPU_H
#define CUDA_ORDERBOOK_CPU_H

#include "types.h"

namespace cuda_orderbook {

/**
 * CPU Sequential Orderbook Implementation
 * 
 * This is a pure CPU implementation for benchmarking purposes.
 * Provides identical functionality to GPU version but runs sequentially on CPU.
 * 
 * Use this as baseline for performance comparisons.
 */

// ============================================================================
// DATA STRUCTURES FOR CPU
// ============================================================================

/**
 * CPU Orderbook - holds orders for one market
 */
struct OrderbookCPU {
    Order* asks;              // CPU pointer to ask orders
    Order* bids;              // CPU pointer to bid orders
    Trade* trades;            // CPU pointer to trades
    int n_orders_per_side;    // Max orders per side
    int n_trades;             // Max trades
    
    OrderbookCPU()
        : asks(nullptr), bids(nullptr), trades(nullptr),
          n_orders_per_side(0), n_trades(0) {}
    
    ~OrderbookCPU() {
        cleanup();
    }
    
    // Allocate memory
    bool allocate(int n_orders, int n_trades_max);
    
    // Free memory
    void cleanup();
    
    // Initialize to empty state
    void initialize();
};

/**
 * Batch of CPU orderbooks for parallel testing
 */
struct OrderbookBatchCPU {
    OrderbookCPU* books;      // Array of orderbooks
    int num_books;            // Number of orderbooks
    
    OrderbookBatchCPU() : books(nullptr), num_books(0) {}
    
    ~OrderbookBatchCPU() {
        cleanup();
    }
    
    // Allocate batch
    bool allocate(int n_books, int n_orders_per_book, int n_trades_per_book);
    
    // Free batch
    void cleanup();
    
    // Initialize all orderbooks
    void initialize();
};

// ============================================================================
// BASIC OPERATIONS (CPU)
// ============================================================================

/**
 * Add order to orderside
 */
void add_order_cpu(Order* orderside, const Message& msg, int n_orders);

/**
 * Cancel order from orderside
 */
void cancel_order_cpu(Order* orderside, const Message& msg, int n_orders);

/**
 * Match against ask orders (for incoming buy order)
 */
void match_against_asks_cpu(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
);

/**
 * Match against bid orders (for incoming sell order)
 */
void match_against_bids_cpu(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
);

/**
 * Process a single message (dispatches to appropriate function)
 */
void process_message_cpu(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
);

// ============================================================================
// BATCH PROCESSING (CPU)
// ============================================================================

/**
 * Process messages sequentially for a single orderbook
 * This is the main entry point for CPU processing
 */
void process_messages_sequential_cpu(
    OrderbookCPU& book,
    const Message* messages,
    int num_messages
);

/**
 * Process messages for batch of orderbooks
 * Each orderbook processed sequentially, one after another
 */
void process_messages_batch_cpu(
    OrderbookBatchCPU& batch,
    const Message* messages,
    int num_messages_per_book
);

// ============================================================================
// HELPER FUNCTIONS (CPU)
// ============================================================================

/**
 * Get index of best ask order (lowest price, earliest time)
 */
int get_top_ask_order_idx_cpu(const Order* asks, int n_orders);

/**
 * Get index of best bid order (highest price, earliest time)
 */
int get_top_bid_order_idx_cpu(const Order* bids, int n_orders);

/**
 * Remove orders with zero or negative quantity
 */
void remove_zero_neg_quant_cpu(Order* orderside, int n_orders);

/**
 * Match a single order and generate trade
 */
void match_single_order_cpu(
    int top_order_idx,
    Order* orderside,
    int32_t& qtm_remaining,
    Trade* trades,
    int n_trades,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns,
    int n_orders
);

// ============================================================================
// UTILITY FUNCTIONS (CPU)
// ============================================================================

/**
 * Copy orderbook from CPU to CPU (for testing)
 */
void copy_orderbook_cpu(
    const OrderbookCPU& src,
    OrderbookCPU& dst
);

/**
 * Compare two orderbooks for equality (for testing)
 */
bool compare_orderbooks_cpu(
    const OrderbookCPU& book1,
    const OrderbookCPU& book2
);

/**
 * Print orderbook state (for debugging)
 */
void print_orderbook_cpu(
    const OrderbookCPU& book,
    int max_orders = 10
);

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_CPU_H

