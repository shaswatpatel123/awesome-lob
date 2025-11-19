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
 * CPU Orderbook - holds orders for one market (price-aware version)
 */
struct OrderbookCPU {
    Order* asks;              // CPU pointer to ask orders
    Order* bids;              // CPU pointer to bid orders
    Trade* trades;            // CPU pointer to trades
    int n_orders_per_side;    // Max orders per side
    int n_trades;             // Max trades
    
    // Price-aware structures for asks
    OrderMetadata* ask_metadata;      // Metadata for ask orders
    PriceBucket* ask_buckets;         // Price buckets for asks
    PriceMapEntry* ask_price_map;     // Hash map: price -> bucket index for asks
    OrderIDMapEntry* ask_order_id_map; // Hash map: order_id -> order index for asks
    BestPriceTracker ask_tracker;     // Best ask price tracker
    
    // Price-aware structures for bids
    OrderMetadata* bid_metadata;      // Metadata for bid orders
    PriceBucket* bid_buckets;         // Price buckets for bids
    PriceMapEntry* bid_price_map;     // Hash map: price -> bucket index for bids
    OrderIDMapEntry* bid_order_id_map; // Hash map: order_id -> order index for bids
    BestPriceTracker bid_tracker;     // Best bid price tracker
    
    // Capacity constants
    int n_price_buckets;          // Maximum number of active price levels
    int price_map_size;           // Size of price hash map
    int order_id_map_size;        // Size of order-ID hash map
    
    OrderbookCPU()
        : asks(nullptr), bids(nullptr), trades(nullptr),
          n_orders_per_side(0), n_trades(0),
          ask_metadata(nullptr), ask_buckets(nullptr),
          ask_price_map(nullptr), ask_order_id_map(nullptr),
          bid_metadata(nullptr), bid_buckets(nullptr),
          bid_price_map(nullptr), bid_order_id_map(nullptr),
          n_price_buckets(1024), price_map_size(PRICE_MAP_SIZE),
          order_id_map_size(ORDER_ID_MAP_SIZE) {}
    
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
 * Add order to orderside (price-aware version)
 */
void add_order_cpu(OrderbookCPU& book, bool is_ask_side, const Message& msg);

/**
 * Cancel order from orderside (price-aware version)
 */
void cancel_order_cpu(OrderbookCPU& book, bool is_ask_side, const Message& msg);

/**
 * Match against ask orders (for incoming buy order) - price-aware version
 */
void match_against_asks_cpu(OrderbookCPU& book, const Message& msg);

/**
 * Match against bid orders (for incoming sell order) - price-aware version
 */
void match_against_bids_cpu(OrderbookCPU& book, const Message& msg);

/**
 * Process a single message (dispatches to appropriate function) - price-aware version
 */
void process_message_cpu(OrderbookCPU& book, const Message& msg);

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
 * Get index of best ask order (price-aware version - O(1))
 */
int get_top_ask_order_idx_cpu(const OrderbookCPU& book);

/**
 * Get index of best bid order (price-aware version - O(1))
 */
int get_top_bid_order_idx_cpu(const OrderbookCPU& book);

/**
 * Remove orders with zero or negative quantity (price-aware version)
 */
void remove_zero_neg_quant_cpu(OrderbookCPU& book, bool is_ask_side);

/**
 * Match a single order and generate trade (price-aware version)
 */
void match_single_order_cpu(
    OrderbookCPU& book,
    bool is_ask_side,
    int top_order_idx,
    int32_t& qtm_remaining,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns
);

// ============================================================================
// PRICE-AWARE HELPER FUNCTIONS (CPU)
// ============================================================================

/**
 * Hash function for price -> hash table index
 */
int32_t hash_price_cpu(int32_t price, int32_t map_size);

/**
 * Hash function for order ID -> hash table index
 */
int32_t hash_order_id_cpu(int32_t order_id, int32_t map_size);

/**
 * Find price bucket index for a given price using hash map
 */
int32_t find_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t map_size);

/**
 * Insert price -> bucket mapping into hash map
 */
bool insert_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t bucket_idx, int32_t map_size);

/**
 * Remove price -> bucket mapping from hash map
 */
void remove_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t map_size);

/**
 * Find order index by order ID using hash map
 */
int32_t find_order_by_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t map_size);

/**
 * Insert order_id -> order_idx mapping into hash map
 */
bool insert_order_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t order_idx, int32_t map_size);

/**
 * Remove order_id -> order_idx mapping from hash map
 */
void remove_order_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t map_size);

/**
 * Find or create a price bucket for a given price
 */
int32_t get_or_create_price_bucket_cpu(PriceBucket* buckets, PriceMapEntry* price_map, int32_t price, int32_t n_buckets, int32_t map_size);

/**
 * Add order to price bucket (at tail, FIFO)
 */
void add_order_to_bucket_cpu(PriceBucket* buckets, OrderMetadata* metadata, Order* orders, int32_t bucket_idx, int32_t order_idx);

/**
 * Remove order from price bucket
 */
void remove_order_from_bucket_cpu(PriceBucket* buckets, OrderMetadata* metadata, Order* orders, int32_t bucket_idx, int32_t order_idx, int32_t removed_quantity);

/**
 * Update best price tracker for asks (find minimum price)
 */
void update_best_ask_price_cpu(PriceBucket* buckets, PriceMapEntry* price_map, BestPriceTracker* tracker, int32_t n_buckets, int32_t map_size);

/**
 * Update best price tracker for bids (find maximum price)
 */
void update_best_bid_price_cpu(PriceBucket* buckets, PriceMapEntry* price_map, BestPriceTracker* tracker, int32_t n_buckets, int32_t map_size);

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

