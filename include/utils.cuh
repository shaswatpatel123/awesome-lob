#ifndef CUDA_ORDERBOOK_UTILS_H
#define CUDA_ORDERBOOK_UTILS_H

#include "types.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace cuda_orderbook {

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

/**
 * Check CUDA error and print message if error occurs
 * Usage: CHECK_CUDA_ERROR(cudaMalloc(...));
 */
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Check for CUDA kernel launch errors
 * Usage: After kernel launch, call CHECK_KERNEL_ERROR();
 */
#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Sync Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// MEMORY MANAGEMENT UTILITIES
// ============================================================================

/**
 * Allocate device memory for orderbook batch
 * 
 * @param batch Batch structure to populate with device pointers
 * @param num_books Number of orderbooks
 * @param n_orders_per_book Orders per side per book
 * @param n_trades_per_book Trades per book
 * @return true if successful, false otherwise
 */
bool allocate_orderbook_batch(
    OrderbookBatch& batch,
    int num_books,
    int n_orders_per_book,
    int n_trades_per_book
);

/**
 * Free device memory for orderbook batch
 * 
 * @param batch Batch structure with device pointers to free
 */
void free_orderbook_batch(OrderbookBatch& batch);

/**
 * Allocate host (pinned) memory for orderbook batch
 * Enables faster host-device transfers
 * 
 * @param batch Batch structure to populate with host pointers
 * @param num_books Number of orderbooks
 * @param n_orders_per_book Orders per side per book
 * @param n_trades_per_book Trades per book
 * @return true if successful, false otherwise
 */
bool allocate_host_orderbook_batch(
    OrderbookBatch& batch,
    int num_books,
    int n_orders_per_book,
    int n_trades_per_book
);

/**
 * Free host (pinned) memory for orderbook batch
 * 
 * @param batch Batch structure with host pointers to free
 */
void free_host_orderbook_batch(OrderbookBatch& batch);

// ============================================================================
// DATA TRANSFER UTILITIES
// ============================================================================

/**
 * Copy orderbook data from host to device
 * 
 * @param batch Batch with both host and device pointers
 * @param copy_asks Copy ask orders (default: true)
 * @param copy_bids Copy bid orders (default: true)
 * @param copy_trades Copy trades (default: true)
 */
void copy_to_device(
    const OrderbookBatch& batch,
    bool copy_asks = true,
    bool copy_bids = true,
    bool copy_trades = true
);

/**
 * Copy orderbook data from device to host
 * 
 * @param batch Batch with both host and device pointers
 * @param copy_asks Copy ask orders (default: true)
 * @param copy_bids Copy bid orders (default: true)
 * @param copy_trades Copy trades (default: true)
 */
void copy_to_host(
    const OrderbookBatch& batch,
    bool copy_asks = true,
    bool copy_bids = true,
    bool copy_trades = true
);

/**
 * Copy specific orderbook from host to device
 * 
 * @param batch Batch structure
 * @param book_idx Index of orderbook to copy
 */
void copy_single_book_to_device(
    const OrderbookBatch& batch,
    int book_idx
);

/**
 * Copy specific orderbook from device to host
 * 
 * @param batch Batch structure
 * @param book_idx Index of orderbook to copy
 */
void copy_single_book_to_host(
    const OrderbookBatch& batch,
    int book_idx
);

// ============================================================================
// INITIALIZATION UTILITIES
// ============================================================================

/**
 * Initialize orderbook batch to empty state on host
 * All prices set to EMPTY_PRICE (-1)
 * 
 * @param batch Batch structure with host pointers
 */
void init_orderbooks_host(OrderbookBatch& batch);

/**
 * Initialize orderbook batch to empty state on device
 * Launches init_orderbooks_kernel
 * 
 * @param batch Batch structure with device pointers
 */
void init_orderbooks_device(const OrderbookBatch& batch);

/**
 * Initialize single orderbook from L2 snapshot
 * Converts L2 price-quantity pairs to limit orders
 * 
 * @param batch Batch structure
 * @param book_idx Index of orderbook to initialize
 * @param l2_data L2 snapshot data [ask_p1, ask_q1, bid_p1, bid_q1, ...]
 * @param n_levels Number of price levels
 */
void init_from_l2_snapshot(
    OrderbookBatch& batch,
    int book_idx,
    const int32_t* l2_data,
    int n_levels
);

// ============================================================================
// DEBUGGING AND VALIDATION UTILITIES
// ============================================================================

/**
 * Print orderbook state (for debugging)
 * 
 * @param batch Batch structure
 * @param book_idx Index of orderbook to print
 * @param max_orders Maximum orders to print per side (default: 10)
 */
void print_orderbook(
    const OrderbookBatch& batch,
    int book_idx,
    int max_orders = 10
);

/**
 * Print L2 state (for debugging)
 * 
 * @param l2_data L2 snapshot data
 * @param n_levels Number of levels
 */
void print_l2_state(
    const int32_t* l2_data,
    int n_levels
);

/**
 * Validate orderbook integrity (for testing)
 * Checks for invalid orders, negative quantities, etc.
 * 
 * @param batch Batch structure (host pointers)
 * @param book_idx Index of orderbook to validate
 * @return true if valid, false otherwise
 */
bool validate_orderbook(
    const OrderbookBatch& batch,
    int book_idx
);

/**
 * Get GPU device properties and print info
 */
void print_device_info();

/**
 * Calculate optimal grid and block dimensions for batch processing
 * 
 * @param num_books Number of orderbooks
 * @param grid_dim Output grid dimensions
 * @param block_dim Output block dimensions
 */
void calculate_launch_config(
    int num_books,
    dim3& grid_dim,
    dim3& block_dim
);

// ============================================================================
// DEVICE UTILITY FUNCTIONS (callable from kernels)
// ============================================================================

/**
 * Find first empty slot in order array
 * 
 * @param orders Order array
 * @param n_orders Size of array
 * @return Index of first empty slot, or -1 if full
 */
__device__ int find_empty_slot(const Order* orders, int n_orders);

/**
 * Find order by ID
 * 
 * @param orders Order array
 * @param n_orders Size of array
 * @param order_id Order ID to find
 * @return Index of order, or -1 if not found
 */
__device__ int find_order_by_id(
    const Order* orders,
    int n_orders,
    int32_t order_id
);

/**
 * Find order by price (for INITID orders)
 * 
 * @param orders Order array
 * @param n_orders Size of array
 * @param price Price to find
 * @return Index of order, or -1 if not found
 */
__device__ int find_order_by_price(
    const Order* orders,
    int n_orders,
    int32_t price
);

/**
 * Compare two orders for time priority
 * Returns true if order1 has priority over order2
 * 
 * @param order1 First order
 * @param order2 Second order
 * @return true if order1 comes before order2 in time
 */
__device__ bool has_time_priority(const Order& order1, const Order& order2);

/**
 * Atomic min operation for int32_t
 * CUDA doesn't provide atomicMin for all types
 */
__device__ inline void atomic_min_int32(int32_t* address, int32_t val) {
    atomicMin((int*)address, (int)val);
}

/**
 * Atomic max operation for int32_t
 */
__device__ inline void atomic_max_int32(int32_t* address, int32_t val) {
    atomicMax((int*)address, (int)val);
}

// ============================================================================
// PRICE-AWARE DEVICE UTILITY FUNCTIONS
// ============================================================================

/**
 * Hash function for price -> hash table index
 * @param price Price to hash
 * @param map_size Size of hash map
 * @return Hash table index
 */
__device__ inline int32_t hash_price(int32_t price, int32_t map_size) {
    // Simple hash function for prices
    uint32_t hash = (uint32_t)(price) * 2654435761U; // Knuth's multiplicative hash
    return (int32_t)(hash % (uint32_t)map_size);
}

/**
 * Hash function for order ID -> hash table index
 * @param order_id Order ID to hash
 * @param map_size Size of hash map
 * @return Hash table index
 */
__device__ inline int32_t hash_order_id(int32_t order_id, int32_t map_size) {
    uint32_t hash = (uint32_t)(order_id) * 2654435761U;
    return (int32_t)(hash % (uint32_t)map_size);
}

/**
 * Find price bucket index for a given price using hash map
 * Uses linear probing for collisions
 * @param price_map Price map array
 * @param price Price to find
 * @param map_size Size of price map
 * @return Bucket index, or EMPTY_INDEX if not found
 */
__device__ int32_t find_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t map_size
);

/**
 * Insert price -> bucket mapping into hash map
 * Uses linear probing for collisions
 * @param price_map Price map array
 * @param price Price key
 * @param bucket_idx Bucket index to store
 * @param map_size Size of price map
 * @return true if inserted successfully, false if map is full
 */
__device__ bool insert_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t bucket_idx,
    int32_t map_size
);

/**
 * Remove price -> bucket mapping from hash map
 * @param price_map Price map array
 * @param price Price key to remove
 * @param map_size Size of price map
 */
__device__ void remove_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t map_size
);

/**
 * Find order index by order ID using hash map
 * @param order_id_map Order-ID map array
 * @param order_id Order ID to find
 * @param map_size Size of order-ID map
 * @return Order index, or EMPTY_INDEX if not found
 */
__device__ int32_t find_order_by_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t map_size
);

/**
 * Insert order_id -> order_idx mapping into hash map
 * @param order_id_map Order-ID map array
 * @param order_id Order ID key
 * @param order_idx Order index to store
 * @param map_size Size of order-ID map
 * @return true if inserted successfully
 */
__device__ bool insert_order_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t order_idx,
    int32_t map_size
);

/**
 * Remove order_id -> order_idx mapping from hash map
 * @param order_id_map Order-ID map array
 * @param order_id Order ID to remove
 * @param map_size Size of order-ID map
 */
__device__ void remove_order_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t map_size
);

/**
 * Find or create a price bucket for a given price
 * @param buckets Bucket array
 * @param price_map Price map
 * @param price Price level
 * @param n_buckets Maximum number of buckets
 * @param map_size Size of price map
 * @return Bucket index, or EMPTY_INDEX if cannot create
 */
__device__ int32_t get_or_create_price_bucket(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    int32_t price,
    int32_t n_buckets,
    int32_t map_size
);

/**
 * Add order to price bucket (at tail, FIFO)
 * @param buckets Bucket array
 * @param metadata Order metadata array
 * @param orders Order array
 * @param bucket_idx Bucket index
 * @param order_idx Order index to add
 */
__device__ void add_order_to_bucket(
    PriceBucket* buckets,
    OrderMetadata* metadata,
    Order* orders,
    int32_t bucket_idx,
    int32_t order_idx
);

/**
 * Remove order from price bucket
 * @param buckets Bucket array
 * @param metadata Order metadata array
 * @param orders Order array
 * @param bucket_idx Bucket index
 * @param order_idx Order index to remove
 */
__device__ void remove_order_from_bucket(
    PriceBucket* buckets,
    OrderMetadata* metadata,
    Order* orders,
    int32_t bucket_idx,
    int32_t order_idx,
    int32_t removed_quantity
);

/**
 * Update best price tracker for asks (find minimum price)
 * @param buckets Bucket array
 * @param price_map Price map
 * @param tracker Best price tracker to update
 * @param n_buckets Number of buckets
 * @param map_size Size of price map
 */
__device__ void update_best_ask_price(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    BestPriceTracker* tracker,
    int32_t n_buckets,
    int32_t map_size
);

/**
 * Update best price tracker for bids (find maximum price)
 * @param buckets Bucket array
 * @param price_map Price map
 * @param tracker Best price tracker to update
 * @param n_buckets Number of buckets
 * @param map_size Size of price map
 */
__device__ void update_best_bid_price(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    BestPriceTracker* tracker,
    int32_t n_buckets,
    int32_t map_size
);

/**
 * Get best ask order index from tracker
 * @param state Orderbook state
 * @return Order index of best ask, or EMPTY_INDEX if none
 */
__device__ int32_t get_top_ask_order_idx_price_aware(const OrderbookState& state);

/**
 * Get best bid order index from tracker
 * @param state Orderbook state
 * @return Order index of best bid, or EMPTY_INDEX if none
 */
__device__ int32_t get_top_bid_order_idx_price_aware(const OrderbookState& state);

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_UTILS_H

