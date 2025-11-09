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

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_UTILS_H

