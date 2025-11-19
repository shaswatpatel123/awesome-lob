/**
 * Utility functions for CUDA Orderbook
 * Memory management, initialization, and data transfer
 */

#include "utils.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace cuda_orderbook {

// ============================================================================
// MEMORY ALLOCATION
// ============================================================================

bool allocate_orderbook_batch(
    OrderbookBatch& batch,
    int num_books,
    int n_orders_per_book,
    int n_trades_per_book
) {
    batch.num_books = num_books;
    batch.n_orders_per_book = n_orders_per_book;
    batch.n_trades_per_book = n_trades_per_book;

    size_t orders_size = num_books * n_orders_per_book * sizeof(Order);
    size_t trades_size = num_books * n_trades_per_book * sizeof(Trade);

    if (cudaMalloc(&batch.d_asks, orders_size) != cudaSuccess) return false;
    if (cudaMalloc(&batch.d_bids, orders_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        return false;
    }
    if (cudaMalloc(&batch.d_trades, trades_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        return false;
    }

    return true;
}

void free_orderbook_batch(OrderbookBatch& batch) {
    if (batch.d_asks) cudaFree(batch.d_asks);
    if (batch.d_bids) cudaFree(batch.d_bids);
    if (batch.d_trades) cudaFree(batch.d_trades);
    
    batch.d_asks = nullptr;
    batch.d_bids = nullptr;
    batch.d_trades = nullptr;
}

bool allocate_host_orderbook_batch(
    OrderbookBatch& batch,
    int num_books,
    int n_orders_per_book,
    int n_trades_per_book
) {
    batch.num_books = num_books;
    batch.n_orders_per_book = n_orders_per_book;
    batch.n_trades_per_book = n_trades_per_book;


    batch.h_asks = new Order[num_books * n_orders_per_book];
    batch.h_bids = new Order[num_books * n_orders_per_book];
    batch.h_trades = new Trade[num_books * n_trades_per_book];

    return (batch.h_asks && batch.h_bids && batch.h_trades);
}

void free_host_orderbook_batch(OrderbookBatch& batch) {
    if (batch.h_asks) delete[] batch.h_asks;
    if (batch.h_bids) delete[] batch.h_bids;
    if (batch.h_trades) delete[] batch.h_trades;
    
    batch.h_asks = nullptr;
    batch.h_bids = nullptr;
    batch.h_trades = nullptr;
}

// ============================================================================
// DATA TRANSFER
// ============================================================================

void copy_to_device(
    const OrderbookBatch& batch,
    bool copy_asks,
    bool copy_bids,
    bool copy_trades
) {
    size_t orders_size = batch.num_books * batch.n_orders_per_book * sizeof(Order);
    size_t trades_size = batch.num_books * batch.n_trades_per_book * sizeof(Trade);

    if (copy_asks && batch.h_asks && batch.d_asks) {
        cudaMemcpy(batch.d_asks, batch.h_asks, orders_size, cudaMemcpyHostToDevice);
    }
    if (copy_bids && batch.h_bids && batch.d_bids) {
        cudaMemcpy(batch.d_bids, batch.h_bids, orders_size, cudaMemcpyHostToDevice);
    }
    if (copy_trades && batch.h_trades && batch.d_trades) {
        cudaMemcpy(batch.d_trades, batch.h_trades, trades_size, cudaMemcpyHostToDevice);
    }
}

void copy_to_host(
    const OrderbookBatch& batch,
    bool copy_asks,
    bool copy_bids,
    bool copy_trades
) {
    size_t orders_size = batch.num_books * batch.n_orders_per_book * sizeof(Order);
    size_t trades_size = batch.num_books * batch.n_trades_per_book * sizeof(Trade);

    if (copy_asks && batch.h_asks && batch.d_asks) {
        cudaMemcpy(batch.h_asks, batch.d_asks, orders_size, cudaMemcpyDeviceToHost);
    }
    if (copy_bids && batch.h_bids && batch.d_bids) {
        cudaMemcpy(batch.h_bids, batch.d_bids, orders_size, cudaMemcpyDeviceToHost);
    }
    if (copy_trades && batch.h_trades && batch.d_trades) {
        cudaMemcpy(batch.h_trades, batch.d_trades, trades_size, cudaMemcpyDeviceToHost);
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void init_orderbooks_host(OrderbookBatch& batch) {
    int total_orders = batch.num_books * batch.n_orders_per_book;
    int total_trades = batch.num_books * batch.n_trades_per_book;

    for (int i = 0; i < total_orders; i++) {
        batch.h_asks[i] = Order();  // Empty order
        batch.h_bids[i] = Order();
    }

    for (int i = 0; i < total_trades; i++) {
        batch.h_trades[i] = Trade();  // Empty trade
    }
}

void init_orderbooks_device(const OrderbookBatch& batch) {
    dim3 grid_dim, block_dim;
    calculate_launch_config(batch.num_books, grid_dim, block_dim);
    init_orderbooks_kernel<<<grid_dim, block_dim>>>(batch, batch.num_books);
    cudaDeviceSynchronize();
}

void calculate_launch_config(
    int num_books,
    dim3& grid_dim,
    dim3& block_dim
) {
    // Warp-level parallelism: 1 LOB per warp (32 threads)
    // Use 4 warps per block = 128 threads per block
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
    
    int books_per_block = WARPS_PER_BLOCK;
    int num_blocks = (num_books + books_per_block - 1) / books_per_block;
    
    grid_dim = dim3(num_blocks, 1, 1);
    block_dim = dim3(THREADS_PER_BLOCK, 1, 1);
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    }
}

void print_orderbook(
    const OrderbookBatch& batch,
    int book_idx,
    int max_orders
) {
    if (book_idx < 0 || book_idx >= batch.num_books) {
        printf("Invalid book index: %d\n", book_idx);
        return;
    }
    
    printf("=== Orderbook %d ===\n", book_idx);
    printf("Asks:\n");
    
    Order* book_asks = batch.h_asks + (book_idx * batch.n_orders_per_book);
    Order* book_bids = batch.h_bids + (book_idx * batch.n_orders_per_book);
    
    int count = 0;
    for (int i = 0; i < batch.n_orders_per_book && count < max_orders; i++) {
        if (book_asks[i].price != EMPTY_PRICE) {
            printf("  [%d] Price: %d, Qty: %d, ID: %d\n",
                   i, book_asks[i].price, book_asks[i].quantity, book_asks[i].order_id);
            count++;
        }
    }
    
    printf("Bids:\n");
    count = 0;
    for (int i = 0; i < batch.n_orders_per_book && count < max_orders; i++) {
        if (book_bids[i].price != EMPTY_PRICE) {
            printf("  [%d] Price: %d, Qty: %d, ID: %d\n",
                   i, book_bids[i].price, book_bids[i].quantity, book_bids[i].order_id);
            count++;
        }
    }
}

bool validate_orderbook(
    const OrderbookBatch& batch,
    int book_idx
) {
    if (book_idx < 0 || book_idx >= batch.num_books) {
        return false;
    }
    
    Order* book_asks = batch.h_asks + (book_idx * batch.n_orders_per_book);
    Order* book_bids = batch.h_bids + (book_idx * batch.n_orders_per_book);
    
    // Check asks
    for (int i = 0; i < batch.n_orders_per_book; i++) {
        if (book_asks[i].price != EMPTY_PRICE) {
            if (book_asks[i].quantity <= 0) {
                printf("Invalid ask at index %d: negative/zero quantity\n", i);
                return false;
            }
            if (book_asks[i].price < 0) {
                printf("Invalid ask at index %d: negative price\n", i);
                return false;
            }
        }
    }
    
    // Check bids
    for (int i = 0; i < batch.n_orders_per_book; i++) {
        if (book_bids[i].price != EMPTY_PRICE) {
            if (book_bids[i].quantity <= 0) {
                printf("Invalid bid at index %d: negative/zero quantity\n", i);
                return false;
            }
            if (book_bids[i].price < 0) {
                printf("Invalid bid at index %d: negative price\n", i);
                return false;
            }
        }
    }
    
    return true;
}

// ============================================================================
// DEVICE UTILITY FUNCTIONS
// ============================================================================

__device__ int find_empty_slot(const Order* orders, int n_orders) {
    for (int i = 0; i < n_orders; i++) {
        if (orders[i].price == EMPTY_PRICE) {
            return i;
        }
    }
    return -1;
}

__device__ int find_order_by_id(
    const Order* orders,
    int n_orders,
    int32_t order_id
) {
    for (int i = 0; i < n_orders; i++) {
        if (orders[i].order_id == order_id) {
            return i;
        }
    }
    return -1;
}

__device__ int find_order_by_price(
    const Order* orders,
    int n_orders,
    int32_t price
) {
    for (int i = 0; i < n_orders; i++) {
        if (orders[i].price == price) {
            return i;
        }
    }
    return -1;
}

__device__ bool has_time_priority(const Order& order1, const Order& order2) {
    if (order1.time_sec < order2.time_sec) return true;
    if (order1.time_sec > order2.time_sec) return false;
    return order1.time_ns < order2.time_ns;
}

} // namespace cuda_orderbook


