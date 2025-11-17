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
    init_orderbooks_kernel<<<1, 256>>>(batch, batch.num_books);
    cudaDeviceSynchronize();
}

} // namespace cuda_orderbook


