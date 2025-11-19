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
    size_t metadata_size = num_books * n_orders_per_book * sizeof(OrderMetadata);
    size_t buckets_size = num_books * batch.n_price_buckets_per_book * sizeof(PriceBucket);
    size_t price_map_size = num_books * batch.price_map_size * sizeof(PriceMapEntry);
    size_t order_id_map_size = num_books * batch.order_id_map_size * sizeof(OrderIDMapEntry);
    size_t trackers_size = num_books * sizeof(BestPriceTracker);

    // Allocate orders and trades
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

    // Allocate ask metadata and structures
    if (cudaMalloc(&batch.d_ask_metadata, metadata_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        return false;
    }
    if (cudaMalloc(&batch.d_ask_buckets, buckets_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        return false;
    }
    if (cudaMalloc(&batch.d_ask_price_map, price_map_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        return false;
    }
    if (cudaMalloc(&batch.d_ask_order_id_map, order_id_map_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        return false;
    }
    if (cudaMalloc(&batch.d_ask_trackers, trackers_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        return false;
    }

    // Allocate bid metadata and structures
    if (cudaMalloc(&batch.d_bid_metadata, metadata_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        cudaFree(batch.d_ask_trackers);
        return false;
    }
    if (cudaMalloc(&batch.d_bid_buckets, buckets_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        cudaFree(batch.d_ask_trackers);
        cudaFree(batch.d_bid_metadata);
        return false;
    }
    if (cudaMalloc(&batch.d_bid_price_map, price_map_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        cudaFree(batch.d_ask_trackers);
        cudaFree(batch.d_bid_metadata);
        cudaFree(batch.d_bid_buckets);
        return false;
    }
    if (cudaMalloc(&batch.d_bid_order_id_map, order_id_map_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        cudaFree(batch.d_ask_trackers);
        cudaFree(batch.d_bid_metadata);
        cudaFree(batch.d_bid_buckets);
        cudaFree(batch.d_bid_price_map);
        return false;
    }
    if (cudaMalloc(&batch.d_bid_trackers, trackers_size) != cudaSuccess) {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.d_ask_metadata);
        cudaFree(batch.d_ask_buckets);
        cudaFree(batch.d_ask_price_map);
        cudaFree(batch.d_ask_order_id_map);
        cudaFree(batch.d_ask_trackers);
        cudaFree(batch.d_bid_metadata);
        cudaFree(batch.d_bid_buckets);
        cudaFree(batch.d_bid_price_map);
        cudaFree(batch.d_bid_order_id_map);
        return false;
    }

    return true;
}

void free_orderbook_batch(OrderbookBatch& batch) {
    if (batch.d_asks) cudaFree(batch.d_asks);
    if (batch.d_bids) cudaFree(batch.d_bids);
    if (batch.d_trades) cudaFree(batch.d_trades);
    
    if (batch.d_ask_metadata) cudaFree(batch.d_ask_metadata);
    if (batch.d_ask_buckets) cudaFree(batch.d_ask_buckets);
    if (batch.d_ask_price_map) cudaFree(batch.d_ask_price_map);
    if (batch.d_ask_order_id_map) cudaFree(batch.d_ask_order_id_map);
    if (batch.d_ask_trackers) cudaFree(batch.d_ask_trackers);
    
    if (batch.d_bid_metadata) cudaFree(batch.d_bid_metadata);
    if (batch.d_bid_buckets) cudaFree(batch.d_bid_buckets);
    if (batch.d_bid_price_map) cudaFree(batch.d_bid_price_map);
    if (batch.d_bid_order_id_map) cudaFree(batch.d_bid_order_id_map);
    if (batch.d_bid_trackers) cudaFree(batch.d_bid_trackers);
    
    batch.d_asks = nullptr;
    batch.d_bids = nullptr;
    batch.d_trades = nullptr;
    batch.d_ask_metadata = nullptr;
    batch.d_ask_buckets = nullptr;
    batch.d_ask_price_map = nullptr;
    batch.d_ask_order_id_map = nullptr;
    batch.d_ask_trackers = nullptr;
    batch.d_bid_metadata = nullptr;
    batch.d_bid_buckets = nullptr;
    batch.d_bid_price_map = nullptr;
    batch.d_bid_order_id_map = nullptr;
    batch.d_bid_trackers = nullptr;
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

    batch.h_ask_metadata = new OrderMetadata[num_books * n_orders_per_book];
    batch.h_ask_buckets = new PriceBucket[num_books * batch.n_price_buckets_per_book];
    batch.h_ask_price_map = new PriceMapEntry[num_books * batch.price_map_size];
    batch.h_ask_order_id_map = new OrderIDMapEntry[num_books * batch.order_id_map_size];
    batch.h_ask_trackers = new BestPriceTracker[num_books];

    batch.h_bid_metadata = new OrderMetadata[num_books * n_orders_per_book];
    batch.h_bid_buckets = new PriceBucket[num_books * batch.n_price_buckets_per_book];
    batch.h_bid_price_map = new PriceMapEntry[num_books * batch.price_map_size];
    batch.h_bid_order_id_map = new OrderIDMapEntry[num_books * batch.order_id_map_size];
    batch.h_bid_trackers = new BestPriceTracker[num_books];

    return (batch.h_asks && batch.h_bids && batch.h_trades &&
            batch.h_ask_metadata && batch.h_ask_buckets && batch.h_ask_price_map &&
            batch.h_ask_order_id_map && batch.h_ask_trackers &&
            batch.h_bid_metadata && batch.h_bid_buckets && batch.h_bid_price_map &&
            batch.h_bid_order_id_map && batch.h_bid_trackers);
}

void free_host_orderbook_batch(OrderbookBatch& batch) {
    if (batch.h_asks) delete[] batch.h_asks;
    if (batch.h_bids) delete[] batch.h_bids;
    if (batch.h_trades) delete[] batch.h_trades;
    
    if (batch.h_ask_metadata) delete[] batch.h_ask_metadata;
    if (batch.h_ask_buckets) delete[] batch.h_ask_buckets;
    if (batch.h_ask_price_map) delete[] batch.h_ask_price_map;
    if (batch.h_ask_order_id_map) delete[] batch.h_ask_order_id_map;
    if (batch.h_ask_trackers) delete[] batch.h_ask_trackers;
    
    if (batch.h_bid_metadata) delete[] batch.h_bid_metadata;
    if (batch.h_bid_buckets) delete[] batch.h_bid_buckets;
    if (batch.h_bid_price_map) delete[] batch.h_bid_price_map;
    if (batch.h_bid_order_id_map) delete[] batch.h_bid_order_id_map;
    if (batch.h_bid_trackers) delete[] batch.h_bid_trackers;
    
    batch.h_asks = nullptr;
    batch.h_bids = nullptr;
    batch.h_trades = nullptr;
    batch.h_ask_metadata = nullptr;
    batch.h_ask_buckets = nullptr;
    batch.h_ask_price_map = nullptr;
    batch.h_ask_order_id_map = nullptr;
    batch.h_ask_trackers = nullptr;
    batch.h_bid_metadata = nullptr;
    batch.h_bid_buckets = nullptr;
    batch.h_bid_price_map = nullptr;
    batch.h_bid_order_id_map = nullptr;
    batch.h_bid_trackers = nullptr;
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
    size_t metadata_size = batch.num_books * batch.n_orders_per_book * sizeof(OrderMetadata);
    size_t buckets_size = batch.num_books * batch.n_price_buckets_per_book * sizeof(PriceBucket);
    size_t price_map_size = batch.num_books * batch.price_map_size * sizeof(PriceMapEntry);
    size_t order_id_map_size = batch.num_books * batch.order_id_map_size * sizeof(OrderIDMapEntry);
    size_t trackers_size = batch.num_books * sizeof(BestPriceTracker);

    if (copy_asks && batch.h_asks && batch.d_asks) {
        cudaMemcpy(batch.d_asks, batch.h_asks, orders_size, cudaMemcpyHostToDevice);
        if (batch.h_ask_metadata && batch.d_ask_metadata) {
            cudaMemcpy(batch.d_ask_metadata, batch.h_ask_metadata, metadata_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_ask_buckets && batch.d_ask_buckets) {
            cudaMemcpy(batch.d_ask_buckets, batch.h_ask_buckets, buckets_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_ask_price_map && batch.d_ask_price_map) {
            cudaMemcpy(batch.d_ask_price_map, batch.h_ask_price_map, price_map_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_ask_order_id_map && batch.d_ask_order_id_map) {
            cudaMemcpy(batch.d_ask_order_id_map, batch.h_ask_order_id_map, order_id_map_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_ask_trackers && batch.d_ask_trackers) {
            cudaMemcpy(batch.d_ask_trackers, batch.h_ask_trackers, trackers_size, cudaMemcpyHostToDevice);
        }
    }
    if (copy_bids && batch.h_bids && batch.d_bids) {
        cudaMemcpy(batch.d_bids, batch.h_bids, orders_size, cudaMemcpyHostToDevice);
        if (batch.h_bid_metadata && batch.d_bid_metadata) {
            cudaMemcpy(batch.d_bid_metadata, batch.h_bid_metadata, metadata_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_bid_buckets && batch.d_bid_buckets) {
            cudaMemcpy(batch.d_bid_buckets, batch.h_bid_buckets, buckets_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_bid_price_map && batch.d_bid_price_map) {
            cudaMemcpy(batch.d_bid_price_map, batch.h_bid_price_map, price_map_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_bid_order_id_map && batch.d_bid_order_id_map) {
            cudaMemcpy(batch.d_bid_order_id_map, batch.h_bid_order_id_map, order_id_map_size, cudaMemcpyHostToDevice);
        }
        if (batch.h_bid_trackers && batch.d_bid_trackers) {
            cudaMemcpy(batch.d_bid_trackers, batch.h_bid_trackers, trackers_size, cudaMemcpyHostToDevice);
        }
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
    size_t metadata_size = batch.num_books * batch.n_orders_per_book * sizeof(OrderMetadata);
    size_t buckets_size = batch.num_books * batch.n_price_buckets_per_book * sizeof(PriceBucket);
    size_t price_map_size = batch.num_books * batch.price_map_size * sizeof(PriceMapEntry);
    size_t order_id_map_size = batch.num_books * batch.order_id_map_size * sizeof(OrderIDMapEntry);
    size_t trackers_size = batch.num_books * sizeof(BestPriceTracker);

    if (copy_asks && batch.h_asks && batch.d_asks) {
        cudaMemcpy(batch.h_asks, batch.d_asks, orders_size, cudaMemcpyDeviceToHost);
        if (batch.h_ask_metadata && batch.d_ask_metadata) {
            cudaMemcpy(batch.h_ask_metadata, batch.d_ask_metadata, metadata_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_ask_buckets && batch.d_ask_buckets) {
            cudaMemcpy(batch.h_ask_buckets, batch.d_ask_buckets, buckets_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_ask_price_map && batch.d_ask_price_map) {
            cudaMemcpy(batch.h_ask_price_map, batch.d_ask_price_map, price_map_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_ask_order_id_map && batch.d_ask_order_id_map) {
            cudaMemcpy(batch.h_ask_order_id_map, batch.d_ask_order_id_map, order_id_map_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_ask_trackers && batch.d_ask_trackers) {
            cudaMemcpy(batch.h_ask_trackers, batch.d_ask_trackers, trackers_size, cudaMemcpyDeviceToHost);
        }
    }
    if (copy_bids && batch.h_bids && batch.d_bids) {
        cudaMemcpy(batch.h_bids, batch.d_bids, orders_size, cudaMemcpyDeviceToHost);
        if (batch.h_bid_metadata && batch.d_bid_metadata) {
            cudaMemcpy(batch.h_bid_metadata, batch.d_bid_metadata, metadata_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_bid_buckets && batch.d_bid_buckets) {
            cudaMemcpy(batch.h_bid_buckets, batch.d_bid_buckets, buckets_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_bid_price_map && batch.d_bid_price_map) {
            cudaMemcpy(batch.h_bid_price_map, batch.d_bid_price_map, price_map_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_bid_order_id_map && batch.d_bid_order_id_map) {
            cudaMemcpy(batch.h_bid_order_id_map, batch.d_bid_order_id_map, order_id_map_size, cudaMemcpyDeviceToHost);
        }
        if (batch.h_bid_trackers && batch.d_bid_trackers) {
            cudaMemcpy(batch.h_bid_trackers, batch.d_bid_trackers, trackers_size, cudaMemcpyDeviceToHost);
        }
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
    int total_buckets = batch.num_books * batch.n_price_buckets_per_book;
    int total_price_map = batch.num_books * batch.price_map_size;
    int total_order_id_map = batch.num_books * batch.order_id_map_size;

    for (int i = 0; i < total_orders; i++) {
        batch.h_asks[i] = Order();  // Empty order
        batch.h_bids[i] = Order();
        if (batch.h_ask_metadata) batch.h_ask_metadata[i] = OrderMetadata();
        if (batch.h_bid_metadata) batch.h_bid_metadata[i] = OrderMetadata();
    }

    for (int i = 0; i < total_trades; i++) {
        batch.h_trades[i] = Trade();  // Empty trade
    }

    if (batch.h_ask_buckets) {
        for (int i = 0; i < total_buckets; i++) {
            batch.h_ask_buckets[i] = PriceBucket();
        }
    }
    if (batch.h_bid_buckets) {
        for (int i = 0; i < total_buckets; i++) {
            batch.h_bid_buckets[i] = PriceBucket();
        }
    }

    if (batch.h_ask_price_map) {
        for (int i = 0; i < total_price_map; i++) {
            batch.h_ask_price_map[i] = PriceMapEntry();
        }
    }
    if (batch.h_bid_price_map) {
        for (int i = 0; i < total_price_map; i++) {
            batch.h_bid_price_map[i] = PriceMapEntry();
        }
    }

    if (batch.h_ask_order_id_map) {
        for (int i = 0; i < total_order_id_map; i++) {
            batch.h_ask_order_id_map[i] = OrderIDMapEntry();
        }
    }
    if (batch.h_bid_order_id_map) {
        for (int i = 0; i < total_order_id_map; i++) {
            batch.h_bid_order_id_map[i] = OrderIDMapEntry();
        }
    }

    if (batch.h_ask_trackers) {
        for (int i = 0; i < batch.num_books; i++) {
            batch.h_ask_trackers[i] = BestPriceTracker();
        }
    }
    if (batch.h_bid_trackers) {
        for (int i = 0; i < batch.num_books; i++) {
            batch.h_bid_trackers[i] = BestPriceTracker();
        }
    }
}

void init_orderbooks_device(const OrderbookBatch& batch) {
    init_orderbooks_kernel<<<1, 256>>>(batch, batch.num_books);
    cudaDeviceSynchronize();
}

// ============================================================================
// PRICE-AWARE DEVICE FUNCTION IMPLEMENTATIONS
// ============================================================================

__device__ int32_t find_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t map_size
) {
    if (price == EMPTY_PRICE) return EMPTY_INDEX;
    
    int32_t start_idx = hash_price(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].price == price && price_map[idx].is_active) {
            return price_map[idx].bucket_idx;
        }
        if (price_map[idx].is_empty()) {
            return EMPTY_INDEX;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return EMPTY_INDEX;  // Map is full, not found
}

__device__ bool insert_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t bucket_idx,
    int32_t map_size
) {
    if (price == EMPTY_PRICE) return false;
    
    int32_t start_idx = hash_price(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].is_empty() || price_map[idx].is_tombstone() ||
            (price_map[idx].price == price && price_map[idx].is_active)) {
            price_map[idx].price = price;
            price_map[idx].bucket_idx = bucket_idx;
            price_map[idx].is_active = true;
            price_map[idx].was_deleted = false;
            return true;
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return false;  // Map is full
}

__device__ void remove_price_bucket(
    PriceMapEntry* price_map,
    int32_t price,
    int32_t map_size
) {
    if (price == EMPTY_PRICE) return;
    
    int32_t start_idx = hash_price(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].price == price && price_map[idx].is_active) {
            price_map[idx].is_active = false;
            price_map[idx].was_deleted = true;
            price_map[idx].price = EMPTY_PRICE;
            price_map[idx].bucket_idx = EMPTY_INDEX;
            return;
        }
        if (price_map[idx].is_empty()) {
            return;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
}

__device__ int32_t find_order_by_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t map_size
) {
    if (order_id == 0) return EMPTY_INDEX;
    
    int32_t start_idx = hash_order_id(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active) {
            return order_id_map[idx].order_idx;
        }
        if (order_id_map[idx].is_empty()) {
            return EMPTY_INDEX;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return EMPTY_INDEX;  // Not found
}

__device__ bool insert_order_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t order_idx,
    int32_t map_size
) {
    if (order_id == 0) return false;
    
    int32_t start_idx = hash_order_id(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].is_empty() || order_id_map[idx].is_tombstone() ||
            (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active)) {
            order_id_map[idx].order_id = order_id;
            order_id_map[idx].order_idx = order_idx;
            order_id_map[idx].is_active = true;
            order_id_map[idx].was_deleted = false;
            return true;
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return false;  // Map is full
}

__device__ void remove_order_id_map(
    OrderIDMapEntry* order_id_map,
    int32_t order_id,
    int32_t map_size
) {
    if (order_id == 0) return;
    
    int32_t start_idx = hash_order_id(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active) {
            order_id_map[idx].is_active = false;
            order_id_map[idx].was_deleted = true;
            order_id_map[idx].order_id = 0;
            order_id_map[idx].order_idx = EMPTY_INDEX;
            return;
        }
        if (order_id_map[idx].is_empty()) {
            return;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
}

__device__ int32_t get_or_create_price_bucket(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    int32_t price,
    int32_t n_buckets,
    int32_t map_size
) {
    if (price == EMPTY_PRICE) return EMPTY_INDEX;
    
    // First try to find existing bucket
    int32_t bucket_idx = find_price_bucket(price_map, price, map_size);
    if (bucket_idx != EMPTY_INDEX) {
        return bucket_idx;
    }
    
    // Find empty bucket slot
    for (int32_t i = 0; i < n_buckets; i++) {
        if (!buckets[i].is_active) {
            buckets[i].price = price;
            buckets[i].is_active = true;
            buckets[i].head_idx = EMPTY_INDEX;
            buckets[i].tail_idx = EMPTY_INDEX;
            buckets[i].total_quantity = 0;
            
            // Insert into price map
            if (insert_price_bucket(price_map, price, i, map_size)) {
                return i;
            }
            // Map insert failed, mark bucket as inactive again
            buckets[i].is_active = false;
            return EMPTY_INDEX;
        }
    }
    
    return EMPTY_INDEX;  // No free buckets
}

__device__ void add_order_to_bucket(
    PriceBucket* buckets,
    OrderMetadata* metadata,
    Order* orders,
    int32_t bucket_idx,
    int32_t order_idx
) {
    if (bucket_idx == EMPTY_INDEX || order_idx == EMPTY_INDEX) return;
    
    PriceBucket& bucket = buckets[bucket_idx];
    OrderMetadata& meta = metadata[order_idx];
    Order& order = orders[order_idx];
    
    meta.price_bucket_idx = bucket_idx;
    meta.is_valid = true;
    
    // Add to tail (FIFO)
    if (bucket.is_empty()) {
        // First order at this price
        bucket.head_idx = order_idx;
        bucket.tail_idx = order_idx;
        meta.next_idx = EMPTY_INDEX;
        meta.prev_idx = EMPTY_INDEX;
    } else {
        // Add to tail
        int32_t old_tail = bucket.tail_idx;
        OrderMetadata& old_tail_meta = metadata[old_tail];
        
        old_tail_meta.next_idx = order_idx;
        meta.prev_idx = old_tail;
        meta.next_idx = EMPTY_INDEX;
        bucket.tail_idx = order_idx;
    }
    
    bucket.total_quantity += order.quantity;
}

__device__ void remove_order_from_bucket(
    PriceBucket* buckets,
    OrderMetadata* metadata,
    Order* orders,
    int32_t bucket_idx,
    int32_t order_idx,
    int32_t removed_quantity
) {
    if (bucket_idx == EMPTY_INDEX || order_idx == EMPTY_INDEX) return;
    (void)orders;
    
    PriceBucket& bucket = buckets[bucket_idx];
    OrderMetadata& meta = metadata[order_idx];
    
    // Remove from linked list
    if (meta.prev_idx != EMPTY_INDEX) {
        metadata[meta.prev_idx].next_idx = meta.next_idx;
    } else {
        // This was head
        bucket.head_idx = meta.next_idx;
    }
    
    if (meta.next_idx != EMPTY_INDEX) {
        metadata[meta.next_idx].prev_idx = meta.prev_idx;
    } else {
        // This was tail
        bucket.tail_idx = meta.prev_idx;
    }
    
    int32_t qty = removed_quantity > 0 ? removed_quantity : 0;
    bucket.total_quantity -= qty;
    if (bucket.total_quantity < 0) {
        bucket.total_quantity = 0;
    }
    
    // Clear metadata
    meta.next_idx = EMPTY_INDEX;
    meta.prev_idx = EMPTY_INDEX;
    meta.price_bucket_idx = EMPTY_INDEX;
    meta.is_valid = false;
    
    // If bucket is now empty, mark as inactive
    if (bucket.is_empty()) {
        bucket.is_active = false;
        bucket.price = EMPTY_PRICE;
        bucket.total_quantity = 0;
    }
}

__device__ void update_best_ask_price(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    BestPriceTracker* tracker,
    int32_t n_buckets,
    int32_t map_size
) {
    int32_t best_price = MAX_INT;
    int32_t best_bucket_idx = EMPTY_INDEX;
    
    // Scan all active buckets to find minimum price
    for (int32_t i = 0; i < n_buckets; i++) {
        if (buckets[i].is_active && !buckets[i].is_empty()) {
            if (buckets[i].price < best_price) {
                best_price = buckets[i].price;
                best_bucket_idx = i;
            }
        }
    }
    
    tracker->best_ask_price = best_price;
    tracker->best_ask_bucket_idx = best_bucket_idx;
}

__device__ void update_best_bid_price(
    PriceBucket* buckets,
    PriceMapEntry* price_map,
    BestPriceTracker* tracker,
    int32_t n_buckets,
    int32_t map_size
) {
    int32_t best_price = EMPTY_PRICE;
    int32_t best_bucket_idx = EMPTY_INDEX;
    
    // Scan all active buckets to find maximum price
    for (int32_t i = 0; i < n_buckets; i++) {
        if (buckets[i].is_active && !buckets[i].is_empty()) {
            if (buckets[i].price > best_price) {
                best_price = buckets[i].price;
                best_bucket_idx = i;
            }
        }
    }
    
    tracker->best_bid_price = best_price;
    tracker->best_bid_bucket_idx = best_bucket_idx;
}

__device__ int32_t get_top_ask_order_idx_price_aware(const OrderbookState& state) {
    if (!state.ask_tracker->has_best_ask()) {
        return EMPTY_INDEX;
    }
    
    int32_t bucket_idx = state.ask_tracker->best_ask_bucket_idx;
    if (bucket_idx == EMPTY_INDEX || bucket_idx >= state.n_price_buckets) {
        return EMPTY_INDEX;
    }
    
    PriceBucket& bucket = state.ask_buckets[bucket_idx];
    if (bucket.is_empty()) {
        return EMPTY_INDEX;
    }
    
    return bucket.head_idx;  // First order at best price (FIFO)
}

__device__ int32_t get_top_bid_order_idx_price_aware(const OrderbookState& state) {
    if (!state.bid_tracker->has_best_bid()) {
        return EMPTY_INDEX;
    }
    
    int32_t bucket_idx = state.bid_tracker->best_bid_bucket_idx;
    if (bucket_idx == EMPTY_INDEX || bucket_idx >= state.n_price_buckets) {
        return EMPTY_INDEX;
    }
    
    PriceBucket& bucket = state.bid_buckets[bucket_idx];
    if (bucket.is_empty()) {
        return EMPTY_INDEX;
    }
    
    return bucket.head_idx;  // First order at best price (FIFO)
}

} // namespace cuda_orderbook

