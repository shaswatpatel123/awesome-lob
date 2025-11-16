/**
 * Hash-Accelerated Orderbook Operations - Device Functions
 * 
 * This file implements LOB operations using hash tables for O(1) order lookup
 * Combined with lazy sorting for O(1) best price access
 * 
 * Architecture: Hash table + sorted array hybrid
 * - Hash table: Fast cancel by order_id (O(1))
 * - Sorted array: Fast best price access (O(1) after sort)
 * - Lazy sorting: Only sort when best price is queried
 */

#include "types.h"
#include "simple_hash.cuh"
#include "cuco_wrapper.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace cuda_orderbook {

// ============================================================================
// COMPARATORS FOR SORTING
// ============================================================================

/**
 * Comparator for ask orders (ascending price, then time priority)
 */
struct AskComparator {
    __device__ bool operator()(const Order& a, const Order& b) const {
        // Empty orders go to the end
        if (a.price == EMPTY_PRICE) return false;
        if (b.price == EMPTY_PRICE) return true;
        
        // Price priority (ascending for asks)
        if (a.price != b.price) {
            return a.price < b.price;
        }
        
        // Time priority (earlier is better)
        if (a.time_sec != b.time_sec) {
            return a.time_sec < b.time_sec;
        }
        
        return a.time_ns < b.time_ns;
    }
};

/**
 * Comparator for bid orders (descending price, then time priority)
 */
struct BidComparator {
    __device__ bool operator()(const Order& a, const Order& b) const {
        // Empty orders go to the end
        if (a.price == EMPTY_PRICE) return false;
        if (b.price == EMPTY_PRICE) return true;
        
        // Price priority (descending for bids)
        if (a.price != b.price) {
            return a.price > b.price;
        }
        
        // Time priority (earlier is better)
        if (a.time_sec != b.time_sec) {
            return a.time_sec < b.time_sec;
        }
        
        return a.time_ns < b.time_ns;
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Find first empty slot in order array
 * Returns -1 if array is full
 */
__device__ int find_empty_slot(const Order* orders, int n_orders) {
    for (int i = 0; i < n_orders; i++) {
        if (orders[i].price == EMPTY_PRICE) {
            return i;
        }
    }
    return -1;
}

/**
 * Ensure orders array is sorted
 * Uses Thrust for efficient GPU sorting
 */
__device__ void ensure_asks_sorted(Order* asks, int n_orders, bool* is_sorted) {
    if (!*is_sorted) {
        thrust::sort(thrust::device, asks, asks + n_orders, AskComparator());
        *is_sorted = true;
    }
}

__device__ void ensure_bids_sorted(Order* bids, int n_orders, bool* is_sorted) {
    if (!*is_sorted) {
        thrust::sort(thrust::device, bids, bids + n_orders, BidComparator());
        *is_sorted = true;
    }
}

// ============================================================================
// ADD ORDER
// ============================================================================

/**
 * Add order with hash index
 * O(1) operation: find empty slot, add to array, insert to hash
 */
__device__ void add_order_hash_device(
    Order* orderside,
    void* hash_map,
    const Message& msg,
    int n_orders,
    HashImplementation impl,
    bool* is_sorted
) {
    // Find empty slot
    int idx = find_empty_slot(orderside, n_orders);
    if (idx == -1) {
        // Orderbook full
        return;
    }
    
    // Create order
    orderside[idx].price = msg.price;
    orderside[idx].quantity = max(0, msg.quantity);
    orderside[idx].order_id = msg.order_id;
    orderside[idx].trader_id = msg.trader_id;
    orderside[idx].time_sec = msg.time_sec;
    orderside[idx].time_ns = msg.time_ns;
    
    // Add to hash index
    hash_map_insert(hash_map, msg.order_id, idx, impl);
    
    // Mark as unsorted
    *is_sorted = false;
}

// ============================================================================
// CANCEL ORDER
// ============================================================================

/**
 * Cancel order using hash lookup
 * O(1) operation: hash lookup, modify quantity, erase if needed
 */
__device__ void cancel_order_hash_device(
    Order* orderside,
    void* hash_map,
    const Message& msg,
    int n_orders,
    HashImplementation impl,
    bool* is_sorted
) {
    // Try hash lookup first
    int32_t idx = hash_map_find(hash_map, msg.order_id, impl);
    
    // Fallback: search by price for INITID orders
    if (idx == -1) {
        for (int i = 0; i < n_orders; i++) {
            if (orderside[i].price == msg.price && 
                orderside[i].order_id <= INITID) {
                idx = i;
                break;
            }
        }
    }
    
    if (idx == -1) {
        // Order not found
        return;
    }
    
    // Reduce quantity
    orderside[idx].quantity -= msg.quantity;
    
    // If quantity <= 0, remove order
    if (orderside[idx].quantity <= 0) {
        orderside[idx].price = EMPTY_PRICE;
        orderside[idx].quantity = 0;
        orderside[idx].order_id = 0;
        orderside[idx].trader_id = 0;
        orderside[idx].time_sec = 0;
        orderside[idx].time_ns = 0;
        
        // Remove from hash
        hash_map_erase(hash_map, msg.order_id, impl);
        
        // Mark as unsorted (for cleanup)
        *is_sorted = false;
    }
}

// ============================================================================
// BEST PRICE QUERIES
// ============================================================================

/**
 * Get best ask price
 * O(1) if already sorted, O(n log n) first time
 */
__device__ int32_t get_best_ask_hash_device(
    Order* asks,
    int n_orders,
    bool* is_sorted
) {
    // Ensure sorted
    ensure_asks_sorted(asks, n_orders, is_sorted);
    
    // Best ask is first element (if not empty)
    if (asks[0].price != EMPTY_PRICE) {
        return asks[0].price;
    }
    
    return -1;  // No asks available
}

/**
 * Get best bid price
 * O(1) if already sorted, O(n log n) first time
 */
__device__ int32_t get_best_bid_hash_device(
    Order* bids,
    int n_orders,
    bool* is_sorted
) {
    // Ensure sorted
    ensure_bids_sorted(bids, n_orders, is_sorted);
    
    // Best bid is first element (if not empty)
    if (bids[0].price != EMPTY_PRICE) {
        return bids[0].price;
    }
    
    return -1;  // No bids available
}

/**
 * Get best ask order index (after sorting)
 */
__device__ int get_best_ask_idx_hash_device(
    Order* asks,
    int n_orders,
    bool* is_sorted
) {
    ensure_asks_sorted(asks, n_orders, is_sorted);
    
    if (asks[0].price != EMPTY_PRICE) {
        return 0;
    }
    return -1;
}

/**
 * Get best bid order index (after sorting)
 */
__device__ int get_best_bid_idx_hash_device(
    Order* bids,
    int n_orders,
    bool* is_sorted
) {
    ensure_bids_sorted(bids, n_orders, is_sorted);
    
    if (bids[0].price != EMPTY_PRICE) {
        return 0;
    }
    return -1;
}

// ============================================================================
// MATCHING ENGINE
// ============================================================================

/**
 * Match a single order (same logic as original, but uses sorted array)
 */
__device__ void match_single_order_hash_device(
    Order* orderside,
    int top_order_idx,
    int32_t& qtm_remaining,
    Trade* trades,
    int n_trades,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns,
    void* hash_map,
    HashImplementation impl
) {
    if (top_order_idx < 0) return;
    if (qtm_remaining <= 0) return;
    
    Order& passive_order = orderside[top_order_idx];
    if (passive_order.price == EMPTY_PRICE) return;
    
    // Calculate matched quantity
    int32_t matched_qty = min(qtm_remaining, passive_order.quantity);
    int32_t new_quantity = max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity
    qtm_remaining = max(0, qtm_remaining - passive_order.quantity);
    
    // Record trade
    for (int i = 0; i < n_trades; i++) {
        if (trades[i].price == EMPTY_PRICE) {
            trades[i].price = passive_order.price;
            trades[i].quantity = matched_qty;
            trades[i].passive_order_id = passive_order.order_id;
            trades[i].aggressive_order_id = aggressive_order_id;
            trades[i].time_sec = time_sec;
            trades[i].time_ns = time_ns;
            break;
        }
    }
    
    // Update passive order
    passive_order.quantity = new_quantity;
    
    if (new_quantity <= 0) {
        passive_order.price = EMPTY_PRICE;
        hash_map_erase(hash_map, passive_order.order_id, impl);
        passive_order.order_id = 0;
        passive_order.trader_id = 0;
        passive_order.time_sec = 0;
        passive_order.time_ns = 0;
    }
}

/**
 * Match against asks (buy order matching)
 */
__device__ void match_against_asks_hash_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades,
    void* ask_hash_map,
    void* bid_hash_map,
    HashImplementation impl,
    bool* asks_sorted,
    bool* bids_sorted
) {
    // Ensure asks are sorted for matching
    ensure_asks_sorted(asks, n_orders, asks_sorted);
    
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    int current_idx = 0;
    
    // Match against sorted asks
    while (qtm_remaining > 0 && current_idx < n_orders) {
        if (asks[current_idx].price == EMPTY_PRICE) break;
        if (asks[current_idx].price > limit_price) break;
        
        match_single_order_hash_device(
            asks,
            current_idx,
            qtm_remaining,
            trades,
            n_trades,
            msg.order_id,
            msg.time_sec,
            msg.time_ns,
            ask_hash_map,
            impl
        );
        
        current_idx++;
    }
}

/**
 * Match against bids (sell order matching)
 */
__device__ void match_against_bids_hash_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades,
    void* ask_hash_map,
    void* bid_hash_map,
    HashImplementation impl,
    bool* asks_sorted,
    bool* bids_sorted
) {
    // Ensure bids are sorted for matching
    ensure_bids_sorted(bids, n_orders, bids_sorted);
    
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    int current_idx = 0;
    
    // Match against sorted bids
    while (qtm_remaining > 0 && current_idx < n_orders) {
        if (bids[current_idx].price == EMPTY_PRICE) break;
        if (bids[current_idx].price < limit_price) break;
        
        match_single_order_hash_device(
            bids,
            current_idx,
            qtm_remaining,
            trades,
            n_trades,
            msg.order_id,
            msg.time_sec,
            msg.time_ns,
            bid_hash_map,
            impl
        );
        
        current_idx++;
    }
}

// ============================================================================
// PROCESS MESSAGE (MAIN DISPATCH)
// ============================================================================

/**
 * Process a single message with hash acceleration
 */
__device__ void process_message_hash_device(
    HashOrderbookState* state,
    const Message& msg
) {
    Order* asks = state->asks;
    Order* bids = state->bids;
    Trade* trades = state->trades;
    void* ask_hash = state->ask_hash_map;
    void* bid_hash = state->bid_hash_map;
    int n_orders = state->n_orders;
    int n_trades = state->n_trades;
    HashImplementation impl = state->hash_impl;
    
    if (msg.type == Message::CANCEL || msg.type == Message::DELETE) {
        // Cancel order
        if (msg.side == Message::ASK) {
            cancel_order_hash_device(asks, ask_hash, msg, n_orders, impl, &state->asks_sorted);
        } else if (msg.side == Message::BID) {
            cancel_order_hash_device(bids, bid_hash, msg, n_orders, impl, &state->bids_sorted);
        }
    }
    else if (msg.type == Message::LIMIT) {
        // Limit order
        if (msg.side == Message::ASK) {
            // Match against bids first
            int32_t initial_qty = msg.quantity;
            match_against_bids_hash_device(
                asks, bids, trades, msg, n_orders, n_trades,
                ask_hash, bid_hash, impl,
                &state->asks_sorted, &state->bids_sorted
            );
            
            // Calculate remaining (simplified - could track in match function)
            // For now, add full order and let matching handle it
            Message remaining_msg = msg;
            add_order_hash_device(asks, ask_hash, remaining_msg, n_orders, impl, &state->asks_sorted);
            
        } else if (msg.side == Message::BID) {
            // Match against asks first
            match_against_asks_hash_device(
                asks, bids, trades, msg, n_orders, n_trades,
                ask_hash, bid_hash, impl,
                &state->asks_sorted, &state->bids_sorted
            );
            
            Message remaining_msg = msg;
            add_order_hash_device(bids, bid_hash, remaining_msg, n_orders, impl, &state->bids_sorted);
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            match_msg.price = MAX_INT;
            match_against_asks_hash_device(
                asks, bids, trades, match_msg, n_orders, n_trades,
                ask_hash, bid_hash, impl,
                &state->asks_sorted, &state->bids_sorted
            );
        } else if (msg.side == Message::ASK) {
            match_msg.price = 0;
            match_against_bids_hash_device(
                asks, bids, trades, match_msg, n_orders, n_trades,
                ask_hash, bid_hash, impl,
                &state->asks_sorted, &state->bids_sorted
            );
        }
    }
}

} // namespace cuda_orderbook

