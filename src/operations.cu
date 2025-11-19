/**
 * Orderbook Operations - Device Functions
 * 
 * This file contains all device functions for orderbook operations.
 * These functions are called from CUDA kernels.
 * 
 * Team 1: add_order_device, cancel_order_device
 * Team 2: matching functions (get_top_*, match_against_*)
 */

#include "types.h"
#include "utils.cuh"

namespace cuda_orderbook {

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Remove orders with zero or negative quantity (price-aware version)
 * Maps to JAX __removeZeroNegQuant (JaxOrderBookArrays.py:40-41)
 * 
 * Note: In price-aware version, this is handled automatically during
 * cancel/match operations. This function is kept for compatibility but
 * is less efficient than direct operations.
 */
__device__ void remove_zero_neg_quant_device(const OrderbookState& state, bool is_ask_side) {
    Order* orderside = is_ask_side ? state.asks : state.bids;
    OrderMetadata* metadata = is_ask_side ? state.ask_metadata : state.bid_metadata;
    PriceBucket* buckets = is_ask_side ? state.ask_buckets : state.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? state.ask_price_map : state.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? state.ask_order_id_map : state.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? state.ask_tracker : state.bid_tracker;
    
    for (int i = 0; i < state.n_orders; i++) {
        if (orderside[i].quantity <= 0 && orderside[i].price != EMPTY_PRICE) {
            int32_t bucket_idx = metadata[i].price_bucket_idx;
            int32_t removed_qty = max(0, orderside[i].quantity);
            
            // Remove from price bucket
            if (bucket_idx != EMPTY_INDEX) {
                remove_order_from_bucket(buckets, metadata, orderside, bucket_idx, i, removed_qty);
                
                // If bucket is now empty, remove from price map
                if (buckets[bucket_idx].is_empty()) {
                    remove_price_bucket(price_map, orderside[i].price, state.price_map_size);
                }
                
                // Update best price tracker if needed
                bool need_update = false;
                if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                    need_update = true;
                } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                    need_update = true;
                }
                
                if (need_update) {
                    if (is_ask_side) {
                        update_best_ask_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                    } else {
                        update_best_bid_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                    }
                }
            }
            
            // Remove from order-ID map
            remove_order_id_map(order_id_map, orderside[i].order_id, state.order_id_map_size);
            
            // Mark as empty
            orderside[i].price = EMPTY_PRICE;
            orderside[i].quantity = 0;
            orderside[i].order_id = 0;
            orderside[i].trader_id = 0;
            orderside[i].time_sec = 0;
            orderside[i].time_ns = 0;
        }
    }
}

// ============================================================================
// TEAM 1: ADD AND CANCEL OPERATIONS
// ============================================================================

/**
 * Add order to orderside (price-aware version)
 * Maps to JAX add_order (JaxOrderBookArrays.py:32-37)
 * 
 * Uses price-aware structures for O(1) price-level access
 * Maintains linked lists per price level with FIFO ordering
 */
__device__ void add_order_device(
    const OrderbookState& state,
    bool is_ask_side,
    const Message& msg
) {
    if (msg.price == EMPTY_PRICE || msg.quantity <= 0) return;
    
    // Select appropriate side structures
    Order* orderside = is_ask_side ? state.asks : state.bids;
    OrderMetadata* metadata = is_ask_side ? state.ask_metadata : state.bid_metadata;
    PriceBucket* buckets = is_ask_side ? state.ask_buckets : state.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? state.ask_price_map : state.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? state.ask_order_id_map : state.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? state.ask_tracker : state.bid_tracker;
    
    // Find first empty slot
    int empty_idx = -1;
    for (int i = 0; i < state.n_orders; i++) {
        if (orderside[i].price == EMPTY_PRICE) {
            empty_idx = i;
            break;
        }
    }
    
    if (empty_idx == -1) {
        // Orderbook full - cannot add
        return;
    }
    
    // Add the order
    Order& order = orderside[empty_idx];
    order.price = msg.price;
    order.quantity = max(0, msg.quantity);
    order.order_id = msg.order_id;
    order.trader_id = msg.trader_id;
    order.time_sec = msg.time_sec;
    order.time_ns = msg.time_ns;
    
    // Get or create price bucket
    int32_t bucket_idx = get_or_create_price_bucket(
        buckets, price_map, msg.price, 
        state.n_price_buckets, state.price_map_size
    );
    
    if (bucket_idx == EMPTY_INDEX) {
        // Cannot create bucket - revert order
        order.price = EMPTY_PRICE;
        return;
    }
    
    // Add order to price bucket (at tail for FIFO)
    add_order_to_bucket(buckets, metadata, orderside, bucket_idx, empty_idx);
    
    // Insert into order-ID map for O(1) cancel lookup
    insert_order_id_map(order_id_map, msg.order_id, empty_idx, state.order_id_map_size);
    
    // Update best price tracker
    if (is_ask_side) {
        // For asks, update if this is a better (lower) price
        if (msg.price < tracker->best_ask_price) {
            tracker->best_ask_price = msg.price;
            tracker->best_ask_bucket_idx = bucket_idx;
        }
    } else {
        // For bids, update if this is a better (higher) price
        if (msg.price > tracker->best_bid_price) {
            tracker->best_bid_price = msg.price;
            tracker->best_bid_bucket_idx = bucket_idx;
        }
    }
}

/**
 * Cancel order from orderside (price-aware version)
 * Maps to JAX cancel_order (JaxOrderBookArrays.py:52-65)
 * 
 * Uses O(1) order-ID lookup via hash map
 * Removes from price bucket if fully cancelled
 */
__device__ void cancel_order_device(
    const OrderbookState& state,
    bool is_ask_side,
    const Message& msg
) {
    // Select appropriate side structures
    Order* orderside = is_ask_side ? state.asks : state.bids;
    OrderMetadata* metadata = is_ask_side ? state.ask_metadata : state.bid_metadata;
    PriceBucket* buckets = is_ask_side ? state.ask_buckets : state.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? state.ask_price_map : state.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? state.ask_order_id_map : state.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? state.ask_tracker : state.bid_tracker;
    
    // Find order by ID using hash map (O(1))
    int32_t idx = find_order_by_id_map(order_id_map, msg.order_id, state.order_id_map_size);
    
    // If not found and this might be an INITID order, search by price
    if (idx == EMPTY_INDEX) {
        for (int i = 0; i < state.n_orders; i++) {
            if (orderside[i].price == msg.price && 
                orderside[i].order_id <= INITID &&
                orderside[i].price != EMPTY_PRICE) {
                idx = i;
                break;
            }
        }
    }
    
    if (idx == EMPTY_INDEX || idx >= state.n_orders) {
        // Order not found
        return;
    }
    
    Order& order = orderside[idx];
    if (order.price == EMPTY_PRICE) return;
    
    int32_t old_quantity = order.quantity;
    
    // Reduce quantity
    order.quantity = max(0, order.quantity - msg.quantity);
    
    // If fully cancelled, remove from structures
    if (order.quantity <= 0) {
        // Get bucket index from metadata
        int32_t bucket_idx = metadata[idx].price_bucket_idx;
        
        // Remove from price bucket
        if (bucket_idx != EMPTY_INDEX) {
            remove_order_from_bucket(buckets, metadata, orderside, bucket_idx, idx, old_quantity);
            
            // If bucket is now empty, remove from price map
            if (buckets[bucket_idx].is_empty()) {
                remove_price_bucket(price_map, order.price, state.price_map_size);
            }
            
            // Update best price tracker if we removed the best price
            bool need_update = false;
            if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                need_update = true;
            } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                need_update = true;
            }
            
            if (need_update) {
                if (is_ask_side) {
                    update_best_ask_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                } else {
                    update_best_bid_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                }
            }
        }
        
        // Remove from order-ID map
        remove_order_id_map(order_id_map, msg.order_id, state.order_id_map_size);
        
        // Clear order
        order.price = EMPTY_PRICE;
        order.quantity = 0;
        order.order_id = 0;
        order.trader_id = 0;
        order.time_sec = 0;
        order.time_ns = 0;
    } else {
        // Partial cancel - update bucket quantity
        int32_t bucket_idx = metadata[idx].price_bucket_idx;
        if (bucket_idx != EMPTY_INDEX) {
            int32_t qty_delta = old_quantity - order.quantity;
            buckets[bucket_idx].total_quantity = max(0, buckets[bucket_idx].total_quantity - qty_delta);
        }
    }
}

// ============================================================================
// TEAM 2: MATCHING ENGINE - PRIORITY SELECTION
// ============================================================================

/**
 * Get index of top ask order (price-aware version)
 * Uses best price tracker for O(1) access
 * Priority: Lowest price, then earliest time (FIFO within price level)
 */
__device__ int get_top_ask_order_idx(const OrderbookState& state) {
    return get_top_ask_order_idx_price_aware(state);
}

/**
 * Get index of top bid order (price-aware version)
 * Uses best price tracker for O(1) access
 * Priority: Highest price, then earliest time (FIFO within price level)
 */
__device__ int get_top_bid_order_idx(const OrderbookState& state) {
    return get_top_bid_order_idx_price_aware(state);
}

// ============================================================================
// TEAM 2: MATCHING ENGINE - ORDER MATCHING
// ============================================================================

/**
 * Match a single order and generate trade (price-aware version)
 * Maps to JAX match_order (JaxOrderBookArrays.py:78-86)
 * 
 * @param state Orderbook state
 * @param is_ask_side Whether matching against ask side
 * @param top_order_idx Index of order to match against
 * @param qtm_remaining Quantity remaining to match (will be updated)
 * @param aggressive_order_id ID of incoming order
 * @param time_sec Timestamp seconds
 * @param time_ns Timestamp nanoseconds
 */
__device__ void match_single_order_device(
    const OrderbookState& state,
    bool is_ask_side,
    int top_order_idx,
    int32_t& qtm_remaining,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns
) {
    if (top_order_idx < 0 || top_order_idx >= state.n_orders) return;
    if (qtm_remaining <= 0) return;
    
    Order* orderside = is_ask_side ? state.asks : state.bids;
    OrderMetadata* metadata = is_ask_side ? state.ask_metadata : state.bid_metadata;
    PriceBucket* buckets = is_ask_side ? state.ask_buckets : state.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? state.ask_price_map : state.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? state.ask_order_id_map : state.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? state.ask_tracker : state.bid_tracker;
    
    Order& passive_order = orderside[top_order_idx];
    if (passive_order.price == EMPTY_PRICE) return;
    
    // Calculate matched quantity
    int32_t matched_qty = min(qtm_remaining, passive_order.quantity);
    int32_t old_quantity = passive_order.quantity;
    int32_t new_quantity = max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity to match
    qtm_remaining = max(0, qtm_remaining - matched_qty);
    
    // Find empty trade slot and record trade
    for (int i = 0; i < state.n_trades; i++) {
        if (state.trades[i].price == EMPTY_PRICE) {
            state.trades[i].price = passive_order.price;
            state.trades[i].quantity = matched_qty;
            state.trades[i].passive_order_id = passive_order.order_id;
            state.trades[i].aggressive_order_id = aggressive_order_id;
            state.trades[i].time_sec = time_sec;
            state.trades[i].time_ns = time_ns;
            break;
        }
    }
    
    // Update passive order quantity
    passive_order.quantity = new_quantity;
    
    int32_t bucket_idx = metadata[top_order_idx].price_bucket_idx;
    
    // If fully matched, remove from structures
    if (new_quantity <= 0) {
        // Remove from price bucket
        if (bucket_idx != EMPTY_INDEX) {
            remove_order_from_bucket(buckets, metadata, orderside, bucket_idx, top_order_idx, old_quantity);
            
            // If bucket is now empty, remove from price map
            if (buckets[bucket_idx].is_empty()) {
                remove_price_bucket(price_map, passive_order.price, state.price_map_size);
            }
            
            // Update best price tracker if we removed the best price
            bool need_update = false;
            if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                need_update = true;
            } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                need_update = true;
            }
            
            if (need_update) {
                if (is_ask_side) {
                    update_best_ask_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                } else {
                    update_best_bid_price(buckets, price_map, tracker, state.n_price_buckets, state.price_map_size);
                }
            }
        }
        
        // Remove from order-ID map
        remove_order_id_map(order_id_map, passive_order.order_id, state.order_id_map_size);
        
        // Clear order
        passive_order.price = EMPTY_PRICE;
        passive_order.order_id = 0;
        passive_order.trader_id = 0;
        passive_order.time_sec = 0;
        passive_order.time_ns = 0;
    } else {
        // Partial match - update bucket quantity
        if (bucket_idx != EMPTY_INDEX) {
            int32_t qty_delta = old_quantity - new_quantity;
            buckets[bucket_idx].total_quantity = max(0, buckets[bucket_idx].total_quantity - qty_delta);
        }
    }
}

/**
 * Match against ask orders (for incoming buy order) - price-aware version
 * Maps to JAX _match_against_ask_orders (JaxOrderBookArrays.py:127-130)
 * 
 * Iteratively matches against best ask until:
 * - No more quantity to match (qtm_remaining <= 0)
 * - No more matching asks (price > limit_price)
 * - No more ask orders available
 */
__device__ void match_against_asks_device(
    const OrderbookState& state,
    const Message& msg
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid asks
    while (qtm_remaining > 0) {
        // Get best ask using price-aware lookup (O(1))
        int top_ask_idx = get_top_ask_order_idx(state);
        
        // Check if we can match
        if (top_ask_idx == EMPTY_INDEX) break;  // No asks available
        if (state.asks[top_ask_idx].price == EMPTY_PRICE) break;  // No valid ask
        if (state.asks[top_ask_idx].price > limit_price) break;  // Price too high
        
        // Match against this ask
        match_single_order_device(
            state,
            true,  // is_ask_side
            top_ask_idx,
            qtm_remaining,
            msg.order_id,
            msg.time_sec,
            msg.time_ns
        );
    }
}

/**
 * Match against bid orders (for incoming sell order) - price-aware version
 * Maps to JAX _match_against_bid_orders (JaxOrderBookArrays.py:115-118)
 * 
 * Iteratively matches against best bid until:
 * - No more quantity to match (qtm_remaining <= 0)
 * - No more matching bids (price < limit_price)
 * - No more bid orders available
 */
__device__ void match_against_bids_device(
    const OrderbookState& state,
    const Message& msg
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid bids
    while (qtm_remaining > 0) {
        // Get best bid using price-aware lookup (O(1))
        int top_bid_idx = get_top_bid_order_idx(state);
        
        // Check if we can match
        if (top_bid_idx == EMPTY_INDEX) break;  // No bids available
        if (state.bids[top_bid_idx].price == EMPTY_PRICE) break;  // No valid bid
        if (state.bids[top_bid_idx].price < limit_price) break;  // Price too low
        
        // Match against this bid
        match_single_order_device(
            state,
            false,  // is_ask_side
            top_bid_idx,
            qtm_remaining,
            msg.order_id,
            msg.time_sec,
            msg.time_ns
        );
    }
}

// ============================================================================
// COMBINED ORDER PROCESSING
// ============================================================================

/**
 * Process a single message (add, cancel, or match) - price-aware version
 * Maps to JAX cond_type_side (JaxOrderBookArrays.py:181-206)
 * 
 * Dispatches to appropriate function based on message type and side
 */
__device__ void process_message_device(
    const OrderbookState& state,
    const Message& msg
) {
    // Determine action based on type and side
    // Type: 1=limit, 2=cancel, 3=delete, 4=market
    // Side: -1=ask, 1=bid
    
    if (msg.type == Message::CANCEL || msg.type == Message::DELETE) {
        // Cancel order
        if (msg.side == Message::ASK) {
            cancel_order_device(state, true, msg);  // is_ask_side = true
        } else if (msg.side == Message::BID) {
            cancel_order_device(state, false, msg);  // is_ask_side = false
        }
    }
    else if (msg.type == Message::LIMIT) {
        // Limit order - need to track remaining quantity after matching
        if (msg.side == Message::ASK) {
            // Sell limit: match against bids, then add remainder
            int32_t qtm_remaining = msg.quantity;
            int32_t limit_price = msg.price;
            
            // Keep matching while we have quantity and valid bids
            while (qtm_remaining > 0) {
                int top_bid_idx = get_top_bid_order_idx(state);
                if (top_bid_idx == EMPTY_INDEX) break;
                if (state.bids[top_bid_idx].price == EMPTY_PRICE) break;
                if (state.bids[top_bid_idx].price < limit_price) break;
                
                match_single_order_device(
                    state, false, top_bid_idx, qtm_remaining,
                    msg.order_id, msg.time_sec, msg.time_ns
                );
            }
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = qtm_remaining;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_device(state, true, remaining_msg);  // is_ask_side = true
            }
        } else if (msg.side == Message::BID) {
            // Buy limit: match against asks, then add remainder
            int32_t qtm_remaining = msg.quantity;
            int32_t limit_price = msg.price;
            
            // Keep matching while we have quantity and valid asks
            while (qtm_remaining > 0) {
                int top_ask_idx = get_top_ask_order_idx(state);
                if (top_ask_idx == EMPTY_INDEX) break;
                if (state.asks[top_ask_idx].price == EMPTY_PRICE) break;
                if (state.asks[top_ask_idx].price > limit_price) break;
                
                match_single_order_device(
                    state, true, top_ask_idx, qtm_remaining,
                    msg.order_id, msg.time_sec, msg.time_ns
                );
            }
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = qtm_remaining;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_device(state, false, remaining_msg);  // is_ask_side = false
            }
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order - aggressive matching only (no remainder added)
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            // Buy market: match against asks at any price
            match_msg.price = MAX_INT;  // Will match any ask price
            match_against_asks_device(state, match_msg);
        } else if (msg.side == Message::ASK) {
            // Sell market: match against bids at any price
            match_msg.price = 0;  // Will match any bid price
            match_against_bids_device(state, match_msg);
        }
    }
}

} // namespace cuda_orderbook

