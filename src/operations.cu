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
 * Remove orders with zero or negative quantity
 * Maps to JAX __removeZeroNegQuant (JaxOrderBookArrays.py:40-41)
 */
__device__ void remove_zero_neg_quant_device(Order* orderside, int n_orders) {
    for (int i = 0; i < n_orders; i++) {
        if (orderside[i].quantity <= 0 && orderside[i].price != EMPTY_PRICE) {
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
 * Add order to orderside
 * Maps to JAX add_order (JaxOrderBookArrays.py:32-37)
 * 
 * Finds first empty slot and inserts the order
 * Cleans up any orders with <= 0 quantity
 */
__device__ void add_order_device(
    Order* orderside,
    const Message& msg,
    int n_orders
) {
    // Find first empty slot (price == -1)
    int empty_idx = -1;
    for (int i = 0; i < n_orders; i++) {
        if (orderside[i].price == EMPTY_PRICE) {
            empty_idx = i;
            break;
        }
    }
    
    if (empty_idx == -1) {
        // Orderbook full - cannot add
        // In production, might want to handle this differently
        return;
    }
    
    // Add the order
    orderside[empty_idx].price = msg.price;
    orderside[empty_idx].quantity = max(0, msg.quantity);
    orderside[empty_idx].order_id = msg.order_id;
    orderside[empty_idx].trader_id = msg.trader_id;
    orderside[empty_idx].time_sec = msg.time_sec;
    orderside[empty_idx].time_ns = msg.time_ns;
    
    // Clean up any orders with zero/negative quantity
    remove_zero_neg_quant_device(orderside, n_orders);
}

/**
 * Cancel order from orderside
 * Maps to JAX cancel_order (JaxOrderBookArrays.py:52-65)
 * 
 * Finds order by ID (or by price for INITID orders)
 * Reduces quantity, removes if <= 0
 */
__device__ void cancel_order_device(
    Order* orderside,
    const Message& msg,
    int n_orders
) {
    // First try to find by order_id
    int idx = -1;
    for (int i = 0; i < n_orders; i++) {
        if (orderside[i].order_id == msg.order_id) {
            idx = i;
            break;
        }
    }
    
    // If not found and this might be an INITID order, search by price
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
    
    // Clean up orders with zero/negative quantity
    remove_zero_neg_quant_device(orderside, n_orders);
}

// ============================================================================
// TEAM 2: MATCHING ENGINE - PRIORITY SELECTION
// ============================================================================

/**
 * Get index of top ask order (best ask with price-time priority)
 * Maps to JAX __get_top_ask_order_idx (JaxOrderBookArrays.py:98-106)
 * 
 * Priority: Lowest price, then earliest time (sec, then ns)
 */
__device__ int get_top_ask_order_idx(const Order* asks, int n_orders) {
    int best_idx = -1;
    int32_t min_price = MAX_INT;
    int32_t min_time_sec = MAX_INT;
    int32_t min_time_ns = MAX_INT;
    
    for (int i = 0; i < n_orders; i++) {
        // Skip empty orders
        if (asks[i].price == EMPTY_PRICE) continue;
        
        // Convert -1 prices to MAX_INT for comparison
        int32_t price = (asks[i].price == EMPTY_PRICE) ? MAX_INT : asks[i].price;
        
        // Check if this is a better price
        bool is_better = false;
        if (price < min_price) {
            is_better = true;
        } else if (price == min_price) {
            // Same price - check time priority
            if (asks[i].time_sec < min_time_sec) {
                is_better = true;
            } else if (asks[i].time_sec == min_time_sec && 
                       asks[i].time_ns < min_time_ns) {
                is_better = true;
            }
        }
        
        if (is_better) {
            best_idx = i;
            min_price = price;
            min_time_sec = asks[i].time_sec;
            min_time_ns = asks[i].time_ns;
        }
    }
    
    return best_idx;
}

/**
 * Get index of top bid order (best bid with price-time priority)
 * Maps to JAX __get_top_bid_order_idx (JaxOrderBookArrays.py:89-95)
 * 
 * Priority: Highest price, then earliest time (sec, then ns)
 */
__device__ int get_top_bid_order_idx(const Order* bids, int n_orders) {
    int best_idx = -1;
    int32_t max_price = -1;
    int32_t min_time_sec = MAX_INT;
    int32_t min_time_ns = MAX_INT;
    
    for (int i = 0; i < n_orders; i++) {
        // Skip empty orders
        if (bids[i].price == EMPTY_PRICE) continue;
        
        // Check if this is a better price
        bool is_better = false;
        if (bids[i].price > max_price) {
            is_better = true;
        } else if (bids[i].price == max_price) {
            // Same price - check time priority
            if (bids[i].time_sec < min_time_sec) {
                is_better = true;
            } else if (bids[i].time_sec == min_time_sec && 
                       bids[i].time_ns < min_time_ns) {
                is_better = true;
            }
        }
        
        if (is_better) {
            best_idx = i;
            max_price = bids[i].price;
            min_time_sec = bids[i].time_sec;
            min_time_ns = bids[i].time_ns;
        }
    }
    
    return best_idx;
}

// ============================================================================
// TEAM 2: MATCHING ENGINE - ORDER MATCHING
// ============================================================================

/**
 * Match a single order and generate trade
 * Maps to JAX match_order (JaxOrderBookArrays.py:78-86)
 * 
 * @param top_order_idx Index of order to match against
 * @param orderside Orders to match against
 * @param qtm_remaining Quantity remaining to match (will be updated)
 * @param trades Trade records array
 * @param n_trades Max trades
 * @param aggressive_order_id ID of incoming order
 * @param time_sec Timestamp seconds
 * @param time_ns Timestamp nanoseconds
 * @param n_orders Number of orders
 */
__device__ void match_single_order_device(
    int top_order_idx,
    Order* orderside,
    int32_t& qtm_remaining,
    Trade* trades,
    int n_trades,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns,
    int n_orders
) {
    if (top_order_idx < 0 || top_order_idx >= n_orders) return;
    if (qtm_remaining <= 0) return;
    
    Order& passive_order = orderside[top_order_idx];
    if (passive_order.price == EMPTY_PRICE) return;
    
    // Calculate matched quantity
    int32_t matched_qty = min(qtm_remaining, passive_order.quantity);
    int32_t new_quantity = max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity to match
    qtm_remaining = max(0, qtm_remaining - passive_order.quantity);
    
    // Find empty trade slot and record trade
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
    
    // Update passive order quantity
    passive_order.quantity = new_quantity;
    
    // Clean up if quantity is zero
    if (new_quantity <= 0) {
        passive_order.price = EMPTY_PRICE;
        passive_order.order_id = 0;
        passive_order.trader_id = 0;
        passive_order.time_sec = 0;
        passive_order.time_ns = 0;
    }
}

/**
 * Match against ask orders (for incoming buy order)
 * Maps to JAX _match_against_ask_orders (JaxOrderBookArrays.py:127-130)
 * 
 * Iteratively matches against best ask until:
 * - No more quantity to match (qtm_remaining <= 0)
 * - No more matching asks (price > limit_price)
 * - No more ask orders available
 */
__device__ void match_against_asks_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid asks
    while (qtm_remaining > 0) {
        // Get best ask
        int top_ask_idx = get_top_ask_order_idx(asks, n_orders);
        
        // Check if we can match
        if (top_ask_idx == -1) break;  // No asks available
        if (asks[top_ask_idx].price == EMPTY_PRICE) break;  // No valid ask
        if (asks[top_ask_idx].price > limit_price) break;  // Price too high
        
        // Match against this ask
        match_single_order_device(
            top_ask_idx,
            asks,
            qtm_remaining,
            trades,
            n_trades,
            msg.order_id,
            msg.time_sec,
            msg.time_ns,
            n_orders
        );
    }
}

/**
 * Match against bid orders (for incoming sell order)
 * Maps to JAX _match_against_bid_orders (JaxOrderBookArrays.py:115-118)
 * 
 * Iteratively matches against best bid until:
 * - No more quantity to match (qtm_remaining <= 0)
 * - No more matching bids (price < limit_price)
 * - No more bid orders available
 */
__device__ void match_against_bids_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid bids
    while (qtm_remaining > 0) {
        // Get best bid
        int top_bid_idx = get_top_bid_order_idx(bids, n_orders);
        
        // Check if we can match
        if (top_bid_idx == -1) break;  // No bids available
        if (bids[top_bid_idx].price == EMPTY_PRICE) break;  // No valid bid
        if (bids[top_bid_idx].price < limit_price) break;  // Price too low
        
        // Match against this bid
        match_single_order_device(
            top_bid_idx,
            bids,
            qtm_remaining,
            trades,
            n_trades,
            msg.order_id,
            msg.time_sec,
            msg.time_ns,
            n_orders
        );
    }
}

// ============================================================================
// COMBINED ORDER PROCESSING
// ============================================================================

/**
 * Process a single message (add, cancel, or match)
 * Maps to JAX cond_type_side (JaxOrderBookArrays.py:181-206)
 * 
 * Dispatches to appropriate function based on message type and side
 */
__device__ void process_message_device(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades
) {
    // Determine action based on type and side
    // Type: 1=limit, 2=cancel, 3=delete, 4=market
    // Side: -1=ask, 1=bid
    
    if (msg.type == Message::CANCEL || msg.type == Message::DELETE) {
        // Cancel order
        if (msg.side == Message::ASK) {
            cancel_order_device(asks, msg, n_orders);
        } else if (msg.side == Message::BID) {
            cancel_order_device(bids, msg, n_orders);
        }
    }
    else if (msg.type == Message::LIMIT) {
        // Limit order - need to track remaining quantity after matching
        if (msg.side == Message::ASK) {
            // Sell limit: match against bids, then add remainder
            
            // Count initial bid volume at or above our price
            int32_t matchable_qty = 0;
            for (int i = 0; i < n_orders; i++) {
                if (bids[i].price != EMPTY_PRICE && bids[i].price >= msg.price) {
                    matchable_qty += bids[i].quantity;
                }
            }
            
            // Match against bids
            match_against_bids_device(asks, bids, trades, msg, n_orders, n_trades);
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = msg.quantity - matchable_qty;
            if (remaining < 0) remaining = 0;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_device(asks, remaining_msg, n_orders);
            }
        } else if (msg.side == Message::BID) {
            // Buy limit: match against asks, then add remainder
            
            // Count initial ask volume at or below our price
            int32_t matchable_qty = 0;
            for (int i = 0; i < n_orders; i++) {
                if (asks[i].price != EMPTY_PRICE && asks[i].price <= msg.price) {
                    matchable_qty += asks[i].quantity;
                }
            }
            
            // Match against asks
            match_against_asks_device(asks, bids, trades, msg, n_orders, n_trades);
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = msg.quantity - matchable_qty;
            if (remaining < 0) remaining = 0;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_device(bids, remaining_msg, n_orders);
            }
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order - aggressive matching only (no remainder added)
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            // Buy market: match against asks at any price
            match_msg.price = MAX_INT;  // Will match any ask price
            match_against_asks_device(asks, bids, trades, match_msg, n_orders, n_trades);
        } else if (msg.side == Message::ASK) {
            // Sell market: match against bids at any price
            match_msg.price = 0;  // Will match any bid price
            match_against_bids_device(asks, bids, trades, match_msg, n_orders, n_trades);
        }
    }
}

} // namespace cuda_orderbook

