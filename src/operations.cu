/**
 * Orderbook Operations - WARP LEVEL Device Functions
 * 
 * All operations redesigned for warp-level parallelism
 * Each warp (32 threads) manages one LOB
 */

#include "types.h"
#include "utils.cuh"

namespace cuda_orderbook {

constexpr int WARP_SIZE = 32;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Remove orders with zero or negative quantity
 * Lane 0 performs the cleanup
 */
__device__ void remove_zero_neg_quant_warp(Order* orderside, int n_orders, int laneId) {
    if (laneId == 0) {
        for (int i = 0; i < n_orders; i++) {
            if (orderside[i].quantity <= 0 && orderside[i].price != EMPTY_PRICE) {
                orderside[i].price = EMPTY_PRICE;
                orderside[i].quantity = 0;
                orderside[i].order_id = 0;
                orderside[i].trader_id = 0;
                orderside[i].time_sec = 0;
                orderside[i].time_ns = 0;
            }
        }
    }
}

// ============================================================================
// ADD AND CANCEL OPERATIONS
// ============================================================================

/**
 * Add order to orderside (warp-level)
 * Lane 0 performs the insertion, all lanes participate
 */
__device__ void add_order_warp(
    Order* orderside,
    const Message& msg,
    int n_orders,
    int laneId
) {
    if (laneId == 0) {
        // Find first empty slot
        int empty_idx = -1;
        for (int i = 0; i < n_orders; i++) {
            if (orderside[i].price == EMPTY_PRICE) {
                empty_idx = i;
                break;
            }
        }
        
        if (empty_idx >= 0) {
            // Add the order
            orderside[empty_idx].price = msg.price;
            orderside[empty_idx].quantity = max(0, msg.quantity);
            orderside[empty_idx].order_id = msg.order_id;
            orderside[empty_idx].trader_id = msg.trader_id;
            orderside[empty_idx].time_sec = msg.time_sec;
            orderside[empty_idx].time_ns = msg.time_ns;
            
            // Clean up zero/negative quantities
            remove_zero_neg_quant_warp(orderside, n_orders, 0);
        }
    }
}

/**
 * Cancel order from orderside (warp-level)
 * Lane 0 performs the cancellation
 */
__device__ void cancel_order_warp(
    Order* orderside,
    const Message& msg,
    int n_orders,
    int laneId
) {
    if (laneId == 0) {
        // Find by order_id
        int idx = -1;
        for (int i = 0; i < n_orders; i++) {
            if (orderside[i].order_id == msg.order_id) {
                idx = i;
                break;
            }
        }
        
        // If not found, search by price for INITID orders
        if (idx == -1) {
            for (int i = 0; i < n_orders; i++) {
                if (orderside[i].price == msg.price && 
                    orderside[i].order_id <= INITID) {
                    idx = i;
                    break;
                }
            }
        }
        
        if (idx >= 0) {
            // Reduce quantity
            orderside[idx].quantity -= msg.quantity;
            
            // Clean up
            remove_zero_neg_quant_warp(orderside, n_orders, 0);
        }
    }
}

// ============================================================================
// MATCHING ENGINE - WARP-LEVEL PARALLEL REDUCTION
// ============================================================================

/**
 * Find best ask order using warp-level reduction
 * All lanes participate - returns valid index in all lanes
 */
__device__ int find_best_ask_warp(const Order* asks, int n_orders, int laneId) {
    // Each lane finds best in its chunk
    int32_t best_price = MAX_INT;
    int32_t best_time_sec = MAX_INT;
    int32_t best_time_ns = MAX_INT;
    int best_idx = -1;

    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
        if (asks[i].price != EMPTY_PRICE) {
            bool is_better = false;
            if (asks[i].price < best_price) {
                is_better = true;
            } else if (asks[i].price == best_price) {
                if (asks[i].time_sec < best_time_sec) {
                    is_better = true;
                } else if (asks[i].time_sec == best_time_sec && asks[i].time_ns < best_time_ns) {
                    is_better = true;
                }
            }
            if (is_better) {
                best_price = asks[i].price;
                best_time_sec = asks[i].time_sec;
                best_time_ns = asks[i].time_ns;
                best_idx = i;
            }
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        int32_t other_price = __shfl_down_sync(0xFFFFFFFF, best_price, offset);
        int32_t other_time_sec = __shfl_down_sync(0xFFFFFFFF, best_time_sec, offset);
        int32_t other_time_ns = __shfl_down_sync(0xFFFFFFFF, best_time_ns, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
        
        bool is_other_better = false;
        if (other_price < best_price) {
            is_other_better = true;
        } else if (other_price == best_price) {
            if (other_time_sec < best_time_sec) {
                is_other_better = true;
            } else if (other_time_sec == best_time_sec && other_time_ns < best_time_ns) {
                is_other_better = true;
            }
        }
        
        if (is_other_better) {
            best_price = other_price;
            best_time_sec = other_time_sec;
            best_time_ns = other_time_ns;
            best_idx = other_idx;
        }
    }

    // Broadcast result from lane 0 to all lanes
    best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);
    return best_idx;
}

/**
 * Find best bid order using warp-level reduction
 * All lanes participate - returns valid index in all lanes
 */
__device__ int find_best_bid_warp(const Order* bids, int n_orders, int laneId) {
    // Each lane finds best in its chunk
    int32_t best_price = -1;
    int32_t best_time_sec = MAX_INT;
    int32_t best_time_ns = MAX_INT;
    int best_idx = -1;

    for (int i = laneId; i < n_orders; i += WARP_SIZE) {
        if (bids[i].price != EMPTY_PRICE) {
            bool is_better = false;
            if (bids[i].price > best_price) {
                is_better = true;
            } else if (bids[i].price == best_price) {
                if (bids[i].time_sec < best_time_sec) {
                    is_better = true;
                } else if (bids[i].time_sec == best_time_sec && bids[i].time_ns < best_time_ns) {
                    is_better = true;
                }
            }
            if (is_better) {
                best_price = bids[i].price;
                best_time_sec = bids[i].time_sec;
                best_time_ns = bids[i].time_ns;
                best_idx = i;
            }
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        int32_t other_price = __shfl_down_sync(0xFFFFFFFF, best_price, offset);
        int32_t other_time_sec = __shfl_down_sync(0xFFFFFFFF, best_time_sec, offset);
        int32_t other_time_ns = __shfl_down_sync(0xFFFFFFFF, best_time_ns, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
        
        bool is_other_better = false;
        if (other_price > best_price) {
            is_other_better = true;
        } else if (other_price == best_price) {
            if (other_time_sec < best_time_sec) {
                is_other_better = true;
            } else if (other_time_sec == best_time_sec && other_time_ns < best_time_ns) {
                is_other_better = true;
            }
        }
        
        if (is_other_better) {
            best_price = other_price;
            best_time_sec = other_time_sec;
            best_time_ns = other_time_ns;
            best_idx = other_idx;
        }
    }

    // Broadcast result from lane 0 to all lanes
    best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);
    return best_idx;
}

// ============================================================================
// ORDER MATCHING
// ============================================================================

/**
 * Match a single order and generate trade
 * Lane 0 performs the matching
 */
__device__ void match_single_order_warp(
    int top_order_idx,
    Order* orderside,
    int32_t& qtm_remaining,
    Trade* trades,
    int n_trades,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns,
    int n_orders,
    int laneId
) {
    if (laneId == 0) {
        if (top_order_idx < 0 || top_order_idx >= n_orders) return;
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
            passive_order.order_id = 0;
            passive_order.trader_id = 0;
            passive_order.time_sec = 0;
            passive_order.time_ns = 0;
        }
    }
}

/**
 * Match against ask orders (for incoming buy order)
 * All lanes participate in finding best, lane 0 executes match
 */
__device__ void match_against_asks_warp(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades,
    int laneId
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;

    while (true) {
        // Broadcast current remaining qty to all lanes
        qtm_remaining = __shfl_sync(0xFFFFFFFF, qtm_remaining, 0);
        if (qtm_remaining <= 0) break;

        // All lanes participate in finding best ask
        int top_ask_idx = find_best_ask_warp(asks, n_orders, laneId);

        // Lane 0 checks if we can continue
        bool can_continue = true;
        if (laneId == 0) {
            can_continue = !(top_ask_idx == -1 ||
                           asks[top_ask_idx].price == EMPTY_PRICE ||
                           asks[top_ask_idx].price > limit_price);
        }
        
        // Broadcast decision to all lanes
        can_continue = __shfl_sync(0xFFFFFFFF, can_continue ? 1 : 0, 0) != 0;
        if (!can_continue) break;
        
        // Lane 0 performs the match
        match_single_order_warp(
            top_ask_idx, asks, qtm_remaining, trades, n_trades,
            msg.order_id, msg.time_sec, msg.time_ns, n_orders, laneId
        );
    }
}

/**
 * Match against bid orders (for incoming sell order)
 * All lanes participate in finding best, lane 0 executes match
 */
__device__ void match_against_bids_warp(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades,
    int laneId
) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;

    while (true) {
        // Broadcast current remaining qty to all lanes
        qtm_remaining = __shfl_sync(0xFFFFFFFF, qtm_remaining, 0);
        if (qtm_remaining <= 0) break;

        // All lanes participate in finding best bid
        int top_bid_idx = find_best_bid_warp(bids, n_orders, laneId);

        // Lane 0 checks if we can continue
        bool can_continue = true;
        if (laneId == 0) {
            can_continue = !(top_bid_idx == -1 ||
                           bids[top_bid_idx].price == EMPTY_PRICE ||
                           bids[top_bid_idx].price < limit_price);
        }
        
        // Broadcast decision to all lanes
        can_continue = __shfl_sync(0xFFFFFFFF, can_continue ? 1 : 0, 0) != 0;
        if (!can_continue) break;
        
        // Lane 0 performs the match
        match_single_order_warp(
            top_bid_idx, bids, qtm_remaining, trades, n_trades,
            msg.order_id, msg.time_sec, msg.time_ns, n_orders, laneId
        );
    }
}

// ============================================================================
// COMBINED ORDER PROCESSING
// ============================================================================

/**
 * Process a single message (add, cancel, or match)
 * All lanes participate in warp operations
 */
__device__ void process_message_warp(
    Order* asks,
    Order* bids,
    Trade* trades,
    const Message& msg,
    int n_orders,
    int n_trades,
    int laneId
) {
    if (msg.type == Message::CANCEL || msg.type == Message::DELETE) {
        // Cancel order
        if (msg.side == Message::ASK) {
            cancel_order_warp(asks, msg, n_orders, laneId);
        } else if (msg.side == Message::BID) {
            cancel_order_warp(bids, msg, n_orders, laneId);
        }
    }
    else if (msg.type == Message::LIMIT) {
        // Limit order
        if (msg.side == Message::ASK) {
            // Sell limit: match against bids, then add remainder
            
            // Lane 0 counts matchable quantity
            int32_t matchable_qty = 0;
            if (laneId == 0) {
                for (int i = 0; i < n_orders; i++) {
                    if (bids[i].price != EMPTY_PRICE && bids[i].price >= msg.price) {
                        matchable_qty += bids[i].quantity;
                    }
                }
            }
            
            // Broadcast to all lanes
            matchable_qty = __shfl_sync(0xFFFFFFFF, matchable_qty, 0);
            
            // All lanes participate in matching
            match_against_bids_warp(asks, bids, trades, msg, n_orders, n_trades, laneId);
            
            // Lane 0 adds remainder
            if (laneId == 0) {
                int32_t remaining = msg.quantity - matchable_qty;
                if (remaining < 0) remaining = 0;
                
                if (remaining > 0) {
                    Message remaining_msg = msg;
                    remaining_msg.quantity = remaining;
                    add_order_warp(asks, remaining_msg, n_orders, 0);
                }
            }
        } else if (msg.side == Message::BID) {
            // Buy limit: match against asks, then add remainder
            
            // Lane 0 counts matchable quantity
            int32_t matchable_qty = 0;
            if (laneId == 0) {
                for (int i = 0; i < n_orders; i++) {
                    if (asks[i].price != EMPTY_PRICE && asks[i].price <= msg.price) {
                        matchable_qty += asks[i].quantity;
                    }
                }
            }
            
            // Broadcast to all lanes
            matchable_qty = __shfl_sync(0xFFFFFFFF, matchable_qty, 0);
            
            // All lanes participate in matching
            match_against_asks_warp(asks, bids, trades, msg, n_orders, n_trades, laneId);
            
            // Lane 0 adds remainder
            if (laneId == 0) {
                int32_t remaining = msg.quantity - matchable_qty;
                if (remaining < 0) remaining = 0;
                
                if (remaining > 0) {
                    Message remaining_msg = msg;
                    remaining_msg.quantity = remaining;
                    add_order_warp(bids, remaining_msg, n_orders, 0);
                }
            }
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            // Buy market: match against asks
            match_msg.price = MAX_INT;
            match_against_asks_warp(asks, bids, trades, match_msg, n_orders, n_trades, laneId);
        } else if (msg.side == Message::ASK) {
            // Sell market: match against bids
            match_msg.price = 0;
            match_against_bids_warp(asks, bids, trades, match_msg, n_orders, n_trades, laneId);
        }
    }
}

} // namespace cuda_orderbook
