/**
 * CPU Sequential Orderbook Implementation
 * 
 * Pure CPU implementation for benchmarking against GPU version
 * Provides baseline performance measurements
 */

#include "orderbook_cpu.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cuda_orderbook {

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

bool OrderbookCPU::allocate(int n_orders, int n_trades_max) {
    n_orders_per_side = n_orders;
    n_trades = n_trades_max;
    
    asks = new Order[n_orders];
    bids = new Order[n_orders];
    trades = new Trade[n_trades_max];
    
    if (!asks || !bids || !trades) {
        cleanup();
        return false;
    }
    
    initialize();
    return true;
}

void OrderbookCPU::cleanup() {
    if (asks) {
        delete[] asks;
        asks = nullptr;
    }
    if (bids) {
        delete[] bids;
        bids = nullptr;
    }
    if (trades) {
        delete[] trades;
        trades = nullptr;
    }
}

void OrderbookCPU::initialize() {
    // Initialize asks
    for (int i = 0; i < n_orders_per_side; i++) {
        asks[i].price = EMPTY_PRICE;
        asks[i].quantity = 0;
        asks[i].order_id = 0;
        asks[i].trader_id = 0;
        asks[i].time_sec = 0;
        asks[i].time_ns = 0;
    }
    
    // Initialize bids
    for (int i = 0; i < n_orders_per_side; i++) {
        bids[i].price = EMPTY_PRICE;
        bids[i].quantity = 0;
        bids[i].order_id = 0;
        bids[i].trader_id = 0;
        bids[i].time_sec = 0;
        bids[i].time_ns = 0;
    }
    
    // Initialize trades
    for (int i = 0; i < n_trades; i++) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
}

bool OrderbookBatchCPU::allocate(int n_books, int n_orders_per_book, int n_trades_per_book) {
    num_books = n_books;
    books = new OrderbookCPU[n_books];
    
    if (!books) {
        return false;
    }
    
    // Allocate each orderbook
    for (int i = 0; i < n_books; i++) {
        if (!books[i].allocate(n_orders_per_book, n_trades_per_book)) {
            cleanup();
            return false;
        }
    }
    
    return true;
}

void OrderbookBatchCPU::cleanup() {
    if (books) {
        delete[] books;
        books = nullptr;
    }
}

void OrderbookBatchCPU::initialize() {
    for (int i = 0; i < num_books; i++) {
        books[i].initialize();
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void remove_zero_neg_quant_cpu(Order* orderside, int n_orders) {
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

int get_top_ask_order_idx_cpu(const Order* asks, int n_orders) {
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

int get_top_bid_order_idx_cpu(const Order* bids, int n_orders) {
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
// BASIC OPERATIONS
// ============================================================================

void add_order_cpu(Order* orderside, const Message& msg, int n_orders) {
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
        return;
    }
    
    // Add the order
    orderside[empty_idx].price = msg.price;
    orderside[empty_idx].quantity = std::max(0, msg.quantity);
    orderside[empty_idx].order_id = msg.order_id;
    orderside[empty_idx].trader_id = msg.trader_id;
    orderside[empty_idx].time_sec = msg.time_sec;
    orderside[empty_idx].time_ns = msg.time_ns;
    
    // Clean up any orders with zero/negative quantity
    remove_zero_neg_quant_cpu(orderside, n_orders);
}

void cancel_order_cpu(Order* orderside, const Message& msg, int n_orders) {
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
    remove_zero_neg_quant_cpu(orderside, n_orders);
}

void match_single_order_cpu(
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
    int32_t matched_qty = std::min(qtm_remaining, passive_order.quantity);
    int32_t new_quantity = std::max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity to match
    qtm_remaining = std::max(0, qtm_remaining - passive_order.quantity);
    
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

void match_against_asks_cpu(
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
        int top_ask_idx = get_top_ask_order_idx_cpu(asks, n_orders);
        
        // Check if we can match
        if (top_ask_idx == -1) break;  // No asks available
        if (asks[top_ask_idx].price == EMPTY_PRICE) break;  // No valid ask
        if (asks[top_ask_idx].price > limit_price) break;  // Price too high
        
        // Match against this ask
        match_single_order_cpu(
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

void match_against_bids_cpu(
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
        int top_bid_idx = get_top_bid_order_idx_cpu(bids, n_orders);
        
        // Check if we can match
        if (top_bid_idx == -1) break;  // No bids available
        if (bids[top_bid_idx].price == EMPTY_PRICE) break;  // No valid bid
        if (bids[top_bid_idx].price < limit_price) break;  // Price too low
        
        // Match against this bid
        match_single_order_cpu(
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

void process_message_cpu(
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
            cancel_order_cpu(asks, msg, n_orders);
        } else if (msg.side == Message::BID) {
            cancel_order_cpu(bids, msg, n_orders);
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
            match_against_bids_cpu(asks, bids, trades, msg, n_orders, n_trades);
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = msg.quantity - matchable_qty;
            if (remaining < 0) remaining = 0;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_cpu(asks, remaining_msg, n_orders);
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
            match_against_asks_cpu(asks, bids, trades, msg, n_orders, n_trades);
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = msg.quantity - matchable_qty;
            if (remaining < 0) remaining = 0;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_cpu(bids, remaining_msg, n_orders);
            }
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order - aggressive matching only (no remainder added)
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            // Buy market: match against asks at any price
            match_msg.price = MAX_INT;  // Will match any ask price
            match_against_asks_cpu(asks, bids, trades, match_msg, n_orders, n_trades);
        } else if (msg.side == Message::ASK) {
            // Sell market: match against bids at any price
            match_msg.price = 0;  // Will match any bid price
            match_against_bids_cpu(asks, bids, trades, match_msg, n_orders, n_trades);
        }
    }
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

void process_messages_sequential_cpu(
    OrderbookCPU& book,
    const Message* messages,
    int num_messages
) {
    for (int i = 0; i < num_messages; i++) {
        const Message& msg = messages[i];
        
        // Skip empty/invalid messages
        if (msg.quantity <= 0 || msg.type == 0) continue;
        
        // Process this message
        process_message_cpu(
            book.asks,
            book.bids,
            book.trades,
            msg,
            book.n_orders_per_side,
            book.n_trades
        );
    }
}

void process_messages_batch_cpu(
    OrderbookBatchCPU& batch,
    const Message* messages,
    int num_messages_per_book
) {
    // Process each orderbook sequentially
    for (int book_idx = 0; book_idx < batch.num_books; book_idx++) {
        const Message* book_messages = messages + (book_idx * num_messages_per_book);
        process_messages_sequential_cpu(batch.books[book_idx], book_messages, num_messages_per_book);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void copy_orderbook_cpu(const OrderbookCPU& src, OrderbookCPU& dst) {
    if (src.n_orders_per_side != dst.n_orders_per_side || 
        src.n_trades != dst.n_trades) {
        std::cerr << "Error: Orderbook sizes don't match for copy" << std::endl;
        return;
    }
    
    std::memcpy(dst.asks, src.asks, src.n_orders_per_side * sizeof(Order));
    std::memcpy(dst.bids, src.bids, src.n_orders_per_side * sizeof(Order));
    std::memcpy(dst.trades, src.trades, src.n_trades * sizeof(Trade));
}

bool compare_orderbooks_cpu(const OrderbookCPU& book1, const OrderbookCPU& book2) {
    if (book1.n_orders_per_side != book2.n_orders_per_side ||
        book1.n_trades != book2.n_trades) {
        return false;
    }
    
    // Compare asks
    for (int i = 0; i < book1.n_orders_per_side; i++) {
        if (std::memcmp(&book1.asks[i], &book2.asks[i], sizeof(Order)) != 0) {
            return false;
        }
    }
    
    // Compare bids
    for (int i = 0; i < book1.n_orders_per_side; i++) {
        if (std::memcmp(&book1.bids[i], &book2.bids[i], sizeof(Order)) != 0) {
            return false;
        }
    }
    
    // Compare trades
    for (int i = 0; i < book1.n_trades; i++) {
        if (std::memcmp(&book1.trades[i], &book2.trades[i], sizeof(Trade)) != 0) {
            return false;
        }
    }
    
    return true;
}

void print_orderbook_cpu(const OrderbookCPU& book, int max_orders) {
    std::cout << "\n=== Orderbook State ===" << std::endl;
    
    std::cout << "\nAsks (top " << max_orders << "):" << std::endl;
    int ask_count = 0;
    for (int i = 0; i < book.n_orders_per_side && ask_count < max_orders; i++) {
        if (book.asks[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.asks[i].price
                     << ", Qty: " << book.asks[i].quantity
                     << ", ID: " << book.asks[i].order_id << std::endl;
            ask_count++;
        }
    }
    
    std::cout << "\nBids (top " << max_orders << "):" << std::endl;
    int bid_count = 0;
    for (int i = 0; i < book.n_orders_per_side && bid_count < max_orders; i++) {
        if (book.bids[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.bids[i].price
                     << ", Qty: " << book.bids[i].quantity
                     << ", ID: " << book.bids[i].order_id << std::endl;
            bid_count++;
        }
    }
    
    std::cout << "\nTrades (top " << max_orders << "):" << std::endl;
    int trade_count = 0;
    for (int i = 0; i < book.n_trades && trade_count < max_orders; i++) {
        if (book.trades[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.trades[i].price
                     << ", Qty: " << book.trades[i].quantity
                     << ", Passive ID: " << book.trades[i].passive_order_id
                     << ", Aggressive ID: " << book.trades[i].aggressive_order_id << std::endl;
            trade_count++;
        }
    }
    
    std::cout << "======================" << std::endl;
}

} // namespace cuda_orderbook

