#include "data_generator.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace cuda_orderbook {

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

Message create_message(
    int32_t type,
    int32_t side,
    int32_t quantity,
    int32_t price,
    int32_t order_id,
    int32_t trader_id,
    int32_t time_sec,
    int32_t time_ns
) {
    Message msg;
    msg.type = type;
    msg.side = side;
    msg.quantity = quantity;
    msg.price = price;
    msg.order_id = order_id;
    msg.trader_id = trader_id;
    msg.time_sec = time_sec;
    msg.time_ns = time_ns;
    return msg;
}

void print_messages(const std::vector<Message>& messages, const char* title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << "Total messages: " << messages.size() << std::endl;
    
    for (size_t i = 0; i < messages.size() && i < 20; i++) {
        const auto& msg = messages[i];
        
        std::cout << "[" << i << "] ";
        
        // Type
        if (msg.type == Message::LIMIT) std::cout << "LIMIT  ";
        else if (msg.type == Message::CANCEL) std::cout << "CANCEL ";
        else if (msg.type == Message::MARKET) std::cout << "MARKET ";
        
        // Side
        if (msg.side == Message::BID) std::cout << "BUY  ";
        else if (msg.side == Message::ASK) std::cout << "SELL ";
        
        std::cout << "qty=" << std::setw(4) << msg.quantity;
        
        if (msg.type != Message::CANCEL) {
            std::cout << " @ " << std::fixed << std::setprecision(2) 
                      << (msg.price / 1000.0);
        }
        
        std::cout << " (id=" << msg.order_id << ")" << std::endl;
    }
    
    if (messages.size() > 20) {
        std::cout << "... (" << (messages.size() - 20) << " more)" << std::endl;
    }
}

// ============================================================================
// SIMPLE TEST SCENARIOS
// ============================================================================

std::vector<Message> generate_perfect_match() {
    std::vector<Message> messages;
    
    // Add SELL 100 @ $101.00
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 100, 101000, 1001, 10, 34200, 0
    ));
    
    // Add BUY 100 @ $101.00 (should match completely)
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 100, 101000, 2001, 20, 34200, 1000000
    ));
    
    return messages;
}

std::vector<Message> generate_partial_fill() {
    std::vector<Message> messages;
    
    // Add SELL 200 @ $101.00
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 200, 101000, 1001, 10, 34200, 0
    ));
    
    // Add BUY 100 @ $101.00 (should match 100, leave 100 remaining)
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 100, 101000, 2001, 20, 34200, 1000000
    ));
    
    return messages;
}

std::vector<Message> generate_no_match() {
    std::vector<Message> messages;
    
    // Add SELL 100 @ $102.00
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 100, 102000, 1001, 10, 34200, 0
    ));
    
    // Add BUY 100 @ $100.00 (should NOT match, spread exists)
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 100, 100000, 2001, 20, 34200, 1000000
    ));
    
    return messages;
}

std::vector<Message> generate_price_improvement() {
    std::vector<Message> messages;
    
    // Build orderbook
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 50, 101000, 1001, 10, 34200, 0
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 30, 102000, 1002, 11, 34200, 100000
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 100, 99000, 2001, 20, 34201, 0
    ));
    
    // Add BUY @ $101.50 (crosses spread, should match @ $101.00)
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 60, 101500, 2002, 21, 34202, 0
    ));
    
    return messages;
}

std::vector<Message> generate_cancel_test() {
    std::vector<Message> messages;
    
    // Add SELL 100 @ $101.00
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 100, 101000, 1001, 10, 34200, 0
    ));
    
    // Add BUY 50 @ $99.00
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 50, 99000, 2001, 20, 34200, 1000000
    ));
    
    // Cancel 30 units from order 1001
    messages.push_back(create_message(
        Message::CANCEL, Message::ASK, 30, 101000, 1001, 10, 34201, 0
    ));
    
    return messages;
}

std::vector<Message> generate_market_order() {
    std::vector<Message> messages;
    
    // Build orderbook with multiple levels
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 50, 101000, 1001, 10, 34200, 0
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 30, 102000, 1002, 11, 34200, 100000
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 20, 103000, 1003, 12, 34200, 200000
    ));
    
    // Market BUY 80 units (should consume first two levels)
    messages.push_back(create_message(
        Message::MARKET, Message::BID, 80, 0, 2001, 20, 34201, 0
    ));
    
    return messages;
}

std::vector<Message> generate_price_time_priority() {
    std::vector<Message> messages;
    
    // Add three orders at same price, different times
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 50, 101000, 1001, 10, 34200, 0
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 30, 101000, 1002, 11, 34200, 100000
    ));
    messages.push_back(create_message(
        Message::LIMIT, Message::ASK, 40, 101000, 1003, 12, 34200, 200000
    ));
    
    // Buy 60 units (should match first order fully, then 10 from second)
    messages.push_back(create_message(
        Message::LIMIT, Message::BID, 60, 101000, 2001, 20, 34201, 0
    ));
    
    return messages;
}

std::vector<Message> generate_multilevel_book() {
    std::vector<Message> messages;
    
    // Build 5-level orderbook
    for (int i = 0; i < 5; i++) {
        // Asks: 101.00, 101.10, 101.20, ...
        messages.push_back(create_message(
            Message::LIMIT, Message::ASK, 100, 
            101000 + i * 100, 1001 + i, 10 + i, 
            34200, i * 100000
        ));
        
        // Bids: 99.90, 99.80, 99.70, ...
        messages.push_back(create_message(
            Message::LIMIT, Message::BID, 100,
            99900 - i * 100, 2001 + i, 20 + i,
            34200, i * 100000 + 50000
        ));
    }
    
    return messages;
}

// ============================================================================
// RANDOM DATA GENERATION
// ============================================================================

std::vector<Message> generate_random_messages(
    int num_messages,
    const DataGenConfig& config
) {
    std::vector<Message> messages;
    std::mt19937 rng(config.seed);
    
    // Track active orders for cancellation
    std::vector<int32_t> active_orders;
    int32_t next_order_id = 1000;
    
    for (int i = 0; i < num_messages; i++) {
        Message msg;
        
        // Determine message type
        float type_rand = static_cast<float>(rng()) / rng.max();
        
        if (type_rand < config.limit_order_pct) {
            // LIMIT ORDER
            msg.type = Message::LIMIT;
            msg.side = (rng() % 2) ? Message::BID : Message::ASK;
            msg.quantity = config.min_quantity + 
                          (rng() % (config.max_quantity - config.min_quantity + 1));
            
            // Generate price around mid with spread
            int32_t price_offset = rng() % config.price_range;
            if (msg.side == Message::BID) {
                msg.price = config.mid_price - (config.price_range / 2) - 
                           (price_offset % (config.price_range / 2));
            } else {
                msg.price = config.mid_price + (config.price_range / 2) + 
                           (price_offset % (config.price_range / 2));
            }
            
            // Round to tick size
            msg.price = (msg.price / config.tick_size) * config.tick_size;
            
            msg.order_id = next_order_id++;
            active_orders.push_back(msg.order_id);
            
        } else if (type_rand < config.limit_order_pct + config.cancel_pct) {
            // CANCEL ORDER
            if (active_orders.empty()) {
                // No orders to cancel, make it a limit order instead
                continue;
            }
            
            msg.type = Message::CANCEL;
            size_t idx = rng() % active_orders.size();
            msg.order_id = active_orders[idx];
            msg.side = (msg.order_id < 2000) ? Message::ASK : Message::BID;
            msg.quantity = config.min_quantity + 
                          (rng() % (config.max_quantity - config.min_quantity + 1));
            msg.price = config.mid_price; // Not used for cancel
            
        } else {
            // MARKET ORDER
            msg.type = Message::MARKET;
            msg.side = (rng() % 2) ? Message::BID : Message::ASK;
            msg.quantity = config.min_quantity + 
                          (rng() % (config.max_quantity - config.min_quantity + 1));
            msg.price = 0; // Not used for market orders
            msg.order_id = next_order_id++;
        }
        
        msg.trader_id = (rng() % 100) + 1;
        msg.time_sec = config.start_time_sec + (i / 1000);
        msg.time_ns = (i % 1000) * config.time_increment_ns;
        
        messages.push_back(msg);
    }
    
    return messages;
}

std::vector<Message> generate_random_batch(
    int num_books,
    int messages_per_book,
    const DataGenConfig& config
) {
    std::vector<Message> all_messages;
    
    // Generate messages for each book with different seed
    for (int book = 0; book < num_books; book++) {
        DataGenConfig book_config = config;
        book_config.seed = config.seed + book;
        
        auto book_messages = generate_random_messages(messages_per_book, book_config);
        all_messages.insert(all_messages.end(), book_messages.begin(), book_messages.end());
    }
    
    return all_messages;
}

} // namespace cuda_orderbook

