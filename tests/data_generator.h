#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include "types.h"
#include <vector>
#include <random>

namespace cuda_orderbook {

// ============================================================================
// CONFIGURATION
// ============================================================================

struct DataGenConfig {
    // Message distribution
    float limit_order_pct = 0.70f;   // 70% limit orders
    float cancel_pct = 0.20f;         // 20% cancels
    float market_pct = 0.10f;         // 10% market orders
    
    // Price parameters
    int32_t mid_price = 100000;       // $100.00 (in cents)
    int32_t price_range = 5000;       // ±$50.00 range
    int32_t tick_size = 100;          // $0.10 minimum price increment
    
    // Quantity parameters
    int32_t min_quantity = 10;
    int32_t max_quantity = 100;
    
    // Timing
    int32_t start_time_sec = 34200;   // 9:30 AM
    int32_t time_increment_ns = 1000000; // 1ms between messages
    
    // Randomness
    unsigned int seed = 42;           // For reproducibility
};

// ============================================================================
// SIMPLE TEST SCENARIOS
// ============================================================================

// Scenario 1: Perfect match (100% fill)
std::vector<Message> generate_perfect_match();

// Scenario 2: Partial fill
std::vector<Message> generate_partial_fill();

// Scenario 3: No match (spread exists)
std::vector<Message> generate_no_match();

// Scenario 4: Price improvement (crosses spread)
std::vector<Message> generate_price_improvement();

// Scenario 5: Cancel order
std::vector<Message> generate_cancel_test();

// Scenario 6: Market order through multiple levels
std::vector<Message> generate_market_order();

// Scenario 7: Price-time priority test
std::vector<Message> generate_price_time_priority();

// Scenario 8: Build orderbook with multiple levels
std::vector<Message> generate_multilevel_book();

// ============================================================================
// RANDOM DATA GENERATION
// ============================================================================

// Generate random messages with given configuration
std::vector<Message> generate_random_messages(
    int num_messages,
    const DataGenConfig& config = DataGenConfig()
);

// Generate random messages for multiple orderbooks
std::vector<Message> generate_random_batch(
    int num_books,
    int messages_per_book,
    const DataGenConfig& config = DataGenConfig()
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Create a single message
Message create_message(
    int32_t type,
    int32_t side,
    int32_t quantity,
    int32_t price,
    int32_t order_id,
    int32_t trader_id = 1,
    int32_t time_sec = 34200,
    int32_t time_ns = 0
);

// Print messages for debugging
void print_messages(const std::vector<Message>& messages, const char* title = "Messages");

} // namespace cuda_orderbook

#endif // DATA_GENERATOR_H

