/**
 * Operation Timing Benchmark
 * 
 * Measures timing for specific orderbook operations:
 * 1. ADD (non-matching inserts)
 * 2. LIMIT order insert + match
 * 3. CANCEL operations
 * 4. MARKET order insert + match
 */

#include "kernels.cuh"
#include "utils.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace cuda_orderbook;

// ============================================================================
// CONFIGURATION
// ============================================================================

const int DEFAULT_BLOCK_SIZE = 256;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct BenchmarkScenario {
    std::string name;
    std::string operation_type;
    std::vector<Message> setup_messages;    // Pre-population (not timed unless time_setup=true)
    std::vector<Message> test_messages;     // Messages to time
    int num_books;
    int n_orders;
    int n_trades;
    bool time_setup;  // If true, time setup instead of test
    
    BenchmarkScenario() : num_books(0), n_orders(0), n_trades(0), time_setup(false) {}
};

struct ScenarioResults {
    std::string scenario_name;
    std::string operation_type;
    int num_messages;
    float time_ms;
    float time_per_op_us;
    float ops_per_second;
};

// ============================================================================
// MESSAGE GENERATION FUNCTIONS
// ============================================================================

/**
 * Generate non-matching limit orders for pure ADD timing
 * Creates wide spread: bids at 9000-9900, asks at 11000-11900
 */
std::vector<Message> generate_nonematch_limits(int num_books, int msgs_per_book) {
    std::vector<Message> msgs;
    int order_id = 1000;
    
    for (int book = 0; book < num_books; book++) {
        for (int i = 0; i < msgs_per_book; i++) {
            Message msg;
            msg.type = Message::LIMIT;
            
            // Alternate between bid and ask
            if (i % 2 == 0) {
                // Low bid (won't match asks at 11000+)
                msg.side = Message::BID;
                msg.price = 9000 + (i / 2) * 10;
                msg.quantity = 100;
            } else {
                // High ask (won't match bids at 9000-9900)
                msg.side = Message::ASK;
                msg.price = 11000 + (i / 2) * 10;
                msg.quantity = 100;
            }
            
            msg.order_id = order_id++;
            msg.trader_id = book;
            msg.time_sec = i / 1000;
            msg.time_ns = (i % 1000) * 1000000;
            
            msgs.push_back(msg);
        }
    }
    return msgs;
}

/**
 * Generate matching limit orders
 * Creates orders that cross the spread and match
 */
std::vector<Message> generate_matching_limits(int num_books, int msgs_per_book) {
    std::vector<Message> msgs;
    int order_id = 50000;
    
    for (int book = 0; book < num_books; book++) {
        for (int i = 0; i < msgs_per_book; i++) {
            Message msg;
            msg.type = Message::LIMIT;
            
            if (i % 2 == 0) {
                // Buy at high price (will match asks at 10050+)
                msg.side = Message::BID;
                msg.price = 10060;  // Crosses ask spread
                msg.quantity = 50;
            } else {
                // Sell at low price (will match bids at 9950-)
                msg.side = Message::ASK;
                msg.price = 9940;   // Crosses bid spread
                msg.quantity = 50;
            }
            
            msg.order_id = order_id++;
            msg.trader_id = book;
            msg.time_sec = 100 + i / 1000;
            msg.time_ns = (i % 1000) * 1000000;
            
            msgs.push_back(msg);
        }
    }
    return msgs;
}

/**
 * Generate cancel messages for existing orders
 */
std::vector<Message> generate_cancels(const std::vector<Message>& orders_to_cancel) {
    std::vector<Message> cancel_msgs;
    
    for (const auto& order : orders_to_cancel) {
        Message cancel;
        cancel.type = Message::CANCEL;
        cancel.side = order.side;
        cancel.order_id = order.order_id;
        cancel.quantity = order.quantity;  // Cancel full quantity
        cancel.price = order.price;
        cancel.trader_id = order.trader_id;
        cancel.time_sec = order.time_sec + 1000;
        cancel.time_ns = order.time_ns;
        
        cancel_msgs.push_back(cancel);
    }
    return cancel_msgs;
}

/**
 * Generate market orders
 */
std::vector<Message> generate_market_orders(int num_books, int msgs_per_book) {
    std::vector<Message> msgs;
    int order_id = 80000;
    
    for (int book = 0; book < num_books; book++) {
        for (int i = 0; i < msgs_per_book; i++) {
            Message msg;
            msg.type = Message::MARKET;
            msg.side = (i % 2 == 0) ? Message::BID : Message::ASK;
            msg.quantity = 50;
            msg.price = 0;  // Ignored for market orders
            msg.order_id = order_id++;
            msg.trader_id = book;
            msg.time_sec = 200 + i / 1000;
            msg.time_ns = (i % 1000) * 1000000;
            
            msgs.push_back(msg);
        }
    }
    return msgs;
}

/**
 * Generate spread liquidity for scenarios 2 and 4
 * Creates orderbook with tight spread: asks at 10050+, bids at 9950-
 */
std::vector<Message> generate_spread_liquidity(int num_books) {
    std::vector<Message> msgs;
    int order_id = 100000;
    
    for (int book = 0; book < num_books; book++) {
        // Add 5 ask levels: 10050, 10060, 10070, 10080, 10090
        for (int i = 0; i < 5; i++) {
            Message ask;
            ask.type = Message::LIMIT;
            ask.side = Message::ASK;
            ask.price = 10050 + i * 10;
            ask.quantity = 100;
            ask.order_id = order_id++;
            ask.trader_id = book;
            ask.time_sec = 0;
            ask.time_ns = i * 1000000;
            msgs.push_back(ask);
        }
        
        // Add 5 bid levels: 9950, 9940, 9930, 9920, 9910
        for (int i = 0; i < 5; i++) {
            Message bid;
            bid.type = Message::LIMIT;
            bid.side = Message::BID;
            bid.price = 9950 - i * 10;
            bid.quantity = 100;
            bid.order_id = order_id++;
            bid.trader_id = book;
            bid.time_sec = 0;
            bid.time_ns = (5 + i) * 1000000;
            msgs.push_back(bid);
        }
    }
    return msgs;
}

// ============================================================================
// BENCHMARK EXECUTION
// ============================================================================

/**
 * Run a single benchmark scenario
 */
ScenarioResults run_scenario(
    const BenchmarkScenario& scenario,
    int block_size = DEFAULT_BLOCK_SIZE
) {
    std::cout << "\nRunning: " << scenario.name << "..." << std::endl;
    
    // Allocate GPU batch
    OrderbookBatch gpu_batch;
    if (!allocate_orderbook_batch(gpu_batch, scenario.num_books, 
                                   scenario.n_orders, scenario.n_trades)) {
        std::cerr << "Failed to allocate GPU batch" << std::endl;
        return ScenarioResults();
    }
    
    if (!allocate_host_orderbook_batch(gpu_batch, scenario.num_books,
                                        scenario.n_orders, scenario.n_trades)) {
        std::cerr << "Failed to allocate host batch" << std::endl;
        free_orderbook_batch(gpu_batch);
        return ScenarioResults();
    }
    
    // Calculate shared memory
    size_t shared_mem_size = block_size * (sizeof(int32_t) * 3 + sizeof(int));
    
    // Initialize
    init_orderbooks_device(gpu_batch);
    
    // Setup phase (if not timing setup)
    if (!scenario.time_setup && !scenario.setup_messages.empty()) {
        std::cout << "  Setting up orderbook state..." << std::endl;
        Message* d_setup_msgs;
        size_t setup_size = scenario.setup_messages.size() * sizeof(Message);
        CHECK_CUDA_ERROR(cudaMalloc(&d_setup_msgs, setup_size));
        CHECK_CUDA_ERROR(cudaMemcpy(d_setup_msgs, scenario.setup_messages.data(), 
                                     setup_size, cudaMemcpyHostToDevice));
        
        int setup_msgs_per_book = scenario.setup_messages.size() / scenario.num_books;
        process_messages_sequential_kernel<<<scenario.num_books, block_size, shared_mem_size>>>(
            gpu_batch, d_setup_msgs, setup_msgs_per_book, scenario.num_books
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_setup_msgs));
    }
    
    // Determine which messages to time
    const auto& messages_to_time = scenario.time_setup ? 
                                   scenario.setup_messages : 
                                   scenario.test_messages;
    
    if (messages_to_time.empty()) {
        std::cerr << "No messages to benchmark!" << std::endl;
        free_orderbook_batch(gpu_batch);
        free_host_orderbook_batch(gpu_batch);
        return ScenarioResults();
    }
    
    // Allocate and copy test messages
    Message* d_test_msgs;
    size_t test_size = messages_to_time.size() * sizeof(Message);
    CHECK_CUDA_ERROR(cudaMalloc(&d_test_msgs, test_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_test_msgs, messages_to_time.data(), 
                                 test_size, cudaMemcpyHostToDevice));
    
    int test_msgs_per_book = messages_to_time.size() / scenario.num_books;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Warm-up run
    std::cout << "  Warm-up run..." << std::endl;
    process_messages_sequential_kernel<<<scenario.num_books, block_size, shared_mem_size>>>(
        gpu_batch, d_test_msgs, test_msgs_per_book, scenario.num_books
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Reset if we're re-running on same data
    if (scenario.time_setup) {
        init_orderbooks_device(gpu_batch);
    }
    
    // Timed run
    std::cout << "  Timed run..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    process_messages_sequential_kernel<<<scenario.num_books, block_size, shared_mem_size>>>(
        gpu_batch, d_test_msgs, test_msgs_per_book, scenario.num_books
    );
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    
    // Build results
    ScenarioResults results;
    results.scenario_name = scenario.name;
    results.operation_type = scenario.operation_type;
    results.num_messages = messages_to_time.size();
    results.time_ms = time_ms;
    results.time_per_op_us = (time_ms * 1000.0f) / results.num_messages;
    results.ops_per_second = results.num_messages / (time_ms / 1000.0f);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_test_msgs));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free_orderbook_batch(gpu_batch);
    free_host_orderbook_batch(gpu_batch);
    
    std::cout << "  ✓ Complete" << std::endl;
    
    return results;
}

// ============================================================================
// RESULTS PRINTING
// ============================================================================

void print_results(const ScenarioResults& results) {
    std::cout << "\n=== " << results.scenario_name << " ===" << std::endl;
    std::cout << "Operation Type:     " << results.operation_type << std::endl;
    std::cout << "Messages Processed: " << results.num_messages << std::endl;
    std::cout << "Total Time:         " << std::fixed << std::setprecision(3) 
              << results.time_ms << " ms" << std::endl;
    std::cout << "Time per Operation: " << std::fixed << std::setprecision(3)
              << results.time_per_op_us << " μs" << std::endl;
    std::cout << "Throughput:         " << std::fixed << std::setprecision(0)
              << results.ops_per_second << " ops/sec" << std::endl;
}

void print_comparison(const std::vector<ScenarioResults>& all_results) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              PERFORMANCE COMPARISON SUMMARY                        ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\n" << std::left << std::setw(32) << "Operation" 
              << std::right << std::setw(12) << "Time (ms)" 
              << std::setw(12) << "μs/op" 
              << std::setw(15) << "ops/sec" << std::endl;
    std::cout << std::string(71, '-') << std::endl;
    
    for (const auto& r : all_results) {
        std::cout << std::left << std::setw(32) << r.scenario_name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.time_ms
                  << std::setw(12) << r.time_per_op_us
                  << std::setw(15) << std::setprecision(0) << r.ops_per_second
                  << std::endl;
    }
    
    std::cout << "\n=== Relative Performance (normalized to fastest) ===" << std::endl;
    
    // Find fastest operation (lowest time)
    float min_time = all_results[0].time_ms;
    for (const auto& r : all_results) {
        min_time = std::min(min_time, r.time_ms);
    }
    
    for (const auto& r : all_results) {
        float relative = r.time_ms / min_time;
        std::cout << std::left << std::setw(32) << r.scenario_name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << relative << "x";
        
        if (relative == 1.0f) {
            std::cout << " ← FASTEST";
        } else if (relative > 2.0f) {
            std::cout << " (slower)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Key Insights ===" << std::endl;
    std::cout << "• Lower μs/op = faster operation" << std::endl;
    std::cout << "• Higher ops/sec = better throughput" << std::endl;
    std::cout << "• Relative performance shows operation cost ratios" << std::endl;
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           ORDERBOOK OPERATION TIMING BENCHMARK                     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "\nUsage: " << argv[0] << " [num_books] [msgs_per_book]" << std::endl;
        std::cout << "\nParameters:" << std::endl;
        std::cout << "  num_books      : Number of orderbooks (default: 100)" << std::endl;
        std::cout << "  msgs_per_book  : Messages per orderbook (default: 1000)" << std::endl;
        std::cout << "\nBenchmark Scenarios:" << std::endl;
        std::cout << "  1. ADD        - Non-matching limit order inserts" << std::endl;
        std::cout << "  2. MATCH      - Limit orders that cross spread" << std::endl;
        std::cout << "  3. CANCEL     - Order cancellation operations" << std::endl;
        std::cout << "  4. MARKET     - Market order execution" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << "              # Default settings" << std::endl;
        std::cout << "  " << argv[0] << " 1000 10000   # 1000 books, 10k msgs each" << std::endl;
        return 0;
    }
    
    // Configuration
    int num_books = (argc > 1) ? std::atoi(argv[1]) : 100;
    int msgs_per_book = (argc > 2) ? std::atoi(argv[2]) : 1000;
    int n_orders = 1000;
    int n_trades = 2000;
    int block_size = DEFAULT_BLOCK_SIZE;
    
    std::cout << "\n📊 Configuration:" << std::endl;
    std::cout << "  Orderbooks:        " << num_books << std::endl;
    std::cout << "  Messages per book: " << msgs_per_book << std::endl;
    std::cout << "  Total messages:    " << (num_books * msgs_per_book) << std::endl;
    std::cout << "  Orders per side:   " << n_orders << std::endl;
    std::cout << "  Max trades:        " << n_trades << std::endl;
    std::cout << "  GPU block size:    " << block_size << " threads" << std::endl;
    
    std::vector<ScenarioResults> all_results;
    
    // ========================================================================
    // SCENARIO 1: Pure ADD (non-matching inserts)
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SCENARIO 1: ADD Operations (Non-Matching Inserts)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    BenchmarkScenario scenario1;
    scenario1.name = "ADD Operations";
    scenario1.operation_type = "ADD";
    scenario1.num_books = num_books;
    scenario1.n_orders = n_orders;
    scenario1.n_trades = n_trades;
    scenario1.setup_messages = generate_nonematch_limits(num_books, msgs_per_book);
    scenario1.time_setup = true;  // TIME the setup (the adds)
    
    auto results1 = run_scenario(scenario1, block_size);
    print_results(results1);
    all_results.push_back(results1);
    
    // Save for reuse in other scenarios
    auto add_messages = scenario1.setup_messages;
    
    // ========================================================================
    // SCENARIO 2: LIMIT insert + match
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SCENARIO 2: LIMIT Order Insert + Match" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    BenchmarkScenario scenario2;
    scenario2.name = "LIMIT Order Insert+Match";
    scenario2.operation_type = "MATCH";
    scenario2.num_books = num_books;
    scenario2.n_orders = n_orders;
    scenario2.n_trades = n_trades;
    scenario2.setup_messages = generate_spread_liquidity(num_books);  // Not timed
    scenario2.test_messages = generate_matching_limits(num_books, msgs_per_book);  // Timed
    scenario2.time_setup = false;
    
    auto results2 = run_scenario(scenario2, block_size);
    print_results(results2);
    all_results.push_back(results2);
    
    // ========================================================================
    // SCENARIO 3: CANCEL operations
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SCENARIO 3: CANCEL Operations" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    BenchmarkScenario scenario3;
    scenario3.name = "CANCEL Operations";
    scenario3.operation_type = "CANCEL";
    scenario3.num_books = num_books;
    scenario3.n_orders = n_orders;
    scenario3.n_trades = n_trades;
    scenario3.setup_messages = add_messages;  // Reuse from Scenario 1 (not timed)
    scenario3.test_messages = generate_cancels(add_messages);  // Timed
    scenario3.time_setup = false;
    
    auto results3 = run_scenario(scenario3, block_size);
    print_results(results3);
    all_results.push_back(results3);
    
    // ========================================================================
    // SCENARIO 4: MARKET orders
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SCENARIO 4: MARKET Order Insert + Match" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    BenchmarkScenario scenario4;
    scenario4.name = "MARKET Order Insert+Match";
    scenario4.operation_type = "MARKET";
    scenario4.num_books = num_books;
    scenario4.n_orders = n_orders;
    scenario4.n_trades = n_trades;
    scenario4.setup_messages = generate_spread_liquidity(num_books);  // Not timed
    scenario4.test_messages = generate_market_orders(num_books, msgs_per_book);  // Timed
    scenario4.time_setup = false;
    
    auto results4 = run_scenario(scenario4, block_size);
    print_results(results4);
    all_results.push_back(results4);
    
    // ========================================================================
    // Comparison summary
    // ========================================================================
    print_comparison(all_results);
    
    std::cout << "\n✓ Benchmark Complete!" << std::endl;
    
    return 0;
}

