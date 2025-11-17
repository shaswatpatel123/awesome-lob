/**
 * Comprehensive Test Suite for CPU vs GPU Orderbook
 * 
 * Tests are organized incrementally:
 * 1. Unit Tests - Individual operations
 * 2. Integration Tests - Scenarios with known outcomes
 * 3. Functional Tests - CPU vs GPU comparison
 * 4. Performance Tests - Benchmarking
 */

#include "types.h"
#include "kernels.cuh"
#include "orderbook_cpu.h"
#include "data_generator.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>

using namespace cuda_orderbook;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// TEST UTILITIES
// ============================================================================

struct TestStats {
    int tests_run = 0;
    int tests_passed = 0;
    int tests_failed = 0;
    
    void record_pass() { tests_run++; tests_passed++; }
    void record_fail() { tests_run++; tests_failed++; }
    
    void print_summary() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total tests: " << tests_run << std::endl;
        std::cout << "✓ Passed: " << tests_passed << std::endl;
        std::cout << "✗ Failed: " << tests_failed << std::endl;
        
        if (tests_failed == 0) {
            std::cout << "\n🎉 ALL TESTS PASSED!" << std::endl;
        } else {
            std::cout << "\n❌ SOME TESTS FAILED" << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;
    }
};

void print_test_header(const char* test_name) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "TEST: " << test_name << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

bool compare_orders(const Order* orders1, const Order* orders2, int n_orders, const char* name) {
    for (int i = 0; i < n_orders; i++) {
        if (std::memcmp(&orders1[i], &orders2[i], sizeof(Order)) != 0) {
            std::cout << "  ✗ " << name << " mismatch at index " << i << std::endl;
            std::cout << "    CPU: price=" << orders1[i].price 
                      << " qty=" << orders1[i].quantity 
                      << " id=" << orders1[i].order_id << std::endl;
            std::cout << "    GPU: price=" << orders2[i].price 
                      << " qty=" << orders2[i].quantity 
                      << " id=" << orders2[i].order_id << std::endl;
            return false;
        }
    }
    return true;
}

bool compare_trades(const Trade* trades1, const Trade* trades2, int n_trades) {
    for (int i = 0; i < n_trades; i++) {
        if (std::memcmp(&trades1[i], &trades2[i], sizeof(Trade)) != 0) {
            std::cout << "  ✗ Trades mismatch at index " << i << std::endl;
            std::cout << "    CPU: price=" << trades1[i].price 
                      << " qty=" << trades1[i].quantity << std::endl;
            std::cout << "    GPU: price=" << trades2[i].price 
                      << " qty=" << trades2[i].quantity << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// LEVEL 1: UNIT TESTS - Individual Operations
// ============================================================================

bool unit_test_add_order(TestStats& stats) {
    print_test_header("Unit Test: Add Order");
    
    // Setup
    OrderbookCPU cpu_book;
    cpu_book.allocate(100, 50);
    
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = 1;
    gpu_batch.n_orders_per_book = 100;
    gpu_batch.n_trades_per_book = 50;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, 50 * sizeof(Trade)));
    
    init_orderbooks_kernel<<<1, 256>>>(gpu_batch, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create single add message
    Message msg = create_message(Message::LIMIT, Message::BID, 100, 99000, 2001);
    
    // Process on CPU
    process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades, 
                       msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    
    // Process on GPU
    Message* d_msg;
    CUDA_CHECK(cudaMalloc(&d_msg, sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_msg, &msg, sizeof(Message), cudaMemcpyHostToDevice));
    
    process_messages_sequential_kernel<<<1, 256>>>(gpu_batch, d_msg, 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy GPU results
    Order* gpu_bids = new Order[100];
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, 100 * sizeof(Order), 
                         cudaMemcpyDeviceToHost));
    
    // Compare
    bool match = compare_orders(cpu_book.bids, gpu_bids, 100, "Bids");
    
    // Verify order was added
    bool order_found = false;
    for (int i = 0; i < 100; i++) {
        if (cpu_book.bids[i].order_id == 2001 && 
            cpu_book.bids[i].price == 99000 &&
            cpu_book.bids[i].quantity == 100) {
            order_found = true;
            break;
        }
    }
    
    // Cleanup
    delete[] gpu_bids;
    cudaFree(gpu_batch.d_asks);
    cudaFree(gpu_batch.d_bids);
    cudaFree(gpu_batch.d_trades);
    cudaFree(d_msg);
    cpu_book.cleanup();
    
    if (match && order_found) {
        std::cout << "  ✓ PASS: Order added correctly, CPU == GPU" << std::endl;
        stats.record_pass();
        return true;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
        stats.record_fail();
        return false;
    }
}

bool unit_test_cancel_order(TestStats& stats) {
    print_test_header("Unit Test: Cancel Order");
    
    // Setup
    OrderbookCPU cpu_book;
    cpu_book.allocate(100, 50);
    
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = 1;
    gpu_batch.n_orders_per_book = 100;
    gpu_batch.n_trades_per_book = 50;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, 50 * sizeof(Trade)));
    
    init_orderbooks_kernel<<<1, 256>>>(gpu_batch, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Add then cancel
    std::vector<Message> messages;
    messages.push_back(create_message(Message::LIMIT, Message::BID, 100, 99000, 2001));
    messages.push_back(create_message(Message::CANCEL, Message::BID, 50, 99000, 2001));
    
    // Process on CPU
    for (const auto& msg : messages) {
        process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades,
                           msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    }
    
    // Process on GPU
    Message* d_msgs;
    CUDA_CHECK(cudaMalloc(&d_msgs, messages.size() * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_msgs, messages.data(), messages.size() * sizeof(Message), 
                         cudaMemcpyHostToDevice));
    
    process_messages_sequential_kernel<<<1, 256>>>(gpu_batch, d_msgs, messages.size(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy and compare
    Order* gpu_bids = new Order[100];
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, 100 * sizeof(Order), 
                         cudaMemcpyDeviceToHost));
    
    bool match = compare_orders(cpu_book.bids, gpu_bids, 100, "Bids");
    
    // Verify quantity reduced to 50
    bool correct_qty = false;
    for (int i = 0; i < 100; i++) {
        if (cpu_book.bids[i].order_id == 2001 && cpu_book.bids[i].quantity == 50) {
            correct_qty = true;
            break;
        }
    }
    
    // Cleanup
    delete[] gpu_bids;
    cudaFree(gpu_batch.d_asks);
    cudaFree(gpu_batch.d_bids);
    cudaFree(gpu_batch.d_trades);
    cudaFree(d_msgs);
    cpu_book.cleanup();
    
    if (match && correct_qty) {
        std::cout << "  ✓ PASS: Order cancelled correctly, CPU == GPU" << std::endl;
        stats.record_pass();
        return true;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
        stats.record_fail();
        return false;
    }
}

bool unit_test_simple_match(TestStats& stats) {
    print_test_header("Unit Test: Simple Match");
    
    // Setup
    OrderbookCPU cpu_book;
    cpu_book.allocate(100, 50);
    
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = 1;
    gpu_batch.n_orders_per_book = 100;
    gpu_batch.n_trades_per_book = 50;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, 50 * sizeof(Trade)));
    
    init_orderbooks_kernel<<<1, 256>>>(gpu_batch, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Perfect match scenario
    auto messages = generate_perfect_match();
    print_messages(messages, "Input");
    
    // Process on CPU
    for (const auto& msg : messages) {
        process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades,
                           msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    }
    
    // Process on GPU
    Message* d_msgs;
    CUDA_CHECK(cudaMalloc(&d_msgs, messages.size() * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_msgs, messages.data(), messages.size() * sizeof(Message),
                         cudaMemcpyHostToDevice));
    
    process_messages_sequential_kernel<<<1, 256>>>(gpu_batch, d_msgs, messages.size(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy and compare
    Order* gpu_asks = new Order[100];
    Order* gpu_bids = new Order[100];
    Trade* gpu_trades = new Trade[50];
    
    CUDA_CHECK(cudaMemcpy(gpu_asks, gpu_batch.d_asks, 100 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, 100 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_trades, gpu_batch.d_trades, 50 * sizeof(Trade), cudaMemcpyDeviceToHost));
    
    bool match = compare_orders(cpu_book.asks, gpu_asks, 100, "Asks") &&
                 compare_orders(cpu_book.bids, gpu_bids, 100, "Bids") &&
                 compare_trades(cpu_book.trades, gpu_trades, 50);
    
    // Verify trade occurred
    bool trade_found = (cpu_book.trades[0].price == 101000 && 
                       cpu_book.trades[0].quantity == 100);
    
    // Verify book is empty
    bool book_empty = (cpu_book.asks[0].price == EMPTY_PRICE && 
                      cpu_book.bids[0].price == EMPTY_PRICE);
    
    // Cleanup
    delete[] gpu_asks;
    delete[] gpu_bids;
    delete[] gpu_trades;
    cudaFree(gpu_batch.d_asks);
    cudaFree(gpu_batch.d_bids);
    cudaFree(gpu_batch.d_trades);
    cudaFree(d_msgs);
    cpu_book.cleanup();
    
    if (match && trade_found && book_empty) {
        std::cout << "  ✓ PASS: Perfect match executed, CPU == GPU" << std::endl;
        stats.record_pass();
        return true;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
        stats.record_fail();
        return false;
    }
}

// ============================================================================
// LEVEL 2: INTEGRATION TESTS - Scenarios with Known Outcomes
// ============================================================================

bool integration_test_scenario(TestStats& stats, const char* name, 
                               std::vector<Message> (*generator)()) {
    print_test_header(name);
    
    // Setup
    OrderbookCPU cpu_book;
    cpu_book.allocate(100, 50);
    
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = 1;
    gpu_batch.n_orders_per_book = 100;
    gpu_batch.n_trades_per_book = 50;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, 100 * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, 50 * sizeof(Trade)));
    
    init_orderbooks_kernel<<<1, 256>>>(gpu_batch, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Generate scenario
    auto messages = generator();
    print_messages(messages, "Input");
    
    // Process on CPU
    for (const auto& msg : messages) {
        process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades,
                           msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    }
    
    // Process on GPU
    Message* d_msgs;
    CUDA_CHECK(cudaMalloc(&d_msgs, messages.size() * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_msgs, messages.data(), messages.size() * sizeof(Message),
                         cudaMemcpyHostToDevice));
    
    process_messages_sequential_kernel<<<1, 256>>>(gpu_batch, d_msgs, messages.size(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy and compare
    Order* gpu_asks = new Order[100];
    Order* gpu_bids = new Order[100];
    Trade* gpu_trades = new Trade[50];
    
    CUDA_CHECK(cudaMemcpy(gpu_asks, gpu_batch.d_asks, 100 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, 100 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_trades, gpu_batch.d_trades, 50 * sizeof(Trade), cudaMemcpyDeviceToHost));
    
    bool match = compare_orders(cpu_book.asks, gpu_asks, 100, "Asks") &&
                 compare_orders(cpu_book.bids, gpu_bids, 100, "Bids") &&
                 compare_trades(cpu_book.trades, gpu_trades, 50);
    
    // Cleanup
    delete[] gpu_asks;
    delete[] gpu_bids;
    delete[] gpu_trades;
    cudaFree(gpu_batch.d_asks);
    cudaFree(gpu_batch.d_bids);
    cudaFree(gpu_batch.d_trades);
    cudaFree(d_msgs);
    cpu_book.cleanup();
    
    if (match) {
        std::cout << "  ✓ PASS: CPU == GPU" << std::endl;
        stats.record_pass();
        return true;
    } else {
        std::cout << "  ✗ FAIL: CPU != GPU" << std::endl;
        stats.record_fail();
        return false;
    }
}

// ============================================================================
// LEVEL 3: FUNCTIONAL TESTS - CPU vs GPU with Random Data
// ============================================================================

bool functional_test_random(TestStats& stats, int num_messages, const char* size_name, int max_orders_override = -1, int num_books = 1) {
    print_test_header((std::string("Functional Test: Random ") + size_name).c_str());
    
    std::cout << "  Testing with " << num_messages << " random messages/book, " << num_books << " orderbook(s)..." << std::endl;
    
    // Match benchmark sizing: 10% of messages (realistic market depth)
    // This is what the benchmarks use and is more realistic than 1:1
    int n_orders = (max_orders_override > 0) ? max_orders_override : std::max(100, num_messages / 10);
    int n_trades = std::max(100, num_messages / 10);
    
    std::cout << "  Orders per side: " << n_orders << ", Max trades: " << n_trades << std::endl;
    
    // Setup CPU (test only first orderbook for correctness)
    OrderbookCPU cpu_book;
    cpu_book.allocate(n_orders, n_trades);
    
    // Setup GPU (all orderbooks)
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = num_books;
    gpu_batch.n_orders_per_book = n_orders;
    gpu_batch.n_trades_per_book = n_trades;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, num_books * n_orders * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, num_books * n_orders * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, num_books * n_trades * sizeof(Trade)));
    
    // Initialize all orderbooks
    dim3 grid(num_books);
    dim3 block(256);
    init_orderbooks_kernel<<<grid, block>>>(gpu_batch, num_books);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Generate random data (same messages for all books for CPU comparison)
    auto messages = generate_random_messages(num_messages);
    
    // Process on CPU (single orderbook)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (const auto& msg : messages) {
        process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades,
                           msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // Allocate and copy messages to GPU (replicate for each book)
    Message* d_msgs;
    size_t total_messages = num_books * num_messages;
    CUDA_CHECK(cudaMalloc(&d_msgs, total_messages * sizeof(Message)));
    
    // Copy same messages for each orderbook
    for (int i = 0; i < num_books; i++) {
        CUDA_CHECK(cudaMemcpy(d_msgs + i * num_messages, messages.data(), 
                             num_messages * sizeof(Message), cudaMemcpyHostToDevice));
    }
    
    // Process on GPU (all orderbooks in parallel)
    auto gpu_start = std::chrono::high_resolution_clock::now();
    dim3 grid_proc(num_books);
    dim3 block_proc(256);
    process_messages_sequential_kernel<<<grid_proc, block_proc>>>(gpu_batch, d_msgs, num_messages, num_books);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    // Copy and compare (only first orderbook for correctness check)
    Order* gpu_asks = new Order[n_orders];
    Order* gpu_bids = new Order[n_orders];
    Trade* gpu_trades = new Trade[n_trades];
    
    CUDA_CHECK(cudaMemcpy(gpu_asks, gpu_batch.d_asks, n_orders * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, n_orders * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_trades, gpu_batch.d_trades, n_trades * sizeof(Trade), cudaMemcpyDeviceToHost));
    
    // Check orderbook utilization (detect if we're close to full = potential drops)
    int cpu_asks_used = 0, cpu_bids_used = 0;
    int gpu_asks_used = 0, gpu_bids_used = 0;
    for (int i = 0; i < n_orders; i++) {
        if (cpu_book.asks[i].price != EMPTY_PRICE) cpu_asks_used++;
        if (cpu_book.bids[i].price != EMPTY_PRICE) cpu_bids_used++;
        if (gpu_asks[i].price != EMPTY_PRICE) gpu_asks_used++;
        if (gpu_bids[i].price != EMPTY_PRICE) gpu_bids_used++;
    }
    
    int max_used = std::max({cpu_asks_used, cpu_bids_used, gpu_asks_used, gpu_bids_used});
    double utilization = (double)max_used / n_orders * 100.0;
    
    bool match = compare_orders(cpu_book.asks, gpu_asks, n_orders, "Asks") &&
                 compare_orders(cpu_book.bids, gpu_bids, n_orders, "Bids") &&
                 compare_trades(cpu_book.trades, gpu_trades, n_trades);
    
    std::cout << "  CPU time: " << cpu_time << " μs" << std::endl;
    std::cout << "  GPU time: " << gpu_time << " μs" << std::endl;
    std::cout << "  Orderbook utilization: " << std::fixed << std::setprecision(1) << utilization << "% "
              << "(" << max_used << "/" << n_orders << " slots used)" << std::endl;
    
    // Warn if orderbook utilization is high (risk of dropped orders)
    if (utilization > 90.0) {
        std::cout << "  ⚠️  WARNING: Orderbook >90% full! Orders may have been dropped!" << std::endl;
        std::cout << "      Rerun with: --max-orders " << (n_orders * 2) << std::endl;
    } else if (utilization > 75.0) {
        std::cout << "  ⚠️  CAUTION: Orderbook >75% full. Consider using --max-orders " 
                  << (n_orders * 3 / 2) << std::endl;
    }
    
    // Calculate speedup and throughput
    if (num_books > 1) {
        double speedup = (double)cpu_time * num_books / gpu_time;
        double throughput = (double)(num_messages * num_books) / (gpu_time / 1000.0);  // msgs/ms
        std::cout << "  Speedup (parallel): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << throughput << " msgs/ms" << std::endl;
    } else {
        double speedup = (double)cpu_time / gpu_time;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Cleanup
    delete[] gpu_asks;
    delete[] gpu_bids;
    delete[] gpu_trades;
    cudaFree(gpu_batch.d_asks);
    cudaFree(gpu_batch.d_bids);
    cudaFree(gpu_batch.d_trades);
    cudaFree(d_msgs);
    cpu_book.cleanup();
    
    if (match) {
        std::cout << "  ✓ PASS: CPU == GPU" << std::endl;
        stats.record_pass();
        return true;
    } else {
        std::cout << "  ✗ FAIL: CPU != GPU" << std::endl;
        stats.record_fail();
        return false;
    }
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA ORDERBOOK TEST SUITE" << std::endl;
    std::cout << "Comprehensive Testing: Unit → Integration → Functional" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Parse command-line arguments
    bool run_unit = true;
    bool run_integration = true;
    bool run_functional = true;
    std::vector<int> message_sizes;
    int max_orders = 1000;
    int num_books = 1;
    
    if (argc > 1) {
        // Check for help
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "\nUsage: " << argv[0] << " [OPTIONS] [MESSAGE_SIZES...]" << std::endl;
            std::cout << "\nOptions:" << std::endl;
            std::cout << "  -h, --help           Show this help message" << std::endl;
            std::cout << "  --unit-only          Run only unit tests" << std::endl;
            std::cout << "  --integration-only   Run only integration tests" << std::endl;
            std::cout << "  --functional-only    Run only functional tests" << std::endl;
            std::cout << "  --skip-unit          Skip unit tests" << std::endl;
            std::cout << "  --skip-integration   Skip integration tests" << std::endl;
            std::cout << "  --skip-functional    Skip functional tests" << std::endl;
            std::cout << "  --max-orders N       Set max orders per side (default: 1000)" << std::endl;
            std::cout << "  --num-books N        Number of orderbooks to test in parallel (default: 1)" << std::endl;
            std::cout << "\nMessage Sizes:" << std::endl;
            std::cout << "  Specify one or more message counts for functional tests" << std::endl;
            std::cout << "  Example: ./test_suite 100 1000 10000" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  ./test_suite                    # Run all tests with defaults" << std::endl;
            std::cout << "  ./test_suite 100 500            # Run all tests, functional with 100 & 500 msgs" << std::endl;
            std::cout << "  ./test_suite --num-books 100 1000  # Test 100 parallel orderbooks, 1000 msgs" << std::endl;
            std::cout << "  ./test_suite --functional-only --num-books 1000 10000  # 1000 books, 10000 msgs" << std::endl;
            std::cout << "  ./test_suite --skip-functional  # Run only unit & integration tests" << std::endl;
            std::cout << "  ./test_suite --max-orders 500 1000 10000  # Custom orders, test 1000 & 10000 msgs" << std::endl;
            return 0;
        }
        
        // Parse arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--unit-only") {
                run_integration = false;
                run_functional = false;
            } else if (arg == "--integration-only") {
                run_unit = false;
                run_functional = false;
            } else if (arg == "--functional-only") {
                run_unit = false;
                run_integration = false;
            } else if (arg == "--skip-unit") {
                run_unit = false;
            } else if (arg == "--skip-integration") {
                run_integration = false;
            } else if (arg == "--skip-functional") {
                run_functional = false;
            } else if (arg == "--max-orders") {
                if (i + 1 < argc) {
                    max_orders = std::atoi(argv[++i]);
                }
            } else if (arg == "--num-books") {
                if (i + 1 < argc) {
                    num_books = std::atoi(argv[++i]);
                }
            } else {
                // Try to parse as message size
                int size = std::atoi(arg.c_str());
                if (size > 0) {
                    message_sizes.push_back(size);
                }
            }
        }
    }
    
    // Default message sizes if none specified
    if (message_sizes.empty() && run_functional) {
        message_sizes = {100, 500, 1000, 5000, 10000};
    }
    
    if (argc > 1) {
        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Unit tests: " << (run_unit ? "YES" : "NO") << std::endl;
        std::cout << "  Integration tests: " << (run_integration ? "YES" : "NO") << std::endl;
        std::cout << "  Functional tests: " << (run_functional ? "YES" : "NO") << std::endl;
        if (run_functional) {
            std::cout << "  Number of orderbooks: " << num_books << std::endl;
            std::cout << "  Message sizes: ";
            for (size_t i = 0; i < message_sizes.size(); i++) {
                std::cout << message_sizes[i];
                if (i < message_sizes.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << "  Max orders per side: " << max_orders << std::endl;
        }
    }
    
    TestStats stats;
    
    // ========================================================================
    // LEVEL 1: UNIT TESTS
    // ========================================================================
    if (run_unit) {
        std::cout << "\n\n" << std::string(60, '=') << std::endl;
        std::cout << "LEVEL 1: UNIT TESTS (Individual Operations)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        unit_test_add_order(stats);
        unit_test_cancel_order(stats);
        unit_test_simple_match(stats);
    }
    
    // ========================================================================
    // LEVEL 2: INTEGRATION TESTS
    // ========================================================================
    if (run_integration) {
        std::cout << "\n\n" << std::string(60, '=') << std::endl;
        std::cout << "LEVEL 2: INTEGRATION TESTS (Known Scenarios)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        integration_test_scenario(stats, "Integration: Partial Fill", generate_partial_fill);
        integration_test_scenario(stats, "Integration: No Match", generate_no_match);
        integration_test_scenario(stats, "Integration: Price Improvement", generate_price_improvement);
        integration_test_scenario(stats, "Integration: Cancel Test", generate_cancel_test);
        integration_test_scenario(stats, "Integration: Market Order", generate_market_order);
        integration_test_scenario(stats, "Integration: Price-Time Priority", generate_price_time_priority);
        integration_test_scenario(stats, "Integration: Multi-Level Book", generate_multilevel_book);
    }
    
    // ========================================================================
    // LEVEL 3: FUNCTIONAL TESTS
    // ========================================================================
    if (run_functional) {
        std::cout << "\n\n" << std::string(60, '=') << std::endl;
        std::cout << "LEVEL 3: FUNCTIONAL TESTS (Random Data, CPU vs GPU)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Run functional tests with specified message sizes and orderbook count
        for (size_t i = 0; i < message_sizes.size(); i++) {
            int num_msgs = message_sizes[i];
            std::string name = "Test (" + std::to_string(num_msgs) + " messages)";
            functional_test_random(stats, num_msgs, name.c_str(), max_orders, num_books);
        }
    }
    
    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    stats.print_summary();
    
    return (stats.tests_failed == 0) ? 0 : 1;
}


