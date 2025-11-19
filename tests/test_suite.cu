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
    
    dim3 grid((1 + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(gpu_batch, d_msg, 1, 1);
    CUDA_CHECK(cudaGetLastError());
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
    
    dim3 grid((1 + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(gpu_batch, d_msgs, messages.size(), 1);
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
    
    dim3 grid((1 + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(gpu_batch, d_msgs, messages.size(), 1);
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
    
    dim3 grid((1 + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(gpu_batch, d_msgs, messages.size(), 1);
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

bool functional_test_random(TestStats& stats, int num_messages, const char* size_name, int max_orders_override = -1) {
    print_test_header((std::string("Functional Test: Random ") + size_name).c_str());
    
    std::cout << "  Testing with " << num_messages << " random messages..." << std::endl;
    
    // Scale orders and trades based on message count or use override
    int n_orders = (max_orders_override > 0) ? max_orders_override : std::max(200, num_messages / 10);
    int n_trades = std::max(100, num_messages / 20);  // At least 100, or 5% of messages
    
    std::cout << "  Orders per side: " << n_orders << ", Max trades: " << n_trades << std::endl;
    
    // Setup
    OrderbookCPU cpu_book;
    cpu_book.allocate(n_orders, n_trades);
    
    OrderbookBatch gpu_batch;
    gpu_batch.num_books = 1;
    gpu_batch.n_orders_per_book = n_orders;
    gpu_batch.n_trades_per_book = n_trades;
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_asks, n_orders * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_bids, n_orders * sizeof(Order)));
    CUDA_CHECK(cudaMalloc(&gpu_batch.d_trades, n_trades * sizeof(Trade)));
    
    init_orderbooks_kernel<<<1, 256>>>(gpu_batch, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Generate random data
    auto messages = generate_random_messages(num_messages);
    
    // Process on CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (const auto& msg : messages) {
        process_message_cpu(cpu_book.asks, cpu_book.bids, cpu_book.trades,
                           msg, cpu_book.n_orders_per_side, cpu_book.n_trades);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // Process on GPU
    Message* d_msgs;
    CUDA_CHECK(cudaMalloc(&d_msgs, messages.size() * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_msgs, messages.data(), messages.size() * sizeof(Message),
                         cudaMemcpyHostToDevice));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    dim3 grid((1 + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(gpu_batch, d_msgs, messages.size(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    // Copy and compare
    Order* gpu_asks = new Order[200];
    Order* gpu_bids = new Order[200];
    Trade* gpu_trades = new Trade[100];
    
    CUDA_CHECK(cudaMemcpy(gpu_asks, gpu_batch.d_asks, 200 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bids, gpu_batch.d_bids, 200 * sizeof(Order), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_trades, gpu_batch.d_trades, 100 * sizeof(Trade), cudaMemcpyDeviceToHost));
    
    bool match = compare_orders(cpu_book.asks, gpu_asks, 200, "Asks") &&
                 compare_orders(cpu_book.bids, gpu_bids, 200, "Bids") &&
                 compare_trades(cpu_book.trades, gpu_trades, 100);
    
    std::cout << "  CPU time: " << cpu_time << " μs" << std::endl;
    std::cout << "  GPU time: " << gpu_time << " μs" << std::endl;
    
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
    
    // Parse command-line arguments for functional test scaling
    int max_messages = (argc > 1) ? std::atoi(argv[1]) : 10000;
    int max_orders = (argc > 2) ? std::atoi(argv[2]) : 1000;
    
    if (argc > 1) {
        std::cout << "\nCustom Functional Test Configuration:" << std::endl;
        std::cout << "  Max messages: " << max_messages << std::endl;
        std::cout << "  Max orders per side: " << max_orders << std::endl;
    }
    
    TestStats stats;
    
    // ========================================================================
    // LEVEL 1: UNIT TESTS
    // ========================================================================
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "LEVEL 1: UNIT TESTS (Individual Operations)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    unit_test_add_order(stats);
    unit_test_cancel_order(stats);
    unit_test_simple_match(stats);
    
    // ========================================================================
    // LEVEL 2: INTEGRATION TESTS
    // ========================================================================
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
    
    // ========================================================================
    // LEVEL 3: FUNCTIONAL TESTS
    // ========================================================================
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "LEVEL 3: FUNCTIONAL TESTS (Random Data, CPU vs GPU)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Use custom values if provided, otherwise use defaults with progressive scaling
    functional_test_random(stats, 100, "Small (100 messages)", max_orders);
    functional_test_random(stats, 500, "Medium (500 messages)", max_orders);
    functional_test_random(stats, 1000, "Large (1000 messages)", max_orders);
    
    // Only run larger tests if max_messages allows
    if (max_messages >= 5000) {
        functional_test_random(stats, 5000, "Very Large (5000 messages)", max_orders);
    }
    if (max_messages >= 10000) {
        functional_test_random(stats, 10000, "Massive (10000 messages)", max_orders);
    }
    
    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    stats.print_summary();
    
    return (stats.tests_failed == 0) ? 0 : 1;
}

