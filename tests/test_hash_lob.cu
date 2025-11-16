/**
 * Hash-Accelerated LOB Test Suite
 * 
 * Correctness tests for hash-based orderbook implementation
 * Tests both cuCollections and simple CUDA hash implementations
 * 
 * Compile: See tests/build_and_test.sh
 * Run: ./test_hash_lob
 */

#include "types.h"
#include "simple_hash.cuh"
#include "cuco_wrapper.cuh"
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstring>

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

// Forward declarations of kernels
extern "C" {
    __global__ void init_hash_orderbooks_kernel(HashOrderbookBatch batch);
    __global__ void add_order_hash_kernel(HashOrderbookBatch batch, const Message* messages, int num_books);
    __global__ void cancel_order_hash_kernel(HashOrderbookBatch batch, const Message* messages, int num_books);
    __global__ void process_messages_hash_kernel(HashOrderbookBatch batch, const Message* messages, int num_messages_per_book, int num_books);
    __global__ void get_best_bid_ask_hash_kernel(const HashOrderbookBatch batch, int32_t* best_asks, int32_t* best_bids, int num_books);
}

// ============================================================================
// Test Helper Functions
// ============================================================================

class HashLOBTest {
private:
    HashOrderbookBatch batch;
    HashOrderbookState* h_states;
    Order* h_asks;
    Order* h_bids;
    Trade* h_trades;
    int num_books;
    int n_orders;
    int n_trades;
    HashImplementation impl;
    
public:
    HashLOBTest(int num_books, int n_orders, int n_trades, HashImplementation impl) 
        : num_books(num_books), n_orders(n_orders), n_trades(n_trades), impl(impl) {
        
        // Allocate host memory
        h_states = new HashOrderbookState[num_books];
        h_asks = new Order[num_books * n_orders];
        h_bids = new Order[num_books * n_orders];
        h_trades = new Trade[num_books * n_trades];
        
        // Allocate device memory for orders and trades
        CUDA_CHECK(cudaMalloc(&batch.d_asks, num_books * n_orders * sizeof(Order)));
        CUDA_CHECK(cudaMalloc(&batch.d_bids, num_books * n_orders * sizeof(Order)));
        CUDA_CHECK(cudaMalloc(&batch.d_trades, num_books * n_trades * sizeof(Trade)));
        CUDA_CHECK(cudaMalloc(&batch.states, num_books * sizeof(HashOrderbookState)));
        
        // Initialize batch
        batch.num_books = num_books;
        batch.n_orders_per_book = n_orders;
        batch.n_trades_per_book = n_trades;
        batch.hash_impl = impl;
        
        // Initialize each orderbook state
        for (int i = 0; i < num_books; i++) {
            h_states[i].asks = batch.d_asks + (i * n_orders);
            h_states[i].bids = batch.d_bids + (i * n_orders);
            h_states[i].trades = batch.d_trades + (i * n_trades);
            h_states[i].n_orders = n_orders;
            h_states[i].n_trades = n_trades;
            h_states[i].hash_impl = impl;
            h_states[i].asks_sorted = false;
            h_states[i].bids_sorted = false;
            
            // Create hash maps
            if (impl == HASH_CUCOLLECTIONS) {
                h_states[i].ask_hash_map = cuco_create_host(n_orders);
                h_states[i].bid_hash_map = cuco_create_host(n_orders);
            } else {
                SimpleHashTable* ask_table = new SimpleHashTable();
                *ask_table = simple_hash_create_host(n_orders * 2);
                h_states[i].ask_hash_map = static_cast<void*>(ask_table);
                
                SimpleHashTable* bid_table = new SimpleHashTable();
                *bid_table = simple_hash_create_host(n_orders * 2);
                h_states[i].bid_hash_map = static_cast<void*>(bid_table);
            }
        }
        
        // Copy states to device
        CUDA_CHECK(cudaMemcpy(batch.states, h_states, 
                             num_books * sizeof(HashOrderbookState),
                             cudaMemcpyHostToDevice));
        
        // Initialize orderbooks
        init_hash_orderbooks_kernel<<<num_books, 256>>>(batch);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    ~HashLOBTest() {
        // Free hash maps
        for (int i = 0; i < num_books; i++) {
            if (impl == HASH_CUCOLLECTIONS) {
                cuco_destroy_host(h_states[i].ask_hash_map);
                cuco_destroy_host(h_states[i].bid_hash_map);
            } else {
                SimpleHashTable* ask_table = static_cast<SimpleHashTable*>(h_states[i].ask_hash_map);
                simple_hash_destroy_host(ask_table);
                delete ask_table;
                
                SimpleHashTable* bid_table = static_cast<SimpleHashTable*>(h_states[i].bid_hash_map);
                simple_hash_destroy_host(bid_table);
                delete bid_table;
            }
        }
        
        // Free device memory
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        cudaFree(batch.states);
        
        // Free host memory
        delete[] h_states;
        delete[] h_asks;
        delete[] h_bids;
        delete[] h_trades;
    }
    
    void copy_to_host() {
        CUDA_CHECK(cudaMemcpy(h_asks, batch.d_asks, 
                             num_books * n_orders * sizeof(Order),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bids, batch.d_bids,
                             num_books * n_orders * sizeof(Order),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_trades, batch.d_trades,
                             num_books * n_trades * sizeof(Trade),
                             cudaMemcpyDeviceToHost));
    }
    
    void print_orderbook(int book_idx = 0) {
        copy_to_host();
        
        std::cout << "\n=== Orderbook " << book_idx << " ===" << std::endl;
        std::cout << "\nAsks:" << std::endl;
        for (int i = 0; i < 10; i++) {
            int idx = book_idx * n_orders + i;
            if (h_asks[idx].price != EMPTY_PRICE) {
                std::cout << "  Price: " << h_asks[idx].price
                         << " Qty: " << h_asks[idx].quantity
                         << " ID: " << h_asks[idx].order_id << std::endl;
            }
        }
        
        std::cout << "\nBids:" << std::endl;
        for (int i = 0; i < 10; i++) {
            int idx = book_idx * n_orders + i;
            if (h_bids[idx].price != EMPTY_PRICE) {
                std::cout << "  Price: " << h_bids[idx].price
                         << " Qty: " << h_bids[idx].quantity
                         << " ID: " << h_bids[idx].order_id << std::endl;
            }
        }
        
        std::cout << "\nTrades:" << std::endl;
        for (int i = 0; i < n_trades; i++) {
            int idx = book_idx * n_trades + i;
            if (h_trades[idx].price != EMPTY_PRICE) {
                std::cout << "  Price: " << h_trades[idx].price
                         << " Qty: " << h_trades[idx].quantity
                         << " Passive: " << h_trades[idx].passive_order_id
                         << " Aggressive: " << h_trades[idx].aggressive_order_id << std::endl;
            }
        }
    }
    
    Order* get_asks() { return h_asks; }
    Order* get_bids() { return h_bids; }
    Trade* get_trades() { return h_trades; }
    HashOrderbookBatch& get_batch() { return batch; }
};

// ============================================================================
// Test Cases
// ============================================================================

bool test_add_order(HashImplementation impl) {
    std::cout << "\n[TEST] Add Order (" 
              << (impl == HASH_CUCOLLECTIONS ? "cuCollections" : "Simple Hash") 
              << ")" << std::endl;
    
    HashLOBTest test(1, 100, 10, impl);
    
    // Create messages
    Message h_messages[2];
    h_messages[0] = {Message::LIMIT, Message::ASK, 100, 10050, 1, 1001, 0, 0};
    h_messages[1] = {Message::LIMIT, Message::BID, 100, 10000, 1, 1002, 0, 0};
    
    Message* d_messages;
    CUDA_CHECK(cudaMalloc(&d_messages, 2 * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_messages, h_messages, 2 * sizeof(Message), cudaMemcpyHostToDevice));
    
    // Add ask
    add_order_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Add bid
    add_order_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages + 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    test.copy_to_host();
    
    bool pass = true;
    
    // Verify ask added
    if (test.get_asks()[0].price != 10050 || test.get_asks()[0].quantity != 100) {
        std::cout << "  FAIL: Ask not added correctly" << std::endl;
        pass = false;
    }
    
    // Verify bid added
    if (test.get_bids()[0].price != 10000 || test.get_bids()[0].quantity != 100) {
        std::cout << "  FAIL: Bid not added correctly" << std::endl;
        pass = false;
    }
    
    cudaFree(d_messages);
    
    if (pass) {
        std::cout << "  PASS" << std::endl;
    }
    
    return pass;
}

bool test_cancel_order(HashImplementation impl) {
    std::cout << "\n[TEST] Cancel Order (" 
              << (impl == HASH_CUCOLLECTIONS ? "cuCollections" : "Simple Hash") 
              << ")" << std::endl;
    
    HashLOBTest test(1, 100, 10, impl);
    
    // Add order then cancel it
    Message h_messages[2];
    h_messages[0] = {Message::LIMIT, Message::ASK, 100, 10050, 1, 1001, 0, 0};  // Add
    h_messages[1] = {Message::CANCEL, Message::ASK, 50, 10050, 1, 1001, 0, 0};  // Cancel half
    
    Message* d_messages;
    CUDA_CHECK(cudaMalloc(&d_messages, 2 * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_messages, h_messages, 2 * sizeof(Message), cudaMemcpyHostToDevice));
    
    // Add
    add_order_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cancel
    cancel_order_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages + 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    test.copy_to_host();
    
    bool pass = true;
    
    // Verify quantity reduced
    if (test.get_asks()[0].quantity != 50) {
        std::cout << "  FAIL: Quantity not reduced correctly (got " 
                  << test.get_asks()[0].quantity << ", expected 50)" << std::endl;
        pass = false;
    }
    
    cudaFree(d_messages);
    
    if (pass) {
        std::cout << "  PASS" << std::endl;
    }
    
    return pass;
}

bool test_best_price(HashImplementation impl) {
    std::cout << "\n[TEST] Best Price (" 
              << (impl == HASH_CUCOLLECTIONS ? "cuCollections" : "Simple Hash") 
              << ")" << std::endl;
    
    HashLOBTest test(1, 100, 10, impl);
    
    // Add multiple orders at different prices
    Message h_messages[4];
    h_messages[0] = {Message::LIMIT, Message::ASK, 100, 10060, 1, 1001, 0, 0};
    h_messages[1] = {Message::LIMIT, Message::ASK, 100, 10050, 1, 1002, 1, 0};  // Best ask
    h_messages[2] = {Message::LIMIT, Message::BID, 100, 9990, 1, 1003, 2, 0};
    h_messages[3] = {Message::LIMIT, Message::BID, 100, 10000, 1, 1004, 3, 0};  // Best bid
    
    Message* d_messages;
    CUDA_CHECK(cudaMalloc(&d_messages, 4 * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_messages, h_messages, 4 * sizeof(Message), cudaMemcpyHostToDevice));
    
    // Add all orders
    for (int i = 0; i < 4; i++) {
        add_order_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages + i, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Query best prices
    int32_t *d_best_asks, *d_best_bids;
    CUDA_CHECK(cudaMalloc(&d_best_asks, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_best_bids, sizeof(int32_t)));
    
    get_best_bid_ask_hash_kernel<<<1, 256>>>(test.get_batch(), d_best_asks, d_best_bids, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int32_t h_best_ask, h_best_bid;
    CUDA_CHECK(cudaMemcpy(&h_best_ask, d_best_asks, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_best_bid, d_best_bids, sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    bool pass = true;
    
    if (h_best_ask != 10050) {
        std::cout << "  FAIL: Best ask incorrect (got " << h_best_ask << ", expected 10050)" << std::endl;
        pass = false;
    }
    
    if (h_best_bid != 10000) {
        std::cout << "  FAIL: Best bid incorrect (got " << h_best_bid << ", expected 10000)" << std::endl;
        pass = false;
    }
    
    cudaFree(d_messages);
    cudaFree(d_best_asks);
    cudaFree(d_best_bids);
    
    if (pass) {
        std::cout << "  PASS" << std::endl;
    }
    
    return pass;
}

bool test_matching(HashImplementation impl) {
    std::cout << "\n[TEST] Order Matching (" 
              << (impl == HASH_CUCOLLECTIONS ? "cuCollections" : "Simple Hash") 
              << ")" << std::endl;
    
    HashLOBTest test(1, 100, 10, impl);
    
    // Add passive asks, then aggressive buy
    Message h_messages[3];
    h_messages[0] = {Message::LIMIT, Message::ASK, 50, 10050, 1, 1001, 0, 0};  // Passive
    h_messages[1] = {Message::LIMIT, Message::ASK, 50, 10060, 1, 1002, 1, 0};  // Passive
    h_messages[2] = {Message::LIMIT, Message::BID, 75, 10055, 2, 1003, 2, 0};  // Aggressive buy
    
    Message* d_messages;
    CUDA_CHECK(cudaMalloc(&d_messages, 3 * sizeof(Message)));
    CUDA_CHECK(cudaMemcpy(d_messages, h_messages, 3 * sizeof(Message), cudaMemcpyHostToDevice));
    
    // Process all messages
    process_messages_hash_kernel<<<1, 256>>>(test.get_batch(), d_messages, 3, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    test.copy_to_host();
    
    bool pass = true;
    
    // Should have matched 50 at 10050
    if (test.get_trades()[0].price != 10050 || test.get_trades()[0].quantity != 50) {
        std::cout << "  FAIL: First trade incorrect" << std::endl;
        pass = false;
    }
    
    cudaFree(d_messages);
    
    if (pass) {
        std::cout << "  PASS" << std::endl;
    }
    
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Hash-Accelerated LOB Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int tests_passed = 0;
    int tests_total = 0;
    
    // Run tests with cuCollections
    std::cout << "\n### Testing with cuCollections ###" << std::endl;
    tests_total++;
    if (test_add_order(HASH_CUCOLLECTIONS)) tests_passed++;
    
    tests_total++;
    if (test_cancel_order(HASH_CUCOLLECTIONS)) tests_passed++;
    
    tests_total++;
    if (test_best_price(HASH_CUCOLLECTIONS)) tests_passed++;
    
    tests_total++;
    if (test_matching(HASH_CUCOLLECTIONS)) tests_passed++;
    
    // Run tests with Simple Hash
    std::cout << "\n### Testing with Simple CUDA Hash ###" << std::endl;
    tests_total++;
    if (test_add_order(HASH_SIMPLE_CUDA)) tests_passed++;
    
    tests_total++;
    if (test_cancel_order(HASH_SIMPLE_CUDA)) tests_passed++;
    
    tests_total++;
    if (test_best_price(HASH_SIMPLE_CUDA)) tests_passed++;
    
    tests_total++;
    if (test_matching(HASH_SIMPLE_CUDA)) tests_passed++;
    
    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << tests_passed << "/" << tests_total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (tests_passed == tests_total) ? 0 : 1;
}

