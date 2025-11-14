/**
 * Matching Engine Test Suite
 * 
 * Comprehensive tests for CUDA orderbook matching engine
 * Run on cloud GPU to verify correctness
 * 
 * Compile: nvcc -I../include -L../build -lcuda_orderbook test_matching.cu -o test_matching
 * Run: ./test_matching
 */

#include "types.h"
#include "kernels.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

using namespace cuda_orderbook;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper function to print orderbook
void print_orderbook(const Order* asks, const Order* bids, int n_orders, const char* title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
    
    std::cout << "\nAsks (Sell Orders):" << std::endl;
    std::cout << std::setw(10) << "Price" << std::setw(10) << "Quantity" 
              << std::setw(10) << "OrderID" << std::setw(10) << "TraderID" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    int ask_count = 0;
    for (int i = 0; i < n_orders && ask_count < 10; i++) {
        if (asks[i].price != EMPTY_PRICE) {
            std::cout << std::setw(10) << asks[i].price 
                      << std::setw(10) << asks[i].quantity
                      << std::setw(10) << asks[i].order_id
                      << std::setw(10) << asks[i].trader_id << std::endl;
            ask_count++;
        }
    }
    if (ask_count == 0) std::cout << "(empty)" << std::endl;
    
    std::cout << "\nBids (Buy Orders):" << std::endl;
    std::cout << std::setw(10) << "Price" << std::setw(10) << "Quantity" 
              << std::setw(10) << "OrderID" << std::setw(10) << "TraderID" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    int bid_count = 0;
    for (int i = 0; i < n_orders && bid_count < 10; i++) {
        if (bids[i].price != EMPTY_PRICE) {
            std::cout << std::setw(10) << bids[i].price 
                      << std::setw(10) << bids[i].quantity
                      << std::setw(10) << bids[i].order_id
                      << std::setw(10) << bids[i].trader_id << std::endl;
            bid_count++;
        }
    }
    if (bid_count == 0) std::cout << "(empty)" << std::endl;
}

// Helper function to print trades
void print_trades(const Trade* trades, int n_trades, const char* title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << std::setw(10) << "Price" << std::setw(10) << "Quantity" 
              << std::setw(12) << "PassiveID" << std::setw(12) << "AggressID" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    int trade_count = 0;
    for (int i = 0; i < n_trades; i++) {
        if (trades[i].price != EMPTY_PRICE) {
            std::cout << std::setw(10) << trades[i].price 
                      << std::setw(10) << trades[i].quantity
                      << std::setw(12) << trades[i].passive_order_id
                      << std::setw(12) << trades[i].aggressive_order_id << std::endl;
            trade_count++;
        }
    }
    if (trade_count == 0) std::cout << "(no trades)" << std::endl;
}

// Test helper class
class OrderbookTest {
private:
    OrderbookBatch batch;
    Order* h_asks;
    Order* h_bids;
    Trade* h_trades;
    int num_books;
    int n_orders;
    int n_trades;
    
public:
    OrderbookTest(int num_books = 1, int n_orders = 100, int n_trades = 100) 
        : num_books(num_books), n_orders(n_orders), n_trades(n_trades) {
        
        // Setup batch
        batch.num_books = num_books;
        batch.n_orders_per_book = n_orders;
        batch.n_trades_per_book = n_trades;
        
        // Allocate device memory
        size_t order_size = num_books * n_orders * sizeof(Order);
        size_t trade_size = num_books * n_trades * sizeof(Trade);
        
        CUDA_CHECK(cudaMalloc(&batch.d_asks, order_size));
        CUDA_CHECK(cudaMalloc(&batch.d_bids, order_size));
        CUDA_CHECK(cudaMalloc(&batch.d_trades, trade_size));
        
        // Allocate host memory
        h_asks = new Order[n_orders];
        h_bids = new Order[n_orders];
        h_trades = new Trade[n_trades];
        
        // Initialize
        init();
    }
    
    ~OrderbookTest() {
        cudaFree(batch.d_asks);
        cudaFree(batch.d_bids);
        cudaFree(batch.d_trades);
        delete[] h_asks;
        delete[] h_bids;
        delete[] h_trades;
    }
    
    void init() {
        dim3 grid(num_books);
        dim3 block(256);
        init_orderbooks_kernel<<<grid, block>>>(batch, num_books);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void process_messages(const Message* h_messages, int num_messages) {
        // Copy messages to device
        Message* d_messages;
        CUDA_CHECK(cudaMalloc(&d_messages, num_messages * sizeof(Message)));
        CUDA_CHECK(cudaMemcpy(d_messages, h_messages, 
                             num_messages * sizeof(Message), 
                             cudaMemcpyHostToDevice));
        
        // Process
        dim3 grid(num_books);
        dim3 block(256);
        process_messages_sequential_kernel<<<grid, block>>>(
            batch, d_messages, num_messages, num_books
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaFree(d_messages);
    }
    
    void copy_to_host() {
        CUDA_CHECK(cudaMemcpy(h_asks, batch.d_asks, 
                             n_orders * sizeof(Order), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bids, batch.d_bids, 
                             n_orders * sizeof(Order), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_trades, batch.d_trades, 
                             n_trades * sizeof(Trade), 
                             cudaMemcpyDeviceToHost));
    }
    
    void print() {
        copy_to_host();
        print_orderbook(h_asks, h_bids, n_orders, "Orderbook State");
        print_trades(h_trades, n_trades, "Trades");
    }
    
    Order* get_asks() { return h_asks; }
    Order* get_bids() { return h_bids; }
    Trade* get_trades() { return h_trades; }
};

// ============================================================================
// TEST CASES
// ============================================================================

bool test_add_order() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 1: Add Order" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    // Create buy limit order
    Message msg;
    msg.type = Message::LIMIT;
    msg.side = Message::BID;
    msg.quantity = 100;
    msg.price = 100000;
    msg.order_id = 1001;
    msg.trader_id = 1;
    msg.time_sec = 34200;
    msg.time_ns = 0;
    
    ob.process_messages(&msg, 1);
    ob.print();
    
    // Verify
    const Order* bids = ob.get_bids();
    bool found = false;
    for (int i = 0; i < 100; i++) {
        if (bids[i].order_id == 1001) {
            found = true;
            if (bids[i].price != 100000 || bids[i].quantity != 100) {
                std::cout << "❌ FAIL: Incorrect order values" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!found) {
        std::cout << "❌ FAIL: Order not found" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Order added correctly" << std::endl;
    return true;
}

bool test_cancel_order() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 2: Cancel Order" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    // Add order
    Message add_msg;
    add_msg.type = Message::LIMIT;
    add_msg.side = Message::BID;
    add_msg.quantity = 100;
    add_msg.price = 100000;
    add_msg.order_id = 1001;
    add_msg.trader_id = 1;
    add_msg.time_sec = 34200;
    add_msg.time_ns = 0;
    
    ob.process_messages(&add_msg, 1);
    std::cout << "\nAfter adding order:" << std::endl;
    ob.copy_to_host();
    print_orderbook(ob.get_asks(), ob.get_bids(), 100, "Before Cancel");
    
    // Cancel partial quantity
    Message cancel_msg;
    cancel_msg.type = Message::CANCEL;
    cancel_msg.side = Message::BID;
    cancel_msg.quantity = 50;
    cancel_msg.price = 100000;
    cancel_msg.order_id = 1001;
    cancel_msg.trader_id = 1;
    cancel_msg.time_sec = 34201;
    cancel_msg.time_ns = 0;
    
    ob.process_messages(&cancel_msg, 1);
    ob.print();
    
    // Verify
    const Order* bids = ob.get_bids();
    bool found = false;
    for (int i = 0; i < 100; i++) {
        if (bids[i].order_id == 1001) {
            found = true;
            if (bids[i].quantity != 50) {
                std::cout << "❌ FAIL: Quantity not reduced correctly (expected 50, got " 
                          << bids[i].quantity << ")" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!found) {
        std::cout << "❌ FAIL: Order disappeared" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Order cancelled correctly" << std::endl;
    return true;
}

bool test_simple_match() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 3: Simple Match" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    Message messages[2];
    
    // Add sell limit at 100100
    messages[0].type = Message::LIMIT;
    messages[0].side = Message::ASK;
    messages[0].quantity = 100;
    messages[0].price = 100100;
    messages[0].order_id = 1001;
    messages[0].trader_id = 1;
    messages[0].time_sec = 34200;
    messages[0].time_ns = 0;
    
    // Add buy limit at 100100 (should match!)
    messages[1].type = Message::LIMIT;
    messages[1].side = Message::BID;
    messages[1].quantity = 100;
    messages[1].price = 100100;
    messages[1].order_id = 1002;
    messages[1].trader_id = 2;
    messages[1].time_sec = 34200;
    messages[1].time_ns = 1;
    
    ob.process_messages(messages, 2);
    ob.print();
    
    // Verify trade occurred
    const Trade* trades = ob.get_trades();
    bool trade_found = false;
    for (int i = 0; i < 100; i++) {
        if (trades[i].price != EMPTY_PRICE) {
            trade_found = true;
            if (trades[i].price != 100100 || trades[i].quantity != 100) {
                std::cout << "❌ FAIL: Incorrect trade (price=" << trades[i].price 
                          << " qty=" << trades[i].quantity << ")" << std::endl;
                return false;
            }
            if (trades[i].passive_order_id != 1001 || trades[i].aggressive_order_id != 1002) {
                std::cout << "❌ FAIL: Incorrect order IDs in trade" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!trade_found) {
        std::cout << "❌ FAIL: No trade generated" << std::endl;
        return false;
    }
    
    // Verify orderbook is empty (full match)
    const Order* asks = ob.get_asks();
    const Order* bids = ob.get_bids();
    bool has_orders = false;
    for (int i = 0; i < 100; i++) {
        if (asks[i].price != EMPTY_PRICE || bids[i].price != EMPTY_PRICE) {
            has_orders = true;
            break;
        }
    }
    
    if (has_orders) {
        std::cout << "❌ FAIL: Orders remain after full match" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Match executed correctly" << std::endl;
    return true;
}

bool test_partial_match() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 4: Partial Match" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    Message messages[2];
    
    // Add sell limit 200 @ 100100
    messages[0].type = Message::LIMIT;
    messages[0].side = Message::ASK;
    messages[0].quantity = 200;
    messages[0].price = 100100;
    messages[0].order_id = 1001;
    messages[0].trader_id = 1;
    messages[0].time_sec = 34200;
    messages[0].time_ns = 0;
    
    // Add buy limit 100 @ 100100 (should match 100, leave 100)
    messages[1].type = Message::LIMIT;
    messages[1].side = Message::BID;
    messages[1].quantity = 100;
    messages[1].price = 100100;
    messages[1].order_id = 1002;
    messages[1].trader_id = 2;
    messages[1].time_sec = 34200;
    messages[1].time_ns = 1;
    
    ob.process_messages(messages, 2);
    ob.print();
    
    // Verify trade
    const Trade* trades = ob.get_trades();
    bool trade_found = false;
    for (int i = 0; i < 100; i++) {
        if (trades[i].price != EMPTY_PRICE) {
            trade_found = true;
            if (trades[i].quantity != 100) {
                std::cout << "❌ FAIL: Incorrect trade quantity (expected 100, got " 
                          << trades[i].quantity << ")" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!trade_found) {
        std::cout << "❌ FAIL: No trade generated" << std::endl;
        return false;
    }
    
    // Verify remaining ask order
    const Order* asks = ob.get_asks();
    bool found_remainder = false;
    for (int i = 0; i < 100; i++) {
        if (asks[i].order_id == 1001) {
            found_remainder = true;
            if (asks[i].quantity != 100) {
                std::cout << "❌ FAIL: Incorrect remainder (expected 100, got " 
                          << asks[i].quantity << ")" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!found_remainder) {
        std::cout << "❌ FAIL: Remainder order not found" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Partial match handled correctly" << std::endl;
    return true;
}

bool test_price_time_priority() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 5: Price-Time Priority" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    Message messages[4];
    
    // Add 3 asks at different prices/times
    messages[0].type = Message::LIMIT;
    messages[0].side = Message::ASK;
    messages[0].quantity = 50;
    messages[0].price = 100200;  // Higher price
    messages[0].order_id = 1001;
    messages[0].trader_id = 1;
    messages[0].time_sec = 34200;
    messages[0].time_ns = 0;
    
    messages[1].type = Message::LIMIT;
    messages[1].side = Message::ASK;
    messages[1].quantity = 50;
    messages[1].price = 100100;  // Best price, later time
    messages[1].order_id = 1002;
    messages[1].trader_id = 1;
    messages[1].time_sec = 34201;
    messages[1].time_ns = 0;
    
    messages[2].type = Message::LIMIT;
    messages[2].side = Message::ASK;
    messages[2].quantity = 50;
    messages[2].price = 100100;  // Best price, earlier time (should match first)
    messages[2].order_id = 1003;
    messages[2].trader_id = 1;
    messages[2].time_sec = 34200;
    messages[2].time_ns = 500;
    
    // Buy order that should match 1003 first (lowest price, earliest time)
    messages[3].type = Message::LIMIT;
    messages[3].side = Message::BID;
    messages[3].quantity = 50;
    messages[3].price = 100200;
    messages[3].order_id = 2001;
    messages[3].trader_id = 2;
    messages[3].time_sec = 34202;
    messages[3].time_ns = 0;
    
    ob.process_messages(messages, 4);
    ob.print();
    
    // Verify correct order was matched
    const Trade* trades = ob.get_trades();
    bool correct_match = false;
    for (int i = 0; i < 100; i++) {
        if (trades[i].price != EMPTY_PRICE) {
            if (trades[i].passive_order_id == 1003) {
                correct_match = true;
                std::cout << "\n✓ Correct order matched (1003 - earliest at best price)" << std::endl;
            } else {
                std::cout << "\n✗ Wrong order matched (got " << trades[i].passive_order_id 
                          << ", expected 1003)" << std::endl;
            }
            break;
        }
    }
    
    if (!correct_match) {
        std::cout << "❌ FAIL: Price-time priority not working" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Price-time priority correct" << std::endl;
    return true;
}

bool test_market_order() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 6: Market Order" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    OrderbookTest ob;
    
    Message messages[2];
    
    // Add sell limit
    messages[0].type = Message::LIMIT;
    messages[0].side = Message::ASK;
    messages[0].quantity = 100;
    messages[0].price = 100100;
    messages[0].order_id = 1001;
    messages[0].trader_id = 1;
    messages[0].time_sec = 34200;
    messages[0].time_ns = 0;
    
    // Market buy (should match at 100100)
    messages[1].type = Message::MARKET;
    messages[1].side = Message::BID;
    messages[1].quantity = 100;
    messages[1].price = 0;  // Ignored for market orders
    messages[1].order_id = 2001;
    messages[1].trader_id = 2;
    messages[1].time_sec = 34200;
    messages[1].time_ns = 1;
    
    ob.process_messages(messages, 2);
    ob.print();
    
    // Verify trade
    const Trade* trades = ob.get_trades();
    bool trade_found = false;
    for (int i = 0; i < 100; i++) {
        if (trades[i].price != EMPTY_PRICE) {
            trade_found = true;
            if (trades[i].price != 100100) {
                std::cout << "❌ FAIL: Incorrect execution price" << std::endl;
                return false;
            }
            break;
        }
    }
    
    if (!trade_found) {
        std::cout << "❌ FAIL: Market order did not execute" << std::endl;
        return false;
    }
    
    std::cout << "\n✅ PASS: Market order executed correctly" << std::endl;
    return true;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA ORDERBOOK MATCHING ENGINE TEST SUITE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nUsing GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Run tests
    int passed = 0;
    int total = 0;
    
    total++; if (test_add_order()) passed++;
    total++; if (test_cancel_order()) passed++;
    total++; if (test_simple_match()) passed++;
    total++; if (test_partial_match()) passed++;
    total++; if (test_price_time_priority()) passed++;
    total++; if (test_market_order()) passed++;
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "\nTests Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉" << std::endl;
        std::cout << "Matching engine is working correctly!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ SOME TESTS FAILED" << std::endl;
        std::cout << "Please review failed tests above" << std::endl;
        return 1;
    }
}

