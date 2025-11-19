/**
 * CPU vs GPU Benchmark
 * 
 * Compares performance of CPU sequential implementation
 * against GPU CUDA implementation
 */

#include "orderbook_cpu.h"
#include "kernels.cuh"
#include "utils.cuh"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_orderbook;
using namespace std::chrono;

// ============================================================================
// MESSAGE GENERATION
// ============================================================================

std::vector<Message> generate_random_messages(
    int num_messages,
    int max_price = 1000,
    int max_quantity = 100,
    int seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> type_dist(1, 4);  // 1=LIMIT, 2=CANCEL, 3=DELETE, 4=MARKET
    std::uniform_int_distribution<int> side_dist(0, 1);  // 0=ASK, 1=BID
    std::uniform_int_distribution<int> price_dist(max_price / 2, max_price);
    std::uniform_int_distribution<int> qty_dist(1, max_quantity);
    
    std::vector<Message> messages(num_messages);
    
    for (int i = 0; i < num_messages; i++) {
        messages[i].type = Message::LIMIT;  // Start with LIMITs for testing
        messages[i].side = (side_dist(rng) == 0) ? Message::ASK : Message::BID;
        messages[i].price = price_dist(rng);
        messages[i].quantity = qty_dist(rng);
        messages[i].order_id = i + 1000;
        messages[i].trader_id = i % 10;
        messages[i].time_sec = i / 1000;
        messages[i].time_ns = (i % 1000) * 1000000;
    }
    
    return messages;
}

// ============================================================================
// BENCHMARK FUNCTIONS
// ============================================================================

double benchmark_cpu(
    int num_books,
    int num_messages_per_book,
    int n_orders_per_book,
    int n_trades_per_book,
    const std::vector<Message>& messages
) {
    std::cout << "\n=== CPU Benchmark ===" << std::endl;
    std::cout << "Allocating CPU memory..." << std::endl;
    
    // Allocate CPU batch
    OrderbookBatchCPU cpu_batch;
    if (!cpu_batch.allocate(num_books, n_orders_per_book, n_trades_per_book)) {
        std::cerr << "Failed to allocate CPU batch" << std::endl;
        return -1.0;
    }
    
    std::cout << "Processing messages on CPU..." << std::endl;
    
    // Benchmark CPU processing
    auto start = high_resolution_clock::now();
    
    process_messages_batch_cpu(cpu_batch, messages.data(), num_messages_per_book);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    double time_ms = duration.count() / 1000.0;
    
    std::cout << "CPU Time: " << time_ms << " ms" << std::endl;
    std::cout << "CPU Throughput: " 
              << (num_books * num_messages_per_book) / time_ms * 1000.0 
              << " messages/sec" << std::endl;
    
    return time_ms;
}

double benchmark_gpu(
    int num_books,
    int num_messages_per_book,
    int n_orders_per_book,
    int n_trades_per_book,
    const std::vector<Message>& messages
) {
    std::cout << "\n=== GPU Benchmark ===" << std::endl;
    std::cout << "Allocating GPU memory..." << std::endl;
    
    // Allocate GPU batch
    OrderbookBatch gpu_batch;
    if (!allocate_orderbook_batch(gpu_batch, num_books, n_orders_per_book, n_trades_per_book)) {
        std::cerr << "Failed to allocate GPU batch" << std::endl;
        return -1.0;
    }
    
    // Allocate host memory for messages
    if (!allocate_host_orderbook_batch(gpu_batch, num_books, n_orders_per_book, n_trades_per_book)) {
        std::cerr << "Failed to allocate host batch" << std::endl;
        free_orderbook_batch(gpu_batch);
        return -1.0;
    }
    
    // Initialize GPU orderbooks
    std::cout << "Initializing GPU orderbooks..." << std::endl;
    init_orderbooks_device(gpu_batch);
    
    // Allocate device memory for messages
    Message* d_messages;
    size_t messages_size = num_books * num_messages_per_book * sizeof(Message);
    CHECK_CUDA_ERROR(cudaMalloc(&d_messages, messages_size));
    
    // Copy messages to GPU
    std::cout << "Copying messages to GPU..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(d_messages, messages.data(), messages_size, cudaMemcpyHostToDevice));
    
    // Warm-up run
    std::cout << "Warm-up run..." << std::endl;
    dim3 grid((num_books + 255) / 256);
    dim3 block(256);
    process_messages_sequential_kernel<<<grid, block>>>(
        gpu_batch,
        d_messages,
        num_messages_per_book,
        num_books
    );
    cudaDeviceSynchronize();
    
    std::cout << "Processing messages on GPU..." << std::endl;
    
    // Benchmark GPU processing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    process_messages_sequential_kernel<<<grid, block>>>(
        gpu_batch,
        d_messages,
        num_messages_per_book,
        num_books
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_messages);
        free_orderbook_batch(gpu_batch);
        free_host_orderbook_batch(gpu_batch);
        return -1.0;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Check for synchronization errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Sync Error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_messages);
        free_orderbook_batch(gpu_batch);
        free_host_orderbook_batch(gpu_batch);
        return -1.0;
    }
    
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    // Verify GPU actually processed data (sample check)
    if (num_books == 1) {
        Order sample_order;
        CHECK_CUDA_ERROR(cudaMemcpy(&sample_order, gpu_batch.d_bids, sizeof(Order), cudaMemcpyDeviceToHost));
        if (sample_order.price == 0 && sample_order.quantity == 0 && sample_order.order_id == 0) {
            std::cerr << "WARNING: GPU results appear to be uninitialized (all zeros)!" << std::endl;
            std::cerr << "This suggests the kernel may not have executed correctly." << std::endl;
        }
    }
    
    std::cout << "GPU Time: " << time_ms << " ms" << std::endl;
    std::cout << "GPU Throughput: " 
              << (num_books * num_messages_per_book) / time_ms * 1000.0 
              << " messages/sec" << std::endl;
    
    // Warn if time seems suspiciously low
    if (time_ms < 0.01 && (num_books * num_messages_per_book) > 100) {
        std::cerr << "WARNING: GPU time is suspiciously low. " << std::endl;
        std::cerr << "This may indicate the kernel did not execute properly." << std::endl;
    }
    
    // Cleanup
    cudaFree(d_messages);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_orderbook_batch(gpu_batch);
    free_host_orderbook_batch(gpu_batch);
    
    return time_ms;
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "=== CPU vs GPU Orderbook Benchmark ===" << std::endl;
    
    // Benchmark parameters
    int num_books = (argc > 1) ? std::atoi(argv[1]) : 100;
    int num_messages_per_book = (argc > 2) ? std::atoi(argv[2]) : 1000;
    int n_orders_per_book = (argc > 3) ? std::atoi(argv[3]) : 100;
    int n_trades_per_book = (argc > 4) ? std::atoi(argv[4]) : 100;
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Number of orderbooks: " << num_books << std::endl;
    std::cout << "  Messages per orderbook: " << num_messages_per_book << std::endl;
    std::cout << "  Orders per side: " << n_orders_per_book << std::endl;
    std::cout << "  Max trades: " << n_trades_per_book << std::endl;
    std::cout << "  Total messages: " << num_books * num_messages_per_book << std::endl;
    
    // Generate test messages
    std::cout << "\nGenerating test messages..." << std::endl;
    auto messages = generate_random_messages(num_books * num_messages_per_book);
    
    // Run CPU benchmark
    double cpu_time_ms = benchmark_cpu(
        num_books,
        num_messages_per_book,
        n_orders_per_book,
        n_trades_per_book,
        messages
    );
    
    // Run GPU benchmark
    double gpu_time_ms = benchmark_gpu(
        num_books,
        num_messages_per_book,
        n_orders_per_book,
        n_trades_per_book,
        messages
    );
    
    // Print comparison
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "CPU Time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_ms << " ms" << std::endl;
    
    if (cpu_time_ms > 0 && gpu_time_ms > 0) {
        double speedup = cpu_time_ms / gpu_time_ms;
        std::cout << "GPU Speedup: " << speedup << "x" << std::endl;
        
        if (speedup > 1.0) {
            std::cout << "✓ GPU is " << speedup << "x faster than CPU!" << std::endl;
        } else {
            std::cout << "⚠ GPU is slower than CPU (speedup: " << speedup << "x)" << std::endl;
        }
    }
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    
    return 0;
}

