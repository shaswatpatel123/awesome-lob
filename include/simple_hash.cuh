/**
 * Simple CUDA Hash Table
 * 
 * Open addressing hash table with linear probing
 * Optimized for GPU execution with FNV-1a hash function
 * 
 * Key features:
 * - Power-of-2 sizing for fast modulo operations
 * - Linear probing for collision resolution
 * - Thread-safe insertions with atomicCAS
 * - Device-side operations
 */

#ifndef CUDA_ORDERBOOK_SIMPLE_HASH_H
#define CUDA_ORDERBOOK_SIMPLE_HASH_H

#include "types.h"
#include <cuda_runtime.h>

namespace cuda_orderbook {

// Empty slot marker
constexpr int32_t HASH_EMPTY = -1;

// ============================================================================
// Hash Functions
// ============================================================================

/**
 * FNV-1a hash function (32-bit)
 * Fast and good distribution for integer keys
 */
__device__ __forceinline__ uint32_t fnv1a_hash(int32_t key) {
    uint32_t hash = 2166136261u;  // FNV offset basis
    
    // Hash each byte of the key
    hash ^= (key & 0xFF);
    hash *= 16777619u;  // FNV prime
    
    hash ^= ((key >> 8) & 0xFF);
    hash *= 16777619u;
    
    hash ^= ((key >> 16) & 0xFF);
    hash *= 16777619u;
    
    hash ^= ((key >> 24) & 0xFF);
    hash *= 16777619u;
    
    return hash;
}

/**
 * Compute hash index with mask (fast modulo for power-of-2)
 */
__device__ __forceinline__ int32_t hash_index(int32_t key, int32_t mask) {
    return fnv1a_hash(key) & mask;
}

// ============================================================================
// Host Functions (Memory Management)
// ============================================================================

/**
 * Create and initialize a simple hash table on device
 * 
 * @param capacity Size of hash table (will be rounded up to power of 2)
 * @return Allocated hash table structure
 */
inline SimpleHashTable simple_hash_create_host(int32_t capacity) {
    SimpleHashTable table;
    
    // Round up to next power of 2
    int32_t pow2_capacity = 1;
    while (pow2_capacity < capacity) {
        pow2_capacity <<= 1;
    }
    
    table.capacity = pow2_capacity;
    table.mask = pow2_capacity - 1;
    table.size = 0;
    
    // Allocate device memory
    cudaMalloc(&table.keys, pow2_capacity * sizeof(int32_t));
    cudaMalloc(&table.values, pow2_capacity * sizeof(int32_t));
    
    // Initialize all keys to empty
    cudaMemset(table.keys, 0xFF, pow2_capacity * sizeof(int32_t));  // -1
    cudaMemset(table.values, 0xFF, pow2_capacity * sizeof(int32_t));
    
    return table;
}

/**
 * Destroy hash table and free memory
 */
inline void simple_hash_destroy_host(SimpleHashTable* table) {
    if (table->keys) cudaFree(table->keys);
    if (table->values) cudaFree(table->values);
    table->keys = nullptr;
    table->values = nullptr;
    table->capacity = 0;
    table->size = 0;
}

// ============================================================================
// Device Functions (Hash Operations)
// ============================================================================

/**
 * Insert key-value pair into hash table
 * Thread-safe using atomicCAS
 * 
 * @param table Hash table
 * @param key Order ID
 * @param value Array index
 * @return true if inserted, false if already exists
 */
__device__ bool simple_hash_insert(
    SimpleHashTable* table,
    int32_t key,
    int32_t value
) {
    int32_t idx = hash_index(key, table->mask);
    
    // Linear probing until we find empty slot or matching key
    for (int32_t i = 0; i < table->capacity; i++) {
        int32_t probe_idx = (idx + i) & table->mask;
        
        // Try to claim this slot with atomicCAS
        int32_t old_key = atomicCAS(&table->keys[probe_idx], HASH_EMPTY, key);
        
        if (old_key == HASH_EMPTY) {
            // Successfully claimed empty slot
            table->values[probe_idx] = value;
            atomicAdd(&table->size, 1);
            return true;
        } else if (old_key == key) {
            // Key already exists, update value
            table->values[probe_idx] = value;
            return true;
        }
        // else: slot occupied by different key, continue probing
    }
    
    // Hash table full (should not happen with proper sizing)
    return false;
}

/**
 * Find value by key
 * 
 * @param table Hash table
 * @param key Order ID to lookup
 * @return Array index, or -1 if not found
 */
__device__ int32_t simple_hash_find(
    const SimpleHashTable* table,
    int32_t key
) {
    int32_t idx = hash_index(key, table->mask);
    
    // Linear probing until we find key or empty slot
    for (int32_t i = 0; i < table->capacity; i++) {
        int32_t probe_idx = (idx + i) & table->mask;
        int32_t stored_key = table->keys[probe_idx];
        
        if (stored_key == key) {
            return table->values[probe_idx];
        } else if (stored_key == HASH_EMPTY) {
            // Empty slot means key not in table
            return -1;
        }
        // else: continue probing
    }
    
    return -1;  // Not found
}

/**
 * Erase key-value pair from hash table
 * Uses tombstone marking (sets value to -1, keeps key for probing)
 * 
 * @param table Hash table
 * @param key Order ID to remove
 * @return true if erased, false if not found
 */
__device__ bool simple_hash_erase(
    SimpleHashTable* table,
    int32_t key
) {
    int32_t idx = hash_index(key, table->mask);
    
    // Linear probing to find key
    for (int32_t i = 0; i < table->capacity; i++) {
        int32_t probe_idx = (idx + i) & table->mask;
        int32_t stored_key = table->keys[probe_idx];
        
        if (stored_key == key) {
            // Mark as erased (tombstone)
            table->keys[probe_idx] = HASH_EMPTY;
            table->values[probe_idx] = -1;
            atomicSub(&table->size, 1);
            return true;
        } else if (stored_key == HASH_EMPTY) {
            // Empty slot means key not in table
            return false;
        }
    }
    
    return false;  // Not found
}

/**
 * Check if hash table contains key
 */
__device__ bool simple_hash_contains(
    const SimpleHashTable* table,
    int32_t key
) {
    return simple_hash_find(table, key) != -1;
}

/**
 * Clear all entries in hash table
 */
__device__ void simple_hash_clear(SimpleHashTable* table) {
    // Reset all keys to empty
    for (int32_t i = 0; i < table->capacity; i++) {
        table->keys[i] = HASH_EMPTY;
        table->values[i] = -1;
    }
    table->size = 0;
}

// ============================================================================
// Kernel for Bulk Operations
// ============================================================================

/**
 * Initialize hash table kernel (parallel clear)
 */
__global__ void simple_hash_init_kernel(
    SimpleHashTable* table
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < table->capacity) {
        table->keys[idx] = HASH_EMPTY;
        table->values[idx] = -1;
    }
    
    if (idx == 0) {
        table->size = 0;
    }
}

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_SIMPLE_HASH_H

