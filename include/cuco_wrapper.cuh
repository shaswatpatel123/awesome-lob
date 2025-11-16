/**
 * cuCollections Wrapper
 * 
 * Provides unified interface for cuCollections static_map
 * Matches the API of simple_hash.cuh for seamless switching
 */

#ifndef CUDA_ORDERBOOK_CUCO_WRAPPER_H
#define CUDA_ORDERBOOK_CUCO_WRAPPER_H

#include "types.h"
#include <cuco/static_map.cuh>
#include <cuda_runtime.h>

namespace cuda_orderbook {

// Type alias for cuCollections map
using CucoMap = cuco::static_map<int32_t, int32_t>;

// Empty key sentinel for cuCollections
constexpr int32_t CUCO_EMPTY_KEY = -1;
constexpr int32_t CUCO_EMPTY_VALUE = -1;

// ============================================================================
// Host Functions (Memory Management)
// ============================================================================

/**
 * Create cuCollections static_map on device
 * 
 * @param capacity Initial capacity (cuCollections will round up)
 * @return Pointer to allocated map (cast to void* for storage in HashOrderbookState)
 */
inline void* cuco_create_host(int32_t capacity) {
    // cuCollections requires capacity to be at least 2x expected inserts
    // and prefers power-of-2 sizes
    int32_t cuco_capacity = capacity * 2;
    
    // Round up to next power of 2
    int32_t pow2_capacity = 1;
    while (pow2_capacity < cuco_capacity) {
        pow2_capacity <<= 1;
    }
    
    // Allocate map on device
    CucoMap* map = new CucoMap(
        pow2_capacity,
        cuco::empty_key{CUCO_EMPTY_KEY},
        cuco::empty_value{CUCO_EMPTY_VALUE}
    );
    
    return static_cast<void*>(map);
}

/**
 * Destroy cuCollections map
 */
inline void cuco_destroy_host(void* map_ptr) {
    if (map_ptr) {
        CucoMap* map = static_cast<CucoMap*>(map_ptr);
        delete map;
    }
}

// ============================================================================
// Device Functions (Map Operations)
// ============================================================================

/**
 * Insert key-value pair into cuCollections map
 * 
 * @param map_ptr Pointer to cuCollections map (void*)
 * @param key Order ID
 * @param value Array index
 * @return true if inserted/updated
 */
__device__ bool cuco_insert(
    void* map_ptr,
    int32_t key,
    int32_t value
) {
    CucoMap* map = static_cast<CucoMap*>(map_ptr);
    
    // Insert key-value pair (overwrites if key exists)
    auto result = map->insert(cuco::make_pair(key, value));
    
    return true;  // cuCollections insert always succeeds or updates
}

/**
 * Find value by key in cuCollections map
 * 
 * @param map_ptr Pointer to cuCollections map
 * @param key Order ID to lookup
 * @return Array index, or -1 if not found
 */
__device__ int32_t cuco_find(
    void* map_ptr,
    int32_t key
) {
    CucoMap* map = static_cast<CucoMap*>(map_ptr);
    
    // Find returns iterator-like object
    auto result = map->find(key);
    
    if (result != map->end()) {
        // Key found, return value
        return result->second;
    }
    
    return -1;  // Not found
}

/**
 * Erase key from cuCollections map
 * 
 * @param map_ptr Pointer to cuCollections map
 * @param key Order ID to remove
 * @return true if erased, false if not found
 */
__device__ bool cuco_erase(
    void* map_ptr,
    int32_t key
) {
    CucoMap* map = static_cast<CucoMap*>(map_ptr);
    
    // Erase key
    auto result = map->erase(key);
    
    return result > 0;  // Returns number of elements erased
}

/**
 * Check if map contains key
 */
__device__ bool cuco_contains(
    void* map_ptr,
    int32_t key
) {
    CucoMap* map = static_cast<CucoMap*>(map_ptr);
    return map->contains(key);
}

// ============================================================================
// Unified Wrapper Functions (Implementation-Agnostic)
// ============================================================================

/**
 * Unified insert that dispatches to appropriate implementation
 */
__device__ bool hash_map_insert(
    void* map_ptr,
    int32_t key,
    int32_t value,
    HashImplementation impl
) {
    if (impl == HASH_CUCOLLECTIONS) {
        return cuco_insert(map_ptr, key, value);
    } else {
        SimpleHashTable* table = static_cast<SimpleHashTable*>(map_ptr);
        return simple_hash_insert(table, key, value);
    }
}

/**
 * Unified find that dispatches to appropriate implementation
 */
__device__ int32_t hash_map_find(
    void* map_ptr,
    int32_t key,
    HashImplementation impl
) {
    if (impl == HASH_CUCOLLECTIONS) {
        return cuco_find(map_ptr, key);
    } else {
        SimpleHashTable* table = static_cast<SimpleHashTable*>(map_ptr);
        return simple_hash_find(table, key);
    }
}

/**
 * Unified erase that dispatches to appropriate implementation
 */
__device__ bool hash_map_erase(
    void* map_ptr,
    int32_t key,
    HashImplementation impl
) {
    if (impl == HASH_CUCOLLECTIONS) {
        return cuco_erase(map_ptr, key);
    } else {
        SimpleHashTable* table = static_cast<SimpleHashTable*>(map_ptr);
        return simple_hash_erase(table, key);
    }
}

/**
 * Unified contains check
 */
__device__ bool hash_map_contains(
    void* map_ptr,
    int32_t key,
    HashImplementation impl
) {
    if (impl == HASH_CUCOLLECTIONS) {
        return cuco_contains(map_ptr, key);
    } else {
        SimpleHashTable* table = static_cast<SimpleHashTable*>(map_ptr);
        return simple_hash_contains(table, key);
    }
}

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_CUCO_WRAPPER_H

