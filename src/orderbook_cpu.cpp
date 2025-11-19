/**
 * CPU Sequential Orderbook Implementation
 * 
 * Pure CPU implementation for benchmarking against GPU version
 * Provides baseline performance measurements
 */

#include "orderbook_cpu.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cuda_orderbook {

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

bool OrderbookCPU::allocate(int n_orders, int n_trades_max) {
    n_orders_per_side = n_orders;
    n_trades = n_trades_max;
    
    asks = new Order[n_orders];
    bids = new Order[n_orders];
    trades = new Trade[n_trades_max];
    
    ask_metadata = new OrderMetadata[n_orders];
    ask_buckets = new PriceBucket[n_price_buckets];
    ask_price_map = new PriceMapEntry[price_map_size];
    ask_order_id_map = new OrderIDMapEntry[order_id_map_size];
    
    bid_metadata = new OrderMetadata[n_orders];
    bid_buckets = new PriceBucket[n_price_buckets];
    bid_price_map = new PriceMapEntry[price_map_size];
    bid_order_id_map = new OrderIDMapEntry[order_id_map_size];
    
    if (!asks || !bids || !trades ||
        !ask_metadata || !ask_buckets || !ask_price_map || !ask_order_id_map ||
        !bid_metadata || !bid_buckets || !bid_price_map || !bid_order_id_map) {
        cleanup();
        return false;
    }
    
    initialize();
    return true;
}

void OrderbookCPU::cleanup() {
    if (asks) {
        delete[] asks;
        asks = nullptr;
    }
    if (bids) {
        delete[] bids;
        bids = nullptr;
    }
    if (trades) {
        delete[] trades;
        trades = nullptr;
    }
    if (ask_metadata) {
        delete[] ask_metadata;
        ask_metadata = nullptr;
    }
    if (ask_buckets) {
        delete[] ask_buckets;
        ask_buckets = nullptr;
    }
    if (ask_price_map) {
        delete[] ask_price_map;
        ask_price_map = nullptr;
    }
    if (ask_order_id_map) {
        delete[] ask_order_id_map;
        ask_order_id_map = nullptr;
    }
    if (bid_metadata) {
        delete[] bid_metadata;
        bid_metadata = nullptr;
    }
    if (bid_buckets) {
        delete[] bid_buckets;
        bid_buckets = nullptr;
    }
    if (bid_price_map) {
        delete[] bid_price_map;
        bid_price_map = nullptr;
    }
    if (bid_order_id_map) {
        delete[] bid_order_id_map;
        bid_order_id_map = nullptr;
    }
}

void OrderbookCPU::initialize() {
    // Initialize asks
    for (int i = 0; i < n_orders_per_side; i++) {
        asks[i].price = EMPTY_PRICE;
        asks[i].quantity = 0;
        asks[i].order_id = 0;
        asks[i].trader_id = 0;
        asks[i].time_sec = 0;
        asks[i].time_ns = 0;
        if (ask_metadata) {
            ask_metadata[i] = OrderMetadata();
        }
    }
    
    // Initialize bids
    for (int i = 0; i < n_orders_per_side; i++) {
        bids[i].price = EMPTY_PRICE;
        bids[i].quantity = 0;
        bids[i].order_id = 0;
        bids[i].trader_id = 0;
        bids[i].time_sec = 0;
        bids[i].time_ns = 0;
        if (bid_metadata) {
            bid_metadata[i] = OrderMetadata();
        }
    }
    
    // Initialize trades
    for (int i = 0; i < n_trades; i++) {
        trades[i].price = EMPTY_PRICE;
        trades[i].quantity = 0;
        trades[i].passive_order_id = 0;
        trades[i].aggressive_order_id = 0;
        trades[i].time_sec = 0;
        trades[i].time_ns = 0;
    }
    
    // Initialize price buckets
    if (ask_buckets) {
        for (int i = 0; i < n_price_buckets; i++) {
            ask_buckets[i] = PriceBucket();
        }
    }
    if (bid_buckets) {
        for (int i = 0; i < n_price_buckets; i++) {
            bid_buckets[i] = PriceBucket();
        }
    }
    
    // Initialize price maps
    if (ask_price_map) {
        for (int i = 0; i < price_map_size; i++) {
            ask_price_map[i] = PriceMapEntry();
        }
    }
    if (bid_price_map) {
        for (int i = 0; i < price_map_size; i++) {
            bid_price_map[i] = PriceMapEntry();
        }
    }
    
    // Initialize order-ID maps
    if (ask_order_id_map) {
        for (int i = 0; i < order_id_map_size; i++) {
            ask_order_id_map[i] = OrderIDMapEntry();
        }
    }
    if (bid_order_id_map) {
        for (int i = 0; i < order_id_map_size; i++) {
            bid_order_id_map[i] = OrderIDMapEntry();
        }
    }
    
    // Initialize best price trackers
    ask_tracker = BestPriceTracker();
    bid_tracker = BestPriceTracker();
}

bool OrderbookBatchCPU::allocate(int n_books, int n_orders_per_book, int n_trades_per_book) {
    num_books = n_books;
    books = new OrderbookCPU[n_books];
    
    if (!books) {
        return false;
    }
    
    // Allocate each orderbook
    for (int i = 0; i < n_books; i++) {
        if (!books[i].allocate(n_orders_per_book, n_trades_per_book)) {
            cleanup();
            return false;
        }
    }
    
    return true;
}

void OrderbookBatchCPU::cleanup() {
    if (books) {
        delete[] books;
        books = nullptr;
    }
}

void OrderbookBatchCPU::initialize() {
    for (int i = 0; i < num_books; i++) {
        books[i].initialize();
    }
}

// ============================================================================
// PRICE-AWARE HELPER FUNCTIONS (CPU)
// ============================================================================

int32_t hash_price_cpu(int32_t price, int32_t map_size) {
    uint32_t hash = (uint32_t)(price) * 2654435761U; // Knuth's multiplicative hash
    return (int32_t)(hash % (uint32_t)map_size);
}

int32_t hash_order_id_cpu(int32_t order_id, int32_t map_size) {
    uint32_t hash = (uint32_t)(order_id) * 2654435761U;
    return (int32_t)(hash % (uint32_t)map_size);
}

int32_t find_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t map_size) {
    if (price == EMPTY_PRICE) return EMPTY_INDEX;
    
    int32_t start_idx = hash_price_cpu(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].price == price && price_map[idx].is_active) {
            return price_map[idx].bucket_idx;
        }
        if (price_map[idx].is_empty()) {
            return EMPTY_INDEX;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return EMPTY_INDEX;  // Not found
}

bool insert_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t bucket_idx, int32_t map_size) {
    if (price == EMPTY_PRICE) return false;
    
    int32_t start_idx = hash_price_cpu(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].is_empty() || price_map[idx].is_tombstone() ||
            (price_map[idx].price == price && price_map[idx].is_active)) {
            price_map[idx].price = price;
            price_map[idx].bucket_idx = bucket_idx;
            price_map[idx].is_active = true;
            price_map[idx].was_deleted = false;
            return true;
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return false;  // Map is full
}

void remove_price_bucket_cpu(PriceMapEntry* price_map, int32_t price, int32_t map_size) {
    if (price == EMPTY_PRICE) return;
    
    int32_t start_idx = hash_price_cpu(price, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (price_map[idx].price == price && price_map[idx].is_active) {
            price_map[idx].is_active = false;
            price_map[idx].was_deleted = true;
            price_map[idx].price = EMPTY_PRICE;
            price_map[idx].bucket_idx = EMPTY_INDEX;
            return;
        }
        if (price_map[idx].is_empty()) {
            return;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
}

int32_t find_order_by_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t map_size) {
    if (order_id == 0) return EMPTY_INDEX;
    
    int32_t start_idx = hash_order_id_cpu(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active) {
            return order_id_map[idx].order_idx;
        }
        if (order_id_map[idx].is_empty()) {
            return EMPTY_INDEX;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return EMPTY_INDEX;  // Not found
}

bool insert_order_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t order_idx, int32_t map_size) {
    if (order_id == 0) return false;
    
    int32_t start_idx = hash_order_id_cpu(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].is_empty() || order_id_map[idx].is_tombstone() ||
            (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active)) {
            order_id_map[idx].order_id = order_id;
            order_id_map[idx].order_idx = order_idx;
            order_id_map[idx].is_active = true;
            order_id_map[idx].was_deleted = false;
            return true;
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
    
    return false;  // Map is full
}

void remove_order_id_map_cpu(OrderIDMapEntry* order_id_map, int32_t order_id, int32_t map_size) {
    if (order_id == 0) return;
    
    int32_t start_idx = hash_order_id_cpu(order_id, map_size);
    int32_t idx = start_idx;
    
    // Linear probing
    do {
        if (order_id_map[idx].order_id == order_id && order_id_map[idx].is_active) {
            order_id_map[idx].is_active = false;
            order_id_map[idx].was_deleted = true;
            order_id_map[idx].order_id = 0;
            order_id_map[idx].order_idx = EMPTY_INDEX;
            return;
        }
        if (order_id_map[idx].is_empty()) {
            return;  // Not found
        }
        idx = (idx + 1) % map_size;
    } while (idx != start_idx);
}

int32_t get_or_create_price_bucket_cpu(PriceBucket* buckets, PriceMapEntry* price_map, int32_t price, int32_t n_buckets, int32_t map_size) {
    if (price == EMPTY_PRICE) return EMPTY_INDEX;
    
    // First try to find existing bucket
    int32_t bucket_idx = find_price_bucket_cpu(price_map, price, map_size);
    if (bucket_idx != EMPTY_INDEX) {
        return bucket_idx;
    }
    
    // Find empty bucket slot
    for (int32_t i = 0; i < n_buckets; i++) {
        if (!buckets[i].is_active) {
            buckets[i].price = price;
            buckets[i].is_active = true;
            buckets[i].head_idx = EMPTY_INDEX;
            buckets[i].tail_idx = EMPTY_INDEX;
            buckets[i].total_quantity = 0;
            
            // Insert into price map
            if (insert_price_bucket_cpu(price_map, price, i, map_size)) {
                return i;
            }
            // Map insert failed, mark bucket as inactive again
            buckets[i].is_active = false;
            return EMPTY_INDEX;
        }
    }
    
    return EMPTY_INDEX;  // No free buckets
}

void add_order_to_bucket_cpu(PriceBucket* buckets, OrderMetadata* metadata, Order* orders, int32_t bucket_idx, int32_t order_idx) {
    if (bucket_idx == EMPTY_INDEX || order_idx == EMPTY_INDEX) return;
    
    PriceBucket& bucket = buckets[bucket_idx];
    OrderMetadata& meta = metadata[order_idx];
    Order& order = orders[order_idx];
    
    meta.price_bucket_idx = bucket_idx;
    meta.is_valid = true;
    
    // Add to tail (FIFO)
    if (bucket.is_empty()) {
        // First order at this price
        bucket.head_idx = order_idx;
        bucket.tail_idx = order_idx;
        meta.next_idx = EMPTY_INDEX;
        meta.prev_idx = EMPTY_INDEX;
    } else {
        // Add to tail
        int32_t old_tail = bucket.tail_idx;
        OrderMetadata& old_tail_meta = metadata[old_tail];
        
        old_tail_meta.next_idx = order_idx;
        meta.prev_idx = old_tail;
        meta.next_idx = EMPTY_INDEX;
        bucket.tail_idx = order_idx;
    }
    
    bucket.total_quantity += order.quantity;
}

void remove_order_from_bucket_cpu(PriceBucket* buckets, OrderMetadata* metadata, Order* orders, int32_t bucket_idx, int32_t order_idx, int32_t removed_quantity) {
    if (bucket_idx == EMPTY_INDEX || order_idx == EMPTY_INDEX) return;
    
    PriceBucket& bucket = buckets[bucket_idx];
    OrderMetadata& meta = metadata[order_idx];
    (void)orders;
    
    // Remove from linked list
    if (meta.prev_idx != EMPTY_INDEX) {
        metadata[meta.prev_idx].next_idx = meta.next_idx;
    } else {
        // This was head
        bucket.head_idx = meta.next_idx;
    }
    
    if (meta.next_idx != EMPTY_INDEX) {
        metadata[meta.next_idx].prev_idx = meta.prev_idx;
    } else {
        // This was tail
        bucket.tail_idx = meta.prev_idx;
    }
    
    int32_t qty = std::max(0, removed_quantity);
    bucket.total_quantity = std::max(0, bucket.total_quantity - qty);
    
    // Clear metadata
    meta.next_idx = EMPTY_INDEX;
    meta.prev_idx = EMPTY_INDEX;
    meta.price_bucket_idx = EMPTY_INDEX;
    meta.is_valid = false;
    
    // If bucket is now empty, mark as inactive
    if (bucket.is_empty()) {
        bucket.is_active = false;
        bucket.price = EMPTY_PRICE;
        bucket.total_quantity = 0;
    }
}

void update_best_ask_price_cpu(PriceBucket* buckets, PriceMapEntry* price_map, BestPriceTracker* tracker, int32_t n_buckets, int32_t map_size) {
    int32_t best_price = MAX_INT;
    int32_t best_bucket_idx = EMPTY_INDEX;
    
    // Scan all active buckets to find minimum price
    for (int32_t i = 0; i < n_buckets; i++) {
        if (buckets[i].is_active && !buckets[i].is_empty()) {
            if (buckets[i].price < best_price) {
                best_price = buckets[i].price;
                best_bucket_idx = i;
            }
        }
    }
    
    tracker->best_ask_price = best_price;
    tracker->best_ask_bucket_idx = best_bucket_idx;
}

void update_best_bid_price_cpu(PriceBucket* buckets, PriceMapEntry* price_map, BestPriceTracker* tracker, int32_t n_buckets, int32_t map_size) {
    int32_t best_price = EMPTY_PRICE;
    int32_t best_bucket_idx = EMPTY_INDEX;
    
    // Scan all active buckets to find maximum price
    for (int32_t i = 0; i < n_buckets; i++) {
        if (buckets[i].is_active && !buckets[i].is_empty()) {
            if (buckets[i].price > best_price) {
                best_price = buckets[i].price;
                best_bucket_idx = i;
            }
        }
    }
    
    tracker->best_bid_price = best_price;
    tracker->best_bid_bucket_idx = best_bucket_idx;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void remove_zero_neg_quant_cpu(OrderbookCPU& book, bool is_ask_side) {
    Order* orderside = is_ask_side ? book.asks : book.bids;
    OrderMetadata* metadata = is_ask_side ? book.ask_metadata : book.bid_metadata;
    PriceBucket* buckets = is_ask_side ? book.ask_buckets : book.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? book.ask_price_map : book.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? book.ask_order_id_map : book.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? &book.ask_tracker : &book.bid_tracker;
    
    for (int i = 0; i < book.n_orders_per_side; i++) {
        if (orderside[i].quantity <= 0 && orderside[i].price != EMPTY_PRICE) {
            int32_t bucket_idx = metadata[i].price_bucket_idx;
            int32_t removed_qty = std::max(0, orderside[i].quantity);
            
            // Remove from price bucket
            if (bucket_idx != EMPTY_INDEX) {
                remove_order_from_bucket_cpu(buckets, metadata, orderside, bucket_idx, i, removed_qty);
                
                // If bucket is now empty, remove from price map
                if (buckets[bucket_idx].is_empty()) {
                    remove_price_bucket_cpu(price_map, orderside[i].price, book.price_map_size);
                }
                
                // Update best price tracker if needed
                bool need_update = false;
                if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                    need_update = true;
                } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                    need_update = true;
                }
                
                if (need_update) {
                    if (is_ask_side) {
                        update_best_ask_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                    } else {
                        update_best_bid_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                    }
                }
            }
            
            // Remove from order-ID map
            remove_order_id_map_cpu(order_id_map, orderside[i].order_id, book.order_id_map_size);
            
            // Mark as empty
            orderside[i].price = EMPTY_PRICE;
            orderside[i].quantity = 0;
            orderside[i].order_id = 0;
            orderside[i].trader_id = 0;
            orderside[i].time_sec = 0;
            orderside[i].time_ns = 0;
        }
    }
}

int get_top_ask_order_idx_cpu(const OrderbookCPU& book) {
    if (!book.ask_tracker.has_best_ask()) {
        return EMPTY_INDEX;
    }
    
    int32_t bucket_idx = book.ask_tracker.best_ask_bucket_idx;
    if (bucket_idx == EMPTY_INDEX || bucket_idx >= book.n_price_buckets) {
        return EMPTY_INDEX;
    }
    
    PriceBucket& bucket = book.ask_buckets[bucket_idx];
    if (bucket.is_empty()) {
        return EMPTY_INDEX;
    }
    
    return bucket.head_idx;  // First order at best price (FIFO)
}

int get_top_bid_order_idx_cpu(const OrderbookCPU& book) {
    if (!book.bid_tracker.has_best_bid()) {
        return EMPTY_INDEX;
    }
    
    int32_t bucket_idx = book.bid_tracker.best_bid_bucket_idx;
    if (bucket_idx == EMPTY_INDEX || bucket_idx >= book.n_price_buckets) {
        return EMPTY_INDEX;
    }
    
    PriceBucket& bucket = book.bid_buckets[bucket_idx];
    if (bucket.is_empty()) {
        return EMPTY_INDEX;
    }
    
    return bucket.head_idx;  // First order at best price (FIFO)
}

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

void add_order_cpu(OrderbookCPU& book, bool is_ask_side, const Message& msg) {
    if (msg.price == EMPTY_PRICE || msg.quantity <= 0) return;
    
    // Select appropriate side structures
    Order* orderside = is_ask_side ? book.asks : book.bids;
    OrderMetadata* metadata = is_ask_side ? book.ask_metadata : book.bid_metadata;
    PriceBucket* buckets = is_ask_side ? book.ask_buckets : book.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? book.ask_price_map : book.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? book.ask_order_id_map : book.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? &book.ask_tracker : &book.bid_tracker;
    
    // Find first empty slot
    int empty_idx = -1;
    for (int i = 0; i < book.n_orders_per_side; i++) {
        if (orderside[i].price == EMPTY_PRICE) {
            empty_idx = i;
            break;
        }
    }
    
    if (empty_idx == -1) {
        // Orderbook full - cannot add
        return;
    }
    
    // Add the order
    Order& order = orderside[empty_idx];
    order.price = msg.price;
    order.quantity = std::max(0, msg.quantity);
    order.order_id = msg.order_id;
    order.trader_id = msg.trader_id;
    order.time_sec = msg.time_sec;
    order.time_ns = msg.time_ns;
    
    // Get or create price bucket
    int32_t bucket_idx = get_or_create_price_bucket_cpu(
        buckets, price_map, msg.price, 
        book.n_price_buckets, book.price_map_size
    );
    
    if (bucket_idx == EMPTY_INDEX) {
        // Cannot create bucket - revert order
        order.price = EMPTY_PRICE;
        return;
    }
    
    // Add order to price bucket (at tail for FIFO)
    add_order_to_bucket_cpu(buckets, metadata, orderside, bucket_idx, empty_idx);
    
    // Insert into order-ID map for O(1) cancel lookup
    insert_order_id_map_cpu(order_id_map, msg.order_id, empty_idx, book.order_id_map_size);
    
    // Update best price tracker
    if (is_ask_side) {
        // For asks, update if this is a better (lower) price
        if (msg.price < tracker->best_ask_price) {
            tracker->best_ask_price = msg.price;
            tracker->best_ask_bucket_idx = bucket_idx;
        }
    } else {
        // For bids, update if this is a better (higher) price
        if (msg.price > tracker->best_bid_price) {
            tracker->best_bid_price = msg.price;
            tracker->best_bid_bucket_idx = bucket_idx;
        }
    }
}

void cancel_order_cpu(OrderbookCPU& book, bool is_ask_side, const Message& msg) {
    // Select appropriate side structures
    Order* orderside = is_ask_side ? book.asks : book.bids;
    OrderMetadata* metadata = is_ask_side ? book.ask_metadata : book.bid_metadata;
    PriceBucket* buckets = is_ask_side ? book.ask_buckets : book.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? book.ask_price_map : book.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? book.ask_order_id_map : book.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? &book.ask_tracker : &book.bid_tracker;
    
    // Find order by ID using hash map (O(1))
    int32_t idx = find_order_by_id_map_cpu(order_id_map, msg.order_id, book.order_id_map_size);
    
    // If not found and this might be an INITID order, search by price
    if (idx == EMPTY_INDEX) {
        for (int i = 0; i < book.n_orders_per_side; i++) {
            if (orderside[i].price == msg.price && 
                orderside[i].order_id <= INITID &&
                orderside[i].price != EMPTY_PRICE) {
                idx = i;
                break;
            }
        }
    }
    
    if (idx == EMPTY_INDEX || idx >= book.n_orders_per_side) {
        // Order not found
        return;
    }
    
    Order& order = orderside[idx];
    if (order.price == EMPTY_PRICE) return;
    
    int32_t old_quantity = order.quantity;
    
    // Reduce quantity
    order.quantity = std::max(0, order.quantity - msg.quantity);
    
    // If fully cancelled, remove from structures
    if (order.quantity <= 0) {
        // Get bucket index from metadata
        int32_t bucket_idx = metadata[idx].price_bucket_idx;
        
        // Remove from price bucket
        if (bucket_idx != EMPTY_INDEX) {
            remove_order_from_bucket_cpu(buckets, metadata, orderside, bucket_idx, idx, old_quantity);
            
            // If bucket is now empty, remove from price map
            if (buckets[bucket_idx].is_empty()) {
                remove_price_bucket_cpu(price_map, order.price, book.price_map_size);
            }
            
            // Update best price tracker if we removed the best price
            bool need_update = false;
            if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                need_update = true;
            } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                need_update = true;
            }
            
            if (need_update) {
                if (is_ask_side) {
                    update_best_ask_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                } else {
                    update_best_bid_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                }
            }
        }
        
        // Remove from order-ID map
        remove_order_id_map_cpu(order_id_map, msg.order_id, book.order_id_map_size);
        
        // Clear order
        order.price = EMPTY_PRICE;
        order.quantity = 0;
        order.order_id = 0;
        order.trader_id = 0;
        order.time_sec = 0;
        order.time_ns = 0;
    } else {
        // Partial cancel - update bucket quantity
        int32_t bucket_idx = metadata[idx].price_bucket_idx;
        if (bucket_idx != EMPTY_INDEX) {
            int32_t qty_delta = old_quantity - order.quantity;
            buckets[bucket_idx].total_quantity = std::max(0, buckets[bucket_idx].total_quantity - qty_delta);
        }
    }
}

void match_single_order_cpu(
    OrderbookCPU& book,
    bool is_ask_side,
    int top_order_idx,
    int32_t& qtm_remaining,
    int32_t aggressive_order_id,
    int32_t time_sec,
    int32_t time_ns
) {
    if (top_order_idx < 0 || top_order_idx >= book.n_orders_per_side) return;
    if (qtm_remaining <= 0) return;
    
    Order* orderside = is_ask_side ? book.asks : book.bids;
    OrderMetadata* metadata = is_ask_side ? book.ask_metadata : book.bid_metadata;
    PriceBucket* buckets = is_ask_side ? book.ask_buckets : book.bid_buckets;
    PriceMapEntry* price_map = is_ask_side ? book.ask_price_map : book.bid_price_map;
    OrderIDMapEntry* order_id_map = is_ask_side ? book.ask_order_id_map : book.bid_order_id_map;
    BestPriceTracker* tracker = is_ask_side ? &book.ask_tracker : &book.bid_tracker;
    
    Order& passive_order = orderside[top_order_idx];
    if (passive_order.price == EMPTY_PRICE) return;
    
    // Calculate matched quantity
    int32_t matched_qty = std::min(qtm_remaining, passive_order.quantity);
    int32_t old_quantity = passive_order.quantity;
    int32_t new_quantity = std::max(0, passive_order.quantity - matched_qty);
    
    // Update remaining quantity to match
    qtm_remaining = std::max(0, qtm_remaining - matched_qty);
    
    // Find empty trade slot and record trade
    for (int i = 0; i < book.n_trades; i++) {
        if (book.trades[i].price == EMPTY_PRICE) {
            book.trades[i].price = passive_order.price;
            book.trades[i].quantity = matched_qty;
            book.trades[i].passive_order_id = passive_order.order_id;
            book.trades[i].aggressive_order_id = aggressive_order_id;
            book.trades[i].time_sec = time_sec;
            book.trades[i].time_ns = time_ns;
            break;
        }
    }
    
    // Update passive order quantity
    passive_order.quantity = new_quantity;
    
    int32_t bucket_idx = metadata[top_order_idx].price_bucket_idx;
    
    // If fully matched, remove from structures
    if (new_quantity <= 0) {
        // Remove from price bucket
        if (bucket_idx != EMPTY_INDEX) {
            remove_order_from_bucket_cpu(buckets, metadata, orderside, bucket_idx, top_order_idx, old_quantity);
            
            // If bucket is now empty, remove from price map
            if (buckets[bucket_idx].is_empty()) {
                remove_price_bucket_cpu(price_map, passive_order.price, book.price_map_size);
            }
            
            // Update best price tracker if we removed the best price
            bool need_update = false;
            if (is_ask_side && bucket_idx == tracker->best_ask_bucket_idx) {
                need_update = true;
            } else if (!is_ask_side && bucket_idx == tracker->best_bid_bucket_idx) {
                need_update = true;
            }
            
            if (need_update) {
                if (is_ask_side) {
                    update_best_ask_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                } else {
                    update_best_bid_price_cpu(buckets, price_map, tracker, book.n_price_buckets, book.price_map_size);
                }
            }
        }
        
        // Remove from order-ID map
        remove_order_id_map_cpu(order_id_map, passive_order.order_id, book.order_id_map_size);
        
        // Clear order
        passive_order.price = EMPTY_PRICE;
        passive_order.order_id = 0;
        passive_order.trader_id = 0;
        passive_order.time_sec = 0;
        passive_order.time_ns = 0;
    } else {
        // Partial match - update bucket quantity
        if (bucket_idx != EMPTY_INDEX) {
            int32_t qty_delta = old_quantity - new_quantity;
            buckets[bucket_idx].total_quantity = std::max(0, buckets[bucket_idx].total_quantity - qty_delta);
        }
    }
}

void match_against_asks_cpu(OrderbookCPU& book, const Message& msg) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid asks
    while (qtm_remaining > 0) {
        // Get best ask using price-aware lookup (O(1))
        int top_ask_idx = get_top_ask_order_idx_cpu(book);
        
        // Check if we can match
        if (top_ask_idx == EMPTY_INDEX) break;  // No asks available
        if (book.asks[top_ask_idx].price == EMPTY_PRICE) break;  // No valid ask
        if (book.asks[top_ask_idx].price > limit_price) break;  // Price too high
        
        // Match against this ask
        match_single_order_cpu(
            book,
            true,  // is_ask_side
            top_ask_idx,
            qtm_remaining,
            msg.order_id,
            msg.time_sec,
            msg.time_ns
        );
    }
}

void match_against_bids_cpu(OrderbookCPU& book, const Message& msg) {
    int32_t qtm_remaining = msg.quantity;
    int32_t limit_price = msg.price;
    
    // Keep matching while we have quantity and valid bids
    while (qtm_remaining > 0) {
        // Get best bid using price-aware lookup (O(1))
        int top_bid_idx = get_top_bid_order_idx_cpu(book);
        
        // Check if we can match
        if (top_bid_idx == EMPTY_INDEX) break;  // No bids available
        if (book.bids[top_bid_idx].price == EMPTY_PRICE) break;  // No valid bid
        if (book.bids[top_bid_idx].price < limit_price) break;  // Price too low
        
        // Match against this bid
        match_single_order_cpu(
            book,
            false,  // is_ask_side
            top_bid_idx,
            qtm_remaining,
            msg.order_id,
            msg.time_sec,
            msg.time_ns
        );
    }
}

void process_message_cpu(OrderbookCPU& book, const Message& msg) {
    // Determine action based on type and side
    // Type: 1=limit, 2=cancel, 3=delete, 4=market
    // Side: -1=ask, 1=bid
    
    if (msg.type == Message::CANCEL || msg.type == Message::DELETE) {
        // Cancel order
        if (msg.side == Message::ASK) {
            cancel_order_cpu(book, true, msg);  // is_ask_side = true
        } else if (msg.side == Message::BID) {
            cancel_order_cpu(book, false, msg);  // is_ask_side = false
        }
    }
    else if (msg.type == Message::LIMIT) {
        // Limit order - need to track remaining quantity after matching
        if (msg.side == Message::ASK) {
            // Sell limit: match against bids, then add remainder
            int32_t qtm_remaining = msg.quantity;
            int32_t limit_price = msg.price;
            
            // Keep matching while we have quantity and valid bids
            while (qtm_remaining > 0) {
                int top_bid_idx = get_top_bid_order_idx_cpu(book);
                if (top_bid_idx == EMPTY_INDEX) break;
                if (book.bids[top_bid_idx].price == EMPTY_PRICE) break;
                if (book.bids[top_bid_idx].price < limit_price) break;
                
                match_single_order_cpu(
                    book, false, top_bid_idx, qtm_remaining,
                    msg.order_id, msg.time_sec, msg.time_ns
                );
            }
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = qtm_remaining;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_cpu(book, true, remaining_msg);  // is_ask_side = true
            }
        } else if (msg.side == Message::BID) {
            // Buy limit: match against asks, then add remainder
            int32_t qtm_remaining = msg.quantity;
            int32_t limit_price = msg.price;
            
            // Keep matching while we have quantity and valid asks
            while (qtm_remaining > 0) {
                int top_ask_idx = get_top_ask_order_idx_cpu(book);
                if (top_ask_idx == EMPTY_INDEX) break;
                if (book.asks[top_ask_idx].price == EMPTY_PRICE) break;
                if (book.asks[top_ask_idx].price > limit_price) break;
                
                match_single_order_cpu(
                    book, true, top_ask_idx, qtm_remaining,
                    msg.order_id, msg.time_sec, msg.time_ns
                );
            }
            
            // Calculate remaining quantity (what wasn't matched)
            int32_t remaining = qtm_remaining;
            
            // Only add if there's remaining quantity
            if (remaining > 0) {
                Message remaining_msg = msg;
                remaining_msg.quantity = remaining;
                add_order_cpu(book, false, remaining_msg);  // is_ask_side = false
            }
        }
    }
    else if (msg.type == Message::MARKET) {
        // Market order - aggressive matching only (no remainder added)
        Message match_msg = msg;
        if (msg.side == Message::BID) {
            // Buy market: match against asks at any price
            match_msg.price = MAX_INT;  // Will match any ask price
            match_against_asks_cpu(book, match_msg);
        } else if (msg.side == Message::ASK) {
            // Sell market: match against bids at any price
            match_msg.price = 0;  // Will match any bid price
            match_against_bids_cpu(book, match_msg);
        }
    }
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

void process_messages_sequential_cpu(
    OrderbookCPU& book,
    const Message* messages,
    int num_messages
) {
    for (int i = 0; i < num_messages; i++) {
        const Message& msg = messages[i];
        
        // Skip empty/invalid messages
        if (msg.quantity <= 0 || msg.type == 0) continue;
        
        // Process this message
        process_message_cpu(book, msg);
    }
}

void process_messages_batch_cpu(
    OrderbookBatchCPU& batch,
    const Message* messages,
    int num_messages_per_book
) {
    // Process each orderbook sequentially
    for (int book_idx = 0; book_idx < batch.num_books; book_idx++) {
        const Message* book_messages = messages + (book_idx * num_messages_per_book);
        process_messages_sequential_cpu(batch.books[book_idx], book_messages, num_messages_per_book);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void copy_orderbook_cpu(const OrderbookCPU& src, OrderbookCPU& dst) {
    if (src.n_orders_per_side != dst.n_orders_per_side || 
        src.n_trades != dst.n_trades) {
        std::cerr << "Error: Orderbook sizes don't match for copy" << std::endl;
        return;
    }
    
    std::memcpy(dst.asks, src.asks, src.n_orders_per_side * sizeof(Order));
    std::memcpy(dst.bids, src.bids, src.n_orders_per_side * sizeof(Order));
    std::memcpy(dst.trades, src.trades, src.n_trades * sizeof(Trade));
}

bool compare_orderbooks_cpu(const OrderbookCPU& book1, const OrderbookCPU& book2) {
    if (book1.n_orders_per_side != book2.n_orders_per_side ||
        book1.n_trades != book2.n_trades) {
        return false;
    }
    
    // Compare asks
    for (int i = 0; i < book1.n_orders_per_side; i++) {
        if (std::memcmp(&book1.asks[i], &book2.asks[i], sizeof(Order)) != 0) {
            return false;
        }
    }
    
    // Compare bids
    for (int i = 0; i < book1.n_orders_per_side; i++) {
        if (std::memcmp(&book1.bids[i], &book2.bids[i], sizeof(Order)) != 0) {
            return false;
        }
    }
    
    // Compare trades
    for (int i = 0; i < book1.n_trades; i++) {
        if (std::memcmp(&book1.trades[i], &book2.trades[i], sizeof(Trade)) != 0) {
            return false;
        }
    }
    
    return true;
}

void print_orderbook_cpu(const OrderbookCPU& book, int max_orders) {
    std::cout << "\n=== Orderbook State ===" << std::endl;
    
    std::cout << "\nAsks (top " << max_orders << "):" << std::endl;
    int ask_count = 0;
    for (int i = 0; i < book.n_orders_per_side && ask_count < max_orders; i++) {
        if (book.asks[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.asks[i].price
                     << ", Qty: " << book.asks[i].quantity
                     << ", ID: " << book.asks[i].order_id << std::endl;
            ask_count++;
        }
    }
    
    std::cout << "\nBids (top " << max_orders << "):" << std::endl;
    int bid_count = 0;
    for (int i = 0; i < book.n_orders_per_side && bid_count < max_orders; i++) {
        if (book.bids[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.bids[i].price
                     << ", Qty: " << book.bids[i].quantity
                     << ", ID: " << book.bids[i].order_id << std::endl;
            bid_count++;
        }
    }
    
    std::cout << "\nTrades (top " << max_orders << "):" << std::endl;
    int trade_count = 0;
    for (int i = 0; i < book.n_trades && trade_count < max_orders; i++) {
        if (book.trades[i].price != EMPTY_PRICE) {
            std::cout << "  Price: " << book.trades[i].price
                     << ", Qty: " << book.trades[i].quantity
                     << ", Passive ID: " << book.trades[i].passive_order_id
                     << ", Aggressive ID: " << book.trades[i].aggressive_order_id << std::endl;
            trade_count++;
        }
    }
    
    std::cout << "======================" << std::endl;
}

} // namespace cuda_orderbook

