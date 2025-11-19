#ifndef CUDA_ORDERBOOK_TYPES_H
#define CUDA_ORDERBOOK_TYPES_H

#include <cstdint>

// Make CUDA keywords work in both CUDA and regular C++ compilation
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace cuda_orderbook {

// Constants
constexpr int32_t INITID = -9000;      // Special ID for L2 snapshot orders
constexpr int32_t MAX_INT = 2147483647; // Sentinel value for empty slots
constexpr int32_t EMPTY_PRICE = -1;     // Empty order indicator
constexpr int32_t EMPTY_INDEX = -1;     // Empty index indicator for linked lists
constexpr int32_t MAX_PRICE = 1000000;  // Maximum price in cents ($10,000)
constexpr int32_t MIN_PRICE = 1;        // Minimum price in cents ($0.01)
constexpr int32_t PRICE_MAP_SIZE = 65536; // Hash table size for price map (2^16)
constexpr int32_t ORDER_ID_MAP_SIZE = 65536; // Hash table size for order-ID map

// Order format: [price, quantity, order_id, trader_id, time_sec, time_ns]
// Maps to JAX array structure from JaxOrderBookArrays.py
struct Order {
    int32_t price;      // Price level (-1 indicates empty slot)
    int32_t quantity;   // Order size
    int32_t order_id;   // Unique order identifier
    int32_t trader_id;  // Trader/agent identifier
    int32_t time_sec;   // Timestamp seconds
    int32_t time_ns;    // Timestamp nanoseconds
    
    // Default constructor for empty order
    __host__ __device__ Order() 
        : price(EMPTY_PRICE), quantity(0), order_id(0), 
          trader_id(0), time_sec(0), time_ns(0) {}
    
    // Check if order slot is empty
    __host__ __device__ inline bool is_empty() const {
        return price == EMPTY_PRICE;
    }
    
    // Check if order is valid (has positive quantity)
    __host__ __device__ inline bool is_valid() const {
        return quantity > 0 && price != EMPTY_PRICE;
    }
};

// Message format: [type, side, quantity, price, trader_id, order_id, time_sec, time_ns]
// Maps to JAX message array from process_order functions
struct Message {
    int32_t type;      // Order type: 1=limit, 2=cancel, 3=delete, 4=market
    int32_t side;      // Order side: -1=ask (sell), 1=bid (buy)
    int32_t quantity;  // Order quantity
    int32_t price;     // Limit price (ignored for market orders)
    int32_t trader_id; // Trader identifier
    int32_t order_id;  // Order identifier
    int32_t time_sec;  // Timestamp seconds
    int32_t time_ns;   // Timestamp nanoseconds
    
    // Order type enum
    enum Type {
        LIMIT = 1,
        CANCEL = 2,
        DELETE = 3,  // Treated same as CANCEL
        MARKET = 4
    };
    
    // Order side enum
    enum Side {
        ASK = -1,  // Sell
        BID = 1    // Buy
    };
    
    __host__ __device__ Message() 
        : type(0), side(0), quantity(0), price(0),
          trader_id(0), order_id(0), time_sec(0), time_ns(0) {}
};

// Trade record: [price, quantity, passive_order_id, aggressive_order_id, time_sec, time_ns]
// Stores executed trade information
struct Trade {
    int32_t price;             // Execution price
    int32_t quantity;          // Executed quantity
    int32_t passive_order_id;  // Resting order ID
    int32_t aggressive_order_id; // Incoming order ID
    int32_t time_sec;          // Execution timestamp seconds
    int32_t time_ns;           // Execution timestamp nanoseconds
    
    __host__ __device__ Trade()
        : price(EMPTY_PRICE), quantity(0), passive_order_id(0),
          aggressive_order_id(0), time_sec(0), time_ns(0) {}
    
    __host__ __device__ inline bool is_empty() const {
        return price == EMPTY_PRICE;
    }
};

// ============================================================================
// PRICE-AWARE DATA STRUCTURES
// ============================================================================

/**
 * Order metadata for price-aware orderbook
 * Stores linked list pointers and price bucket association
 */
struct OrderMetadata {
    int32_t next_idx;        // Index of next order in same price bucket (or EMPTY_INDEX)
    int32_t prev_idx;        // Index of previous order in same price bucket (or EMPTY_INDEX)
    int32_t price_bucket_idx; // Index into price bucket array for this order's price
    bool is_valid;           // Whether this metadata entry is in use
    
    __host__ __device__ OrderMetadata()
        : next_idx(EMPTY_INDEX), prev_idx(EMPTY_INDEX), 
          price_bucket_idx(EMPTY_INDEX), is_valid(false) {}
    
    __host__ __device__ inline bool is_empty() const {
        return !is_valid || price_bucket_idx == EMPTY_INDEX;
    }
};

/**
 * Price bucket - represents a single price level
 * Contains linked list of orders at this price with FIFO ordering (time priority)
 */
struct PriceBucket {
    int32_t head_idx;        // Index of first order at this price (or EMPTY_INDEX)
    int32_t tail_idx;        // Index of last order at this price (or EMPTY_INDEX)
    int32_t total_quantity;  // Sum of all quantities at this price
    int32_t price;           // Price level (for validation)
    bool is_active;          // Whether this bucket has any orders
    
    __host__ __device__ PriceBucket()
        : head_idx(EMPTY_INDEX), tail_idx(EMPTY_INDEX), 
          total_quantity(0), price(EMPTY_PRICE), is_active(false) {}
    
    __host__ __device__ inline bool is_empty() const {
        return !is_active || head_idx == EMPTY_INDEX;
    }
};

/**
 * Price map entry - maps price to bucket index
 * Uses linear probing for hash collisions
 */
struct PriceMapEntry {
    int32_t price;           // Price key (EMPTY_PRICE if empty)
    int32_t bucket_idx;      // Index into price bucket array (or EMPTY_INDEX)
    bool is_active;          // Whether this entry is in use
    bool was_deleted;        // Whether this slot previously held a value
    
    __host__ __device__ PriceMapEntry()
        : price(EMPTY_PRICE), bucket_idx(EMPTY_INDEX), 
          is_active(false), was_deleted(false) {}
    
    __host__ __device__ inline bool is_empty() const {
        return !is_active && !was_deleted;
    }
    
    __host__ __device__ inline bool is_tombstone() const {
        return !is_active && was_deleted;
    }
};

/**
 * Order-ID map entry - maps order_id to order slot index
 * Uses linear probing for hash collisions
 */
struct OrderIDMapEntry {
    int32_t order_id;        // Order ID key (0 if empty)
    int32_t order_idx;       // Index into order array (or EMPTY_INDEX)
    bool is_active;          // Whether this entry is in use
    bool was_deleted;        // Whether this slot previously held a value
    
    __host__ __device__ OrderIDMapEntry()
        : order_id(0), order_idx(EMPTY_INDEX), 
          is_active(false), was_deleted(false) {}
    
    __host__ __device__ inline bool is_empty() const {
        return !is_active && !was_deleted;
    }
    
    __host__ __device__ inline bool is_tombstone() const {
        return !is_active && was_deleted;
    }
};

/**
 * Best price tracker - tracks best bid and ask prices
 * Allows O(1) access to best prices
 */
struct BestPriceTracker {
    int32_t best_ask_price;  // Best (lowest) ask price (MAX_INT if none)
    int32_t best_bid_price;  // Best (highest) bid price (EMPTY_PRICE if none)
    int32_t best_ask_bucket_idx; // Bucket index for best ask (or EMPTY_INDEX)
    int32_t best_bid_bucket_idx; // Bucket index for best bid (or EMPTY_INDEX)
    
    __host__ __device__ BestPriceTracker()
        : best_ask_price(MAX_INT), best_bid_price(EMPTY_PRICE),
          best_ask_bucket_idx(EMPTY_INDEX), best_bid_bucket_idx(EMPTY_INDEX) {}
    
    __host__ __device__ inline bool has_best_ask() const {
        return best_ask_price != MAX_INT && best_ask_bucket_idx != EMPTY_INDEX;
    }
    
    __host__ __device__ inline bool has_best_bid() const {
        return best_bid_price != EMPTY_PRICE && best_bid_bucket_idx != EMPTY_INDEX;
    }
};

// Single orderbook state
// Device pointers to asks, bids, and trades arrays for one orderbook
struct OrderbookState {
    Order* asks;       // Device pointer to ask orders
    Order* bids;       // Device pointer to bid orders
    Trade* trades;     // Device pointer to trade records
    int32_t n_orders;  // Maximum number of orders per side
    int32_t n_trades;  // Maximum number of trades to record
    
    // Price-aware structures for asks
    OrderMetadata* ask_metadata;      // Metadata for ask orders
    PriceBucket* ask_buckets;         // Price buckets for asks
    PriceMapEntry* ask_price_map;     // Hash map: price -> bucket index for asks
    OrderIDMapEntry* ask_order_id_map; // Hash map: order_id -> order index for asks
    BestPriceTracker* ask_tracker;    // Best ask price tracker
    
    // Price-aware structures for bids
    OrderMetadata* bid_metadata;      // Metadata for bid orders
    PriceBucket* bid_buckets;         // Price buckets for bids
    PriceMapEntry* bid_price_map;     // Hash map: price -> bucket index for bids
    OrderIDMapEntry* bid_order_id_map; // Hash map: order_id -> order index for bids
    BestPriceTracker* bid_tracker;    // Best bid price tracker
    
    // Capacity constants
    int32_t n_price_buckets;          // Maximum number of active price levels
    int32_t price_map_size;           // Size of price hash map
    int32_t order_id_map_size;        // Size of order-ID hash map
    
    __host__ __device__ OrderbookState() 
        : asks(nullptr), bids(nullptr), trades(nullptr),
          n_orders(0), n_trades(0),
          ask_metadata(nullptr), ask_buckets(nullptr), 
          ask_price_map(nullptr), ask_order_id_map(nullptr), ask_tracker(nullptr),
          bid_metadata(nullptr), bid_buckets(nullptr),
          bid_price_map(nullptr), bid_order_id_map(nullptr), bid_tracker(nullptr),
          n_price_buckets(0), price_map_size(PRICE_MAP_SIZE), 
          order_id_map_size(ORDER_ID_MAP_SIZE) {}
};

// Batch of orderbooks for parallel processing
// Flattened arrays: [book0_orders, book1_orders, ..., bookN_orders]
// This structure enables parallel processing of multiple independent orderbooks
struct OrderbookBatch {
    Order* d_asks;     // Device pointer: all orderbooks' asks (flattened)
    Order* d_bids;     // Device pointer: all orderbooks' bids (flattened)
    Trade* d_trades;   // Device pointer: all orderbooks' trades (flattened)
    int32_t num_books;          // Number of orderbooks in batch
    int32_t n_orders_per_book;  // Orders per side per book
    int32_t n_trades_per_book;  // Trades per book
    
    // Price-aware structures for asks (flattened across all books)
    OrderMetadata* d_ask_metadata;      // Metadata for ask orders
    PriceBucket* d_ask_buckets;         // Price buckets for asks
    PriceMapEntry* d_ask_price_map;     // Hash map: price -> bucket index for asks
    OrderIDMapEntry* d_ask_order_id_map; // Hash map: order_id -> order index for asks
    BestPriceTracker* d_ask_trackers;   // Best ask price trackers (one per book)
    
    // Price-aware structures for bids (flattened across all books)
    OrderMetadata* d_bid_metadata;      // Metadata for bid orders
    PriceBucket* d_bid_buckets;         // Price buckets for bids
    PriceMapEntry* d_bid_price_map;     // Hash map: price -> bucket index for bids
    OrderIDMapEntry* d_bid_order_id_map; // Hash map: order_id -> order index for bids
    BestPriceTracker* d_bid_trackers;   // Best bid price trackers (one per book)
    
    // Host pointers for data transfer
    Order* h_asks;
    Order* h_bids;
    Trade* h_trades;
    
    // Host pointers for price-aware structures
    OrderMetadata* h_ask_metadata;
    PriceBucket* h_ask_buckets;
    PriceMapEntry* h_ask_price_map;
    OrderIDMapEntry* h_ask_order_id_map;
    BestPriceTracker* h_ask_trackers;
    
    OrderMetadata* h_bid_metadata;
    PriceBucket* h_bid_buckets;
    PriceMapEntry* h_bid_price_map;
    OrderIDMapEntry* h_bid_order_id_map;
    BestPriceTracker* h_bid_trackers;
    
    // Capacity constants
    int32_t n_price_buckets_per_book;   // Maximum active price levels per book
    int32_t price_map_size;             // Size of price hash map
    int32_t order_id_map_size;          // Size of order-ID hash map
    
    OrderbookBatch() 
        : d_asks(nullptr), d_bids(nullptr), d_trades(nullptr),
          num_books(0), n_orders_per_book(0), n_trades_per_book(0),
          d_ask_metadata(nullptr), d_ask_buckets(nullptr),
          d_ask_price_map(nullptr), d_ask_order_id_map(nullptr), d_ask_trackers(nullptr),
          d_bid_metadata(nullptr), d_bid_buckets(nullptr),
          d_bid_price_map(nullptr), d_bid_order_id_map(nullptr), d_bid_trackers(nullptr),
          h_asks(nullptr), h_bids(nullptr), h_trades(nullptr),
          h_ask_metadata(nullptr), h_ask_buckets(nullptr),
          h_ask_price_map(nullptr), h_ask_order_id_map(nullptr), h_ask_trackers(nullptr),
          h_bid_metadata(nullptr), h_bid_buckets(nullptr),
          h_bid_price_map(nullptr), h_bid_order_id_map(nullptr), h_bid_trackers(nullptr),
          n_price_buckets_per_book(1024), price_map_size(PRICE_MAP_SIZE),
          order_id_map_size(ORDER_ID_MAP_SIZE) {}
    
    // Get device pointer to specific orderbook's asks
    __host__ __device__ inline Order* get_asks(int book_idx) const {
        return d_asks + (book_idx * n_orders_per_book);
    }
    
    // Get device pointer to specific orderbook's bids
    __host__ __device__ inline Order* get_bids(int book_idx) const {
        return d_bids + (book_idx * n_orders_per_book);
    }
    
    // Get device pointer to specific orderbook's trades
    __host__ __device__ inline Trade* get_trades(int book_idx) const {
        return d_trades + (book_idx * n_trades_per_book);
    }
    
    // Get device pointer to specific orderbook's ask metadata
    __host__ __device__ inline OrderMetadata* get_ask_metadata(int book_idx) const {
        return d_ask_metadata + (book_idx * n_orders_per_book);
    }
    
    // Get device pointer to specific orderbook's bid metadata
    __host__ __device__ inline OrderMetadata* get_bid_metadata(int book_idx) const {
        return d_bid_metadata + (book_idx * n_orders_per_book);
    }
    
    // Get device pointer to specific orderbook's ask buckets
    __host__ __device__ inline PriceBucket* get_ask_buckets(int book_idx) const {
        return d_ask_buckets + (book_idx * n_price_buckets_per_book);
    }
    
    // Get device pointer to specific orderbook's bid buckets
    __host__ __device__ inline PriceBucket* get_bid_buckets(int book_idx) const {
        return d_bid_buckets + (book_idx * n_price_buckets_per_book);
    }
    
    // Get device pointer to specific orderbook's ask price map
    __host__ __device__ inline PriceMapEntry* get_ask_price_map(int book_idx) const {
        return d_ask_price_map + (book_idx * price_map_size);
    }
    
    // Get device pointer to specific orderbook's bid price map
    __host__ __device__ inline PriceMapEntry* get_bid_price_map(int book_idx) const {
        return d_bid_price_map + (book_idx * price_map_size);
    }
    
    // Get device pointer to specific orderbook's ask order-ID map
    __host__ __device__ inline OrderIDMapEntry* get_ask_order_id_map(int book_idx) const {
        return d_ask_order_id_map + (book_idx * order_id_map_size);
    }
    
    // Get device pointer to specific orderbook's bid order-ID map
    __host__ __device__ inline OrderIDMapEntry* get_bid_order_id_map(int book_idx) const {
        return d_bid_order_id_map + (book_idx * order_id_map_size);
    }
    
    // Get device pointer to specific orderbook's ask tracker
    __host__ __device__ inline BestPriceTracker* get_ask_tracker(int book_idx) const {
        return d_ask_trackers + book_idx;
    }
    
    // Get device pointer to specific orderbook's bid tracker
    __host__ __device__ inline BestPriceTracker* get_bid_tracker(int book_idx) const {
        return d_bid_trackers + book_idx;
    }
    
    // Helper to get OrderbookState for a specific book
    __host__ __device__ inline OrderbookState get_state(int book_idx) const {
        OrderbookState state;
        state.asks = get_asks(book_idx);
        state.bids = get_bids(book_idx);
        state.trades = get_trades(book_idx);
        state.n_orders = n_orders_per_book;
        state.n_trades = n_trades_per_book;
        state.ask_metadata = get_ask_metadata(book_idx);
        state.bid_metadata = get_bid_metadata(book_idx);
        state.ask_buckets = get_ask_buckets(book_idx);
        state.bid_buckets = get_bid_buckets(book_idx);
        state.ask_price_map = get_ask_price_map(book_idx);
        state.bid_price_map = get_bid_price_map(book_idx);
        state.ask_order_id_map = get_ask_order_id_map(book_idx);
        state.bid_order_id_map = get_bid_order_id_map(book_idx);
        state.ask_tracker = get_ask_tracker(book_idx);
        state.bid_tracker = get_bid_tracker(book_idx);
        state.n_price_buckets = n_price_buckets_per_book;
        state.price_map_size = price_map_size;
        state.order_id_map_size = order_id_map_size;
        return state;
    }
};

// L2 orderbook snapshot
// Format: [ask_p1, ask_q1, bid_p1, bid_q1, ..., ask_pN, ask_qN, bid_pN, bid_qN]
struct L2State {
    int32_t* data;      // Flattened array of price-quantity pairs
    int32_t n_levels;   // Number of price levels per side
    
    L2State() : data(nullptr), n_levels(0) {}
    
    // Size in int32_t elements: n_levels * 4 (ask_price, ask_qty, bid_price, bid_qty per level)
    __host__ __device__ inline int32_t size() const {
        return n_levels * 4;
    }
};

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_TYPES_H

