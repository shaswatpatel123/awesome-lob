#ifndef CUDA_ORDERBOOK_TYPES_H
#define CUDA_ORDERBOOK_TYPES_H

#include <cstdint>

namespace cuda_orderbook {

// Constants
constexpr int32_t INITID = -9000;      // Special ID for L2 snapshot orders
constexpr int32_t MAX_INT = 2147483647; // Sentinel value for empty slots
constexpr int32_t EMPTY_PRICE = -1;     // Empty order indicator

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

// Single orderbook state
// Device pointers to asks, bids, and trades arrays for one orderbook
struct OrderbookState {
    Order* asks;       // Device pointer to ask orders
    Order* bids;       // Device pointer to bid orders
    Trade* trades;     // Device pointer to trade records
    int32_t n_orders;  // Maximum number of orders per side
    int32_t n_trades;  // Maximum number of trades to record
    
    OrderbookState() 
        : asks(nullptr), bids(nullptr), trades(nullptr),
          n_orders(0), n_trades(0) {}
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
    
    // Host pointers for data transfer
    Order* h_asks;
    Order* h_bids;
    Trade* h_trades;
    
    OrderbookBatch() 
        : d_asks(nullptr), d_bids(nullptr), d_trades(nullptr),
          h_asks(nullptr), h_bids(nullptr), h_trades(nullptr),
          num_books(0), n_orders_per_book(0), n_trades_per_book(0) {}
    
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

// ============================================================================
// Hash-based LOB structures
// ============================================================================

// Hash implementation selection
enum HashImplementation {
    HASH_CUCOLLECTIONS = 0,  // Default: use NVIDIA cuCollections (fastest)
    HASH_SIMPLE_CUDA = 1     // Fallback: simple CUDA hash table
};

// Simple CUDA hash table (open addressing with linear probing)
struct SimpleHashTable {
    int32_t* keys;       // order_id (-1 = empty slot)
    int32_t* values;     // array index into orders array
    int32_t capacity;    // Hash table size (power of 2)
    int32_t size;        // Current number of entries
    int32_t mask;        // capacity - 1, for fast modulo
    
    SimpleHashTable() 
        : keys(nullptr), values(nullptr), 
          capacity(0), size(0), mask(0) {}
};

// Hash-accelerated orderbook state (single orderbook)
struct HashOrderbookState {
    // Order arrays (sorted on-demand)
    Order* asks;
    Order* bids;
    Trade* trades;
    
    // Hash indices for O(1) lookup by order_id
    // Type is void* to support both cuCollections and SimpleHashTable
    void* ask_hash_map;    // Points to cuco::static_map or SimpleHashTable
    void* bid_hash_map;
    
    // Sort state tracking
    bool asks_sorted;
    bool bids_sorted;
    
    // Configuration
    HashImplementation hash_impl;
    int32_t n_orders;      // Max orders per side
    int32_t n_trades;      // Max trades to record
    
    HashOrderbookState()
        : asks(nullptr), bids(nullptr), trades(nullptr),
          ask_hash_map(nullptr), bid_hash_map(nullptr),
          asks_sorted(false), bids_sorted(false),
          hash_impl(HASH_CUCOLLECTIONS),
          n_orders(0), n_trades(0) {}
};

// Batch of hash-accelerated orderbooks
struct HashOrderbookBatch {
    HashOrderbookState* states;  // Array of orderbook states
    int32_t num_books;           // Number of orderbooks in batch
    
    // Flattened device arrays (for compatibility)
    Order* d_asks;
    Order* d_bids;
    Trade* d_trades;
    int32_t n_orders_per_book;
    int32_t n_trades_per_book;
    
    // Configuration
    HashImplementation hash_impl;
    
    HashOrderbookBatch()
        : states(nullptr), num_books(0),
          d_asks(nullptr), d_bids(nullptr), d_trades(nullptr),
          n_orders_per_book(0), n_trades_per_book(0),
          hash_impl(HASH_CUCOLLECTIONS) {}
    
    // Get specific orderbook state
    __host__ __device__ inline HashOrderbookState* get_state(int book_idx) const {
        return &states[book_idx];
    }
    
    // Get device pointers (for backward compatibility)
    __host__ __device__ inline Order* get_asks(int book_idx) const {
        return d_asks + (book_idx * n_orders_per_book);
    }
    
    __host__ __device__ inline Order* get_bids(int book_idx) const {
        return d_bids + (book_idx * n_orders_per_book);
    }
    
    __host__ __device__ inline Trade* get_trades(int book_idx) const {
        return d_trades + (book_idx * n_trades_per_book);
    }
};

} // namespace cuda_orderbook

#endif // CUDA_ORDERBOOK_TYPES_H

