#pragma once
#include "config.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

// Paged KV Cache for autoregressive decoding.
//
// GPU memory is divided into fixed-size pages (BLOCK_SIZE tokens each).
// A block table maps logical block indices to physical page indices, so the
// sequence's KV vectors can occupy non-contiguous pages — no ring-buffer
// wrap corruption, and pages are allocated only when actually needed.
//
// Pool layout per layer (K and V separately):
//   [total_pages * BLOCK_SIZE, num_kv_heads, head_dim]   (bf16)
//
// Block table (device): block_table[logical_block] = physical_page_index

class KVCache {
public:
    static constexpr int BLOCK_SIZE = 16;  // tokens per page

    KVCache() = default;
    ~KVCache();

    // Allocate GPU page pool for all layers. Call once at startup.
    void init(const ModelConfig& config);

    // Allocate physical pages covering [start_pos, start_pos+seq_len) and
    // upload the updated block table entries to device.
    // Call once per forward pass, before the layer loop.
    void prepare(int seq_len, int start_pos);

    // CUDA kernel: scatter seq_len K/V tokens from k_src/v_src into the
    // paged pool according to the block table. Call once per layer.
    void scatter_kv(int layer,
                    const __nv_bfloat16* k_src,
                    const __nv_bfloat16* v_src,
                    int seq_len, int start_pos);

    // Per-layer pool base pointers (passed to the attention kernel)
    __nv_bfloat16* k_pool(int layer) { return k_pools_[layer]; }
    __nv_bfloat16* v_pool(int layer) { return v_pools_[layer]; }

    // Device block table (passed to the attention kernel)
    const int32_t* d_block_table() const { return d_block_table_; }

    int block_size() const { return BLOCK_SIZE; }

    // Reset for a new conversation: return all pages to the free list.
    void reset();

    // No-op: kept so Engine can call cache_.advance(n) without changes.
    void advance(int) {}

private:
    int32_t alloc_page();   // pops one index from free_pages_

    std::vector<__nv_bfloat16*> k_pools_;       // [num_layers] device ptrs
    std::vector<__nv_bfloat16*> v_pools_;       // [num_layers] device ptrs

    std::vector<int32_t> h_block_table_;        // host mirror [total_pages]
    int32_t*             d_block_table_ = nullptr;

    std::vector<int32_t> free_pages_;           // stack of free physical page ids

    int total_pages_ = 0;
    int kv_dim_      = 0;
    int num_layers_  = 0;
};
