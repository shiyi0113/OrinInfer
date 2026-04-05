#pragma once
#include "config.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

// Ring-buffer KV cache for autoregressive decoding.
// When pos reaches max_seq_len, oldest entries are overwritten.
// Layout per layer: K=[max_seq_len, kv_dim], V=[max_seq_len, kv_dim]
class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    // Allocate GPU memory for all layers
    void init(const ModelConfig& config);

    // Get K/V buffer pointers for a specific layer
    __nv_bfloat16* k(int layer) { return k_bufs_[layer]; }
    __nv_bfloat16* v(int layer) { return v_bufs_[layer]; }

    // Current sequence position (where to write next)
    int pos() const { return pos_; }

    // Number of valid entries (min(pos_, max_seq_len_))
    int len() const { return (pos_ < max_seq_len_) ? pos_ : max_seq_len_; }

    // Advance position by n tokens (after writing)
    void advance(int n) { pos_ += n; }

    // Write index for ring buffer: pos % max_seq_len
    int write_idx() const { return pos_ % max_seq_len_; }

    // Reset for new conversation
    void reset() { pos_ = 0; }

    int max_seq_len() const { return max_seq_len_; }
    int kv_dim()     const { return kv_dim_; }

private:
    std::vector<__nv_bfloat16*> k_bufs_;   // [num_layers]
    std::vector<__nv_bfloat16*> v_bufs_;   // [num_layers]
    int max_seq_len_ = 0;
    int kv_dim_      = 0;
    int pos_         = 0;
};
