#pragma once
#include "config.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

// Sink + Sliding-Window KV Cache for single-batch autoregressive decoding.
//
// Based on the StreamingLLM observation that the first few tokens ("attention
// sinks") receive disproportionately high attention and must never be evicted,
// while middle tokens can be dropped without significant quality loss.
//
// Pool layout per layer (K and V separately):
//   [n_sink + window_size, num_kv_heads, head_dim]   (bf16, row-major)
//
// Physical slot assignment for logical position pos:
//   pos < n_sink   →  slot = pos                          (sink region)
//   pos >= n_sink  →  slot = n_sink + (pos-n_sink) % window_size  (ring buffer)
//
// window_size is NOT fixed in config — it is computed at runtime from available
// GPU memory after weights and activations are already allocated:
//   budget = free_gpu_mem - SYSTEM_RESERVE_BYTES
//   pool_tokens = budget / (2 * num_layers * kv_dim * sizeof(bf16))
//   window_size = pool_tokens - n_sink

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    // Allocate GPU pool for all layers. Call once at startup.
    void init(const ModelConfig& config);

    // Scatter seq_len tokens (starting at logical start_pos) into the pool.
    // Sink tokens go to fixed slots; window tokens round-robin in the ring.
    // Precondition: seq_len <= n_sink_ + window_size_ (enforced at runtime).
    void append_kv(int layer,
                   const __nv_bfloat16* k_src,
                   const __nv_bfloat16* v_src,
                   int seq_len, int start_pos);

    // Per-layer pool base pointers passed to the attention kernel.
    __nv_bfloat16* k_pool(int layer) { return k_pools_[layer]; }
    __nv_bfloat16* v_pool(int layer) { return v_pools_[layer]; }

    int n_sink()      const { return n_sink_; }
    int window_size() const { return window_size_; }
    int max_tokens()  const { return n_sink_ + window_size_; }  // max prefill length

    // Reset for a new conversation; start_pos is managed by the caller.
    void reset() {}

    // No-op: kept so Engine can call cache_.advance(n) without changes.
    void advance(int) {}

private:
    std::vector<__nv_bfloat16*> k_pools_;   // [num_layers] device ptrs
    std::vector<__nv_bfloat16*> v_pools_;   // [num_layers] device ptrs

    int kv_dim_      = 0;
    int num_layers_  = 0;
    int n_sink_      = 0;   // set from config.kv_n_sink in init()
    int window_size_ = 0;   // computed from available GPU memory in init()
};
