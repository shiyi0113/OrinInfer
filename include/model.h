#pragma once
#include "config.h"
#include "weight_store.h"
#include "activation_pool.h"
#include "kv_cache.h"

class Model {
public:
    Model() = default;

    // Initialize with pre-loaded weights and pre-allocated buffers
    void init(const ModelConfig& config,
              const WeightStore& weights,
              ActivationPool& pool,
              KVCache& cache);

    // Single forward pass.
    // token_ids: [seq_len] input token ids (device pointer)
    // seq_len:   number of tokens (N for prefill, 1 for decode)
    // start_pos: position in sequence (for RoPE and KV cache indexing)
    //
    // Returns: device pointer to logits [vocab_size] in fp32
    //          (pointer into ActivationPool, valid until next forward call)
    float* forward(const int32_t* d_token_ids, int seq_len, int start_pos);

private:
    // Run one transformer layer
    void transformer_layer(int layer_idx, half* x, int seq_len, int start_pos);

    // Final: rmsnorm → lm_head projection
    float* lm_head_proj(half* x, int seq_len);

    ModelConfig        config_;
    const WeightStore* weights_ = nullptr;
    ActivationPool*    pool_    = nullptr;
    KVCache*           cache_   = nullptr;

    // Temporary buffers (pointers into ActivationPool scratch)
    half* q_buf_    = nullptr;
    half* k_buf_    = nullptr;
    half* v_buf_    = nullptr;
    half* attn_out_ = nullptr;
    half* gate_buf_ = nullptr;
    half* up_buf_   = nullptr;
    half* residual_ = nullptr;
};
