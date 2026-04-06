#pragma once
#include "config.h"
#include <cuda_bf16.h>

// Pre-allocated GPU memory pool for intermediate activations.
// Uses ping-pong strategy: layer N writes to buf_a, layer N+1 reads from buf_a
// and writes to buf_b, etc. No dynamic allocation during inference.
class ActivationPool {
public:
    ActivationPool() = default;
    ~ActivationPool();

    // Allocate based on config (call once at init)
    // Buffer size = max_seq_len * max(hidden_size, intermediate_size, q_dim)
    void init(const ModelConfig& config);

    // Get current read/write buffers (alternate each layer)
    __nv_bfloat16* input_buf()  { return (flip_ ? buf_b_ : buf_a_); }
    __nv_bfloat16* output_buf() { return (flip_ ? buf_a_ : buf_b_); }
    void  flip()       { flip_ = !flip_; }
    void  reset()      { flip_ = false; }

    // Scratch space for attention scores, logits, etc.
    __nv_bfloat16*  scratch()   { return scratch_; }
    float* logits()    { return logits_; }   // [vocab_size] in fp32

private:
    __nv_bfloat16*  buf_a_    = nullptr;
    __nv_bfloat16*  buf_b_    = nullptr;
    __nv_bfloat16*  scratch_  = nullptr;
    void* pool_base_ = nullptr; 
    float* logits_   = nullptr;
    bool   flip_     = false;
};
