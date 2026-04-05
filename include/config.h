#pragma once
#include <cstdint>

struct ModelConfig {
    int32_t vocab_size        = 151936;
    int32_t hidden_size       = 1024;
    int32_t num_layers        = 28;
    int32_t num_heads         = 16;     // Q heads
    int32_t num_kv_heads      = 8;      // KV heads (GQA)
    int32_t head_dim          = 128;
    int32_t intermediate_size = 3072;
    int32_t max_seq_len       = 32768;   // runtime cap (not model max)
    float   rms_norm_eps      = 1e-6f;
    float   rope_theta        = 1000000.0f;
    bool    tie_embeddings    = true;

    // special token ids
    int32_t bos_id            = 151643;
    int32_t eos_id            = 151645;
    int32_t im_start_id       = 151644;
    int32_t im_end_id         = 151645;

    // derived (computed after loading config)
    int32_t kv_dim() const { return num_kv_heads * head_dim; }
    int32_t q_dim()  const { return num_heads * head_dim; }
    int32_t gqa_ratio() const { return num_heads / num_kv_heads; }
};
