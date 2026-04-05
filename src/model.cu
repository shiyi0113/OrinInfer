#include "model.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <iostream>

void Model::init(const ModelConfig& config,
                 const WeightStore& weights,
                 ActivationPool& pool,
                 KVCache& cache) {
    config_  = config;
    weights_ = &weights;
    pool_    = &pool;
    cache_   = &cache;
    std::cout << "[Model] Initialized: " << config.num_layers << " layers, "
              << config.hidden_size << " hidden, GQA "
              << config.num_heads << "/" << config.num_kv_heads << "\n";
}

float* Model::forward(const int32_t* d_token_ids, int seq_len, int start_pos) {
    // ── 1. Embedding lookup ──────────────────────────────
    __nv_bfloat16* x = pool_->input_buf();
    kernels::embedding(x, weights_->embedding(), d_token_ids,
                       seq_len, config_.hidden_size);

    // ── 2. Transformer layers ────────────────────────────
    for (int i = 0; i < config_.num_layers; i++) {
        transformer_layer(i, x, seq_len, start_pos);
        // x pointer stays valid — ping-pong handled inside transformer_layer
    }

    // ── 3. Final norm + LM head projection ───────────────
    return lm_head_proj(x, seq_len);
}

void Model::transformer_layer(int layer_idx, __nv_bfloat16* x, int seq_len, int start_pos) {
    auto& L = weights_->layer(layer_idx);
    int H   = config_.hidden_size;
    int QD  = config_.q_dim();
    int KVD = config_.kv_dim();
    int IM  = config_.intermediate_size;
    int HD  = config_.head_dim;

    __nv_bfloat16* scratch = pool_->scratch();

    // ── Attention block ──────────────────────────────────

    // 1. Input LayerNorm
    __nv_bfloat16* normed = pool_->output_buf();
    kernels::rmsnorm(normed, x, L.input_layernorm, seq_len, H, config_.rms_norm_eps);

    // 2. Q/K/V projections (matmul auto-dispatches GEMM vs GEMV)
    q_buf_  = scratch;                                    // [seq_len, q_dim]
    k_buf_  = scratch + (size_t)seq_len * QD;             // [seq_len, kv_dim]
    v_buf_  = scratch + (size_t)seq_len * (QD + KVD);     // [seq_len, kv_dim]

    kernels::matmul(q_buf_, normed, L.q_proj, seq_len, QD, H);
    kernels::matmul(k_buf_, normed, L.k_proj, seq_len, KVD, H);
    kernels::matmul(v_buf_, normed, L.v_proj, seq_len, KVD, H);

    // 3. QK-Norm (Qwen3 specific: RMSNorm on each head before RoPE)
    // Apply per-head: q_buf[seq_len, num_heads, head_dim] normed by q_norm[head_dim]
    for (int h = 0; h < config_.num_heads; h++) {
        __nv_bfloat16* q_head = q_buf_ + h * HD;
        kernels::rmsnorm(q_head, q_head, L.q_norm, seq_len, HD, config_.rms_norm_eps);
        // TODO: stride handling — heads are interleaved, need proper offset
    }
    for (int h = 0; h < config_.num_kv_heads; h++) {
        __nv_bfloat16* k_head = k_buf_ + h * HD;
        kernels::rmsnorm(k_head, k_head, L.k_norm, seq_len, HD, config_.rms_norm_eps);
    }

    // 4. RoPE
    kernels::rope(q_buf_, k_buf_, seq_len,
                  config_.num_heads, config_.num_kv_heads,
                  HD, start_pos, config_.rope_theta);

    // 5. Write K/V to cache (ring buffer)
    // TODO: Copy k_buf_ and v_buf_ into cache_->k(layer_idx) and cache_->v(layer_idx)
    //       at position write_idx = (start_pos) % max_seq_len for each of seq_len tokens

    // 6. Attention: Q against cached K/V
    attn_out_ = normed;  // reuse normed buffer
    int cache_len = cache_->len() + seq_len;  // include newly written KV
    kernels::attention(attn_out_, q_buf_,
                       cache_->k(layer_idx), cache_->v(layer_idx),
                       seq_len, cache_len,
                       config_.num_heads, config_.num_kv_heads,
                       HD, 1.0f / sqrtf((float)HD));

    // 7. Output projection
    __nv_bfloat16* attn_proj = pool_->output_buf();
    kernels::matmul(attn_proj, attn_out_, L.o_proj, seq_len, H, QD);

    // 8. Residual add: x = x + attn_proj
    // TODO: CUDA kernel for element-wise add, or fuse into rmsnorm
    // For now placeholder — in production, fuse residual add with next rmsnorm

    // ── FFN block ────────────────────────────────────────

    // 9. Post-attention LayerNorm
    kernels::rmsnorm(normed, x, L.post_attn_norm, seq_len, H, config_.rms_norm_eps);

    // 10. Gate + Up projections
    gate_buf_ = scratch;
    up_buf_   = scratch + (size_t)seq_len * IM;
    kernels::matmul(gate_buf_, normed, L.gate_proj, seq_len, IM, H);
    kernels::matmul(up_buf_,   normed, L.up_proj,   seq_len, IM, H);

    // 11. Fused SwiGLU: SiLU(gate) * up
    kernels::fused_silu_mul(gate_buf_, gate_buf_, up_buf_, seq_len, IM);

    // 12. Down projection
    __nv_bfloat16* ffn_out = pool_->output_buf();
    kernels::matmul(ffn_out, gate_buf_, L.down_proj, seq_len, H, IM);

    // 13. Residual add: x = x + ffn_out
    // TODO: element-wise add kernel
}

float* Model::lm_head_proj(__nv_bfloat16* x, int seq_len) {
    // Final RMSNorm
    __nv_bfloat16* normed = pool_->output_buf();
    kernels::rmsnorm(normed, x, weights_->final_norm(),
                     seq_len, config_.hidden_size, config_.rms_norm_eps);

    // LM head: project hidden → vocab logits
    // Only need last token's logits for generation
    // TODO: For decode (seq_len=1), just matmul the single vector
    //       For prefill, only matmul the last row
    __nv_bfloat16* last_hidden = normed + (size_t)(seq_len - 1) * config_.hidden_size;
    float* logits = pool_->logits();

    // matmul: [1, hidden] x [vocab, hidden]^T → [1, vocab]
    // Result needs to be fp32 for numerical stability in sampler
    // TODO: Implement fp16→fp32 matmul variant, or matmul then cast
    kernels::matmul((__nv_bfloat16*)logits, last_hidden, weights_->lm_head(),
                    1, config_.vocab_size, config_.hidden_size);
    // TODO: Cast result to fp32

    return logits;
}
