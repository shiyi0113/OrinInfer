#include "model.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cmath>
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
    // ── 1. Embedding lookup → buf_a (residual stream x) ─────
    __nv_bfloat16* x = pool_->input_buf();  // buf_a: persistent residual stream
    kernels::embedding(x, weights_->embedding(), d_token_ids,
                       seq_len, config_.hidden_size);

    // ── 2. Transformer layers ────────────────────────────────
    for (int i = 0; i < config_.num_layers; i++) {
        transformer_layer(i, x, seq_len, start_pos);
    }

    // ── 3. Final norm + LM head projection ───────────────────
    return lm_head_proj(x, seq_len);
}

void Model::transformer_layer(int layer_idx, __nv_bfloat16* x, int seq_len, int start_pos) {
    auto& L = weights_->layer(layer_idx);
    int H   = config_.hidden_size;
    int QD  = config_.q_dim();          // num_heads * head_dim
    int KVD = config_.kv_dim();         // num_kv_heads * head_dim
    int IM  = config_.intermediate_size;
    int HD  = config_.head_dim;

    // buf_b: temporary output (normed, attn output, ffn output)
    // scratch: QKV projections, then o_proj output, then gate/up projections
    __nv_bfloat16* buf_b   = pool_->output_buf();  // never alias with x (buf_a)
    __nv_bfloat16* scratch = pool_->scratch();

    // ── Attention block ──────────────────────────────────────

    // 1. Input LayerNorm: x → buf_b
    kernels::rmsnorm(buf_b, x, L.input_layernorm, seq_len, H, config_.rms_norm_eps);

    // 2. QKV projections: buf_b → scratch
    __nv_bfloat16* q_buf = scratch;
    __nv_bfloat16* k_buf = scratch + (size_t)seq_len * QD;
    __nv_bfloat16* v_buf = scratch + (size_t)seq_len * (QD + KVD);
    kernels::matmul(q_buf, buf_b, L.q_proj, seq_len, QD, H);
    kernels::matmul(k_buf, buf_b, L.k_proj, seq_len, KVD, H);
    kernels::matmul(v_buf, buf_b, L.v_proj, seq_len, KVD, H);

    // 3. QK-Norm (Qwen3): reshape Q as [seq*num_heads, head_dim] for batch RMSNorm
    kernels::rmsnorm(q_buf, q_buf, L.q_norm,
                     seq_len * config_.num_heads, HD, config_.rms_norm_eps);
    kernels::rmsnorm(k_buf, k_buf, L.k_norm,
                     seq_len * config_.num_kv_heads, HD, config_.rms_norm_eps);

    // 4. RoPE in-place on Q and K
    kernels::rope(q_buf, k_buf, seq_len,
                  config_.num_heads, config_.num_kv_heads,
                  HD, start_pos, config_.rope_theta);

    // 5. Write K/V to KV cache ring buffer
    {
        int write_pos        = cache_->write_idx();
        int max_sl           = cache_->max_seq_len();
        int tokens_to_wrap   = max_sl - write_pos;

        if (seq_len <= tokens_to_wrap) {
            // Fits without wrapping
            cudaMemcpy(cache_->k(layer_idx) + (size_t)write_pos * KVD, k_buf,
                       (size_t)seq_len * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(cache_->v(layer_idx) + (size_t)write_pos * KVD, v_buf,
                       (size_t)seq_len * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
        } else {
            // Ring-buffer wrap: two copies
            int first  = tokens_to_wrap;
            int second = seq_len - first;
            cudaMemcpy(cache_->k(layer_idx) + (size_t)write_pos * KVD, k_buf,
                       (size_t)first * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(cache_->k(layer_idx), k_buf + (size_t)first * KVD,
                       (size_t)second * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(cache_->v(layer_idx) + (size_t)write_pos * KVD, v_buf,
                       (size_t)first * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(cache_->v(layer_idx), v_buf + (size_t)first * KVD,
                       (size_t)second * KVD * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice);
        }
    }

    // 6. Attention: Q (scratch) × KV cache → buf_b
    //    cache_len includes the tokens just written above
    int cache_len = cache_->len() + seq_len;
    kernels::attention(buf_b, q_buf,
                       cache_->k(layer_idx), cache_->v(layer_idx),
                       seq_len, cache_len,
                       config_.num_heads, config_.num_kv_heads,
                       HD, 1.0f / sqrtf((float)HD));

    // 7. Output projection: buf_b (attn out) → scratch (q/k/v no longer needed)
    kernels::matmul(scratch, buf_b, L.o_proj, seq_len, H, QD);

    // 8. Residual add: x += scratch
    kernels::add(x, scratch, seq_len * H);

    // ── FFN block ────────────────────────────────────────────

    // 9. Post-attention LayerNorm: x → buf_b
    kernels::rmsnorm(buf_b, x, L.post_attn_norm, seq_len, H, config_.rms_norm_eps);

    // 10. Gate + Up projections: buf_b → scratch
    __nv_bfloat16* gate_buf = scratch;
    __nv_bfloat16* up_buf   = scratch + (size_t)seq_len * IM;
    kernels::matmul(gate_buf, buf_b, L.gate_proj, seq_len, IM, H);
    kernels::matmul(up_buf,   buf_b, L.up_proj,   seq_len, IM, H);

    // 11. Fused SwiGLU: gate_buf = SiLU(gate) * up (in-place)
    kernels::fused_silu_mul(gate_buf, gate_buf, up_buf, seq_len, IM);

    // 12. Down projection: scratch → buf_b
    kernels::matmul(buf_b, gate_buf, L.down_proj, seq_len, H, IM);

    // 13. Residual add: x += buf_b
    kernels::add(x, buf_b, seq_len * H);
}

float* Model::lm_head_proj(__nv_bfloat16* x, int seq_len) {
    // Final RMSNorm: x → buf_b
    __nv_bfloat16* buf_b = pool_->output_buf();
    kernels::rmsnorm(buf_b, x, weights_->final_norm(),
                     seq_len, config_.hidden_size, config_.rms_norm_eps);

    // Take only the last token's hidden state
    __nv_bfloat16* last_hidden = buf_b + (size_t)(seq_len - 1) * config_.hidden_size;

    // LM head: [1, hidden] × [vocab, hidden]^T → [1, vocab] in BF16
    // Store in scratch temporarily (vocab_size=151936 < scratch_dim*max_seq_len)
    __nv_bfloat16* scratch = pool_->scratch();
    kernels::matmul(scratch, last_hidden, weights_->lm_head(),
                    1, config_.vocab_size, config_.hidden_size);

    // Convert BF16 → FP32 for the sampler
    float* logits = pool_->logits();
    kernels::bf16_to_fp32(logits, scratch, config_.vocab_size);
    return logits;
}
