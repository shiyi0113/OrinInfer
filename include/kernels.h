#pragma once
#include <cuda_bf16.h>
#include <cstdint>

// All kernel functions take raw pointers + dimensions.
// No classes, no dependencies on upper layers.
// seq_len=1 → decode path, seq_len>1 → prefill path.
// Kernel dispatch is automatic based on dimensions.

namespace kernels {

// ── embedding lookup ─────────────────────────────────────────
// out[seq_len, hidden_size] = table[token_ids[i], :]
void embedding(
    __nv_bfloat16* out,
    const __nv_bfloat16* table,       // [vocab_size, hidden_size]
    const int32_t* token_ids,// [seq_len]
    int seq_len,
    int hidden_size
);

// ── RMS normalization ────────────────────────────────────────
// out[seq_len, dim] = rmsnorm(x[seq_len, dim], weight[dim], eps)
void rmsnorm(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,      // [dim]
    int seq_len,
    int dim,
    float eps
);

// ── RoPE (rotary positional embedding) ───────────────────────
// Apply RoPE in-place to q[seq_len, num_heads, head_dim]
//                      and k[seq_len, num_kv_heads, head_dim]
void rope(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,           // position offset for decode
    float theta
);

// ── matmul (auto GEMM/GEMV dispatch) ─────────────────────────
// C[M, N] = A[M, K] * B[N, K]^T    (B is stored transposed)
// When M=1, internally dispatches to optimized GEMV kernel
// When M>1, dispatches to GEMM (or cuBLAS)
void matmul(
    __nv_bfloat16* C,
    const __nv_bfloat16* A,           // [M, K]
    const __nv_bfloat16* B,           // [N, K] (row-major, transposed multiply)
    int M, int N, int K
);

// ── attention ────────────────────────────────────────────────
// Grouped-Query Attention with KV cache
// q:      [seq_len, num_heads, head_dim]
// k_cache: [cache_len, num_kv_heads, head_dim]
// v_cache: [cache_len, num_kv_heads, head_dim]
// out:    [seq_len, num_heads, head_dim]
void attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    int seq_len,             // 1 for decode, N for prefill
    int cache_len,           // total valid KV entries
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale              // 1/sqrt(head_dim)
);

// ── fused SwiGLU ─────────────────────────────────────────────
// out[seq_len, hidden] = down_proj(SiLU(gate_proj(x)) * up_proj(x))
// Fuses gate+up projection, SiLU activation, element-wise mul,
// and down projection into minimal memory passes.
// Step 1 (call separately): gate = matmul(x, gate_proj), up = matmul(x, up_proj)
// Step 2 (this kernel):     out = SiLU(gate) * up   (element-wise, in-place possible)
void fused_silu_mul(
    __nv_bfloat16* out,
    const __nv_bfloat16* gate,        // [seq_len, intermediate_size]
    const __nv_bfloat16* up,          // [seq_len, intermediate_size]
    int seq_len,
    int intermediate_size
);

// ── element-wise residual add (in-place: x += y) ─────────────
void add(
    __nv_bfloat16* x,
    const __nv_bfloat16* y,
    int n
);

// ── BF16 → FP32 conversion ───────────────────────────────────
void bf16_to_fp32(
    float* out,
    const __nv_bfloat16* in,
    int n
);

// ── softmax ──────────────────────────────────────────────────
// In-place softmax along last dimension
void softmax(
    float* x,                // [rows, cols], fp32 for numerical stability
    int rows,
    int cols
);

// ── argmax (GPU-side sampler) ────────────────────────────────
// Returns the index of the maximum value in logits[vocab_size]
// Result is written to d_out_token (device memory, single int32)
void argmax(
    int32_t* d_out_token,    // device pointer, single int32
    const float* logits,     // [vocab_size]
    int vocab_size
);

// ── top-p sampling (GPU-side sampler) ────────────────────────
// Nucleus sampling with temperature
void top_p_sample(
    int32_t* d_out_token,    // device pointer, single int32
    float* logits,           // [vocab_size], modified in-place
    int vocab_size,
    float temperature,
    float top_p,
    unsigned long long seed
);

} // namespace kernels
