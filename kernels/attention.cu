#include "kernels.h"

// TODO: Implement GQA attention kernel
// Supports both prefill (seq_len=N, full causal mask) and decode (seq_len=1)
// GQA: multiple Q heads share same KV head (ratio = num_heads/num_kv_heads)
//
// For decode (seq_len=1):
//   score[h, j] = dot(q[h], k_cache[j, h/ratio]) * scale
//   Apply causal mask (j <= current_pos)
//   attn = softmax(score)
//   out[h] = sum_j(attn[j] * v_cache[j, h/ratio])
//
// For prefill (seq_len=N):
//   Same but with NxN score matrix per head, causal mask

namespace kernels {
void attention(half* out, const half* q, const half* k_cache, const half* v_cache,
               int seq_len, int cache_len, int num_heads, int num_kv_heads,
               int head_dim, float scale) {
    // TODO: Implement
    // Key decision: for decode, use a single kernel that does
    //   dot-product + softmax + weighted-sum in one pass (flash-style)
    // For prefill, tile the NxN attention matrix
}
}
