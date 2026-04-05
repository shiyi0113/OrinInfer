#include "kernels.h"
#include <cfloat>
#include <stdexcept>
#include <string>

// GQA attention kernel.
// Grid: (seq_len, num_heads) — one block per (query_token, query_head).
// Block: BLOCK_DIM threads.
// Dynamic shared memory: (cache_len + BLOCK_DIM) floats.
//
// Causal rule: query token qi (0-indexed within current batch) has
//   absolute position = cache_len - seq_len + qi
//   and may attend to KV positions 0 .. cache_len - seq_len + qi
//   → attend_len = cache_len - seq_len + qi + 1
//
// KV cache layout (per layer): [max_seq_len, num_kv_heads, head_dim]
// Q layout:                    [seq_len,     num_heads,    head_dim]
// Output layout:               [seq_len,     num_heads,    head_dim]
__global__ void attn_kernel(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale
) {
    int qi   = blockIdx.x;          // query token index within batch
    int h    = blockIdx.y;          // Q head index
    int kv_h = h / (num_heads / num_kv_heads);  // GQA: mapped KV head

    // Number of KV positions this token can attend to
    int attend_len = cache_len - seq_len + qi + 1;

    // Shared memory layout: [attend_len] scores then [blockDim.x] reduction buf
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* rdbuf  = sdata + cache_len;  // fixed offset so pointer is block-uniform

    // Pointer to this query vector
    const __nv_bfloat16* q_ptr = q + (qi * num_heads + h) * head_dim;

    // ── Step 1: compute dot-product scores ──────────────────
    // Each thread computes scores for j = threadIdx.x, threadIdx.x+BLOCK, ...
    // Inner loop over head_dim is sequential (128 ops per dot product).
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x) {
        const __nv_bfloat16* k_ptr = k_cache + ((size_t)j * num_kv_heads + kv_h) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            s += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        scores[j] = s * scale;
    }
    __syncthreads();

    // ── Step 2: softmax over attend_len scores ───────────────
    // 2a. Find max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    rdbuf[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            rdbuf[threadIdx.x] = fmaxf(rdbuf[threadIdx.x], rdbuf[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = rdbuf[0];

    // 2b. Exp and sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    rdbuf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) rdbuf[threadIdx.x] += rdbuf[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / rdbuf[0];

    // 2c. Normalize
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x)
        scores[j] *= inv_sum;
    __syncthreads();

    // ── Step 3: weighted sum of V ────────────────────────────
    // Thread d handles output dimension d (blockDim.x == head_dim for clean tiling).
    __nv_bfloat16* out_ptr = out + (qi * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            const __nv_bfloat16* v_ptr = v_cache + ((size_t)j * num_kv_heads + kv_h) * head_dim;
            val += scores[j] * __bfloat162float(v_ptr[d]);
        }
        out_ptr[d] = __float2bfloat16(val);
    }
}

namespace kernels {

// Max shared memory per block on SM 8.7 (Jetson Orin) is 164 KB.
// scores[cache_len] + rdbuf[BLOCK_DIM] in float32.
// Supports up to (164*1024/4 - 128) ≈ 41984 tokens per context.
static constexpr size_t ATTN_MAX_SMEM = 163840; // 160 KB, leave 4 KB headroom

void attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale
) {
    constexpr int BLOCK_DIM = 128;
    dim3 grid(seq_len, num_heads);
    size_t smem = ((size_t)cache_len + BLOCK_DIM) * sizeof(float);

    if (smem > ATTN_MAX_SMEM) {
        // Context too long for shared-memory attention — would need Flash Attention
        throw std::runtime_error("[attention] context too long for shared-memory kernel: "
                                 + std::to_string(cache_len) + " tokens");
    }

    // Unlock large shared memory (default cap is 48 KB; SM 8.7 supports up to 164 KB)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(attn_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             ATTN_MAX_SMEM);
        smem_configured = true;
    }

    attn_kernel<<<grid, BLOCK_DIM, smem>>>(
        out, q, k_cache, v_cache,
        seq_len, cache_len, num_heads, num_kv_heads, head_dim, scale);
}

} // namespace kernels
