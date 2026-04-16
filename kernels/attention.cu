#include "kernels.h"
#include <cfloat>
#include <stdexcept>
#include <string>

// Sink + Sliding-Window GQA attention kernel.
//
// Grid : (seq_len, num_heads) — one block per (query_token, query_head).
// Block: BLOCK_DIM threads.
// Dynamic shared memory layout:
//   scores[n_sink + window_size]   — QK dot products (only first `total` used)
//   rdbuf [BLOCK_DIM]              — tree-reduction scratch
//
// For query token qi, the set of attended KV positions is:
//   Sink   : logical 0 .. sink_len-1          (physical slot = logical pos)
//   Window : logical oldest_w .. attend_pos   (physical slot via ring formula)
//
// attend_pos  = cache_len - seq_len + qi   (causal mask upper bound, inclusive)
// sink_len    = min(n_sink, attend_pos + 1)
// window_len  = min(max(attend_pos + 1 - n_sink, 0), window_size)
// oldest_w    = attend_pos + 1 - window_len   (oldest logical pos in window)
// win_offset  = (oldest_w - n_sink) % window_size  (ring buffer start slot offset)
//
// Physical slot for window token w (0 = oldest, window_len-1 = newest):
//   n_sink + (win_offset + w) % window_size

__global__ void attn_kernel(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale,
    int n_sink, int window_size
) {
    int qi   = blockIdx.x;
    int h    = blockIdx.y;
    int kv_h = h / (num_heads / num_kv_heads);

    int attend_pos = cache_len - seq_len + qi;
    int sink_len   = min(n_sink, attend_pos + 1);
    int window_len = min(max(attend_pos + 1 - n_sink, 0), window_size);
    int total      = sink_len + window_len;

    // Oldest logical position in the window that is still stored.
    // Physical ring offset: where the oldest window token sits in the ring.
    int oldest_w   = attend_pos + 1 - window_len;   // only used when window_len > 0
    int win_offset = (window_len > 0) ? (oldest_w - n_sink) % window_size : 0;

    // Shared memory: scores[n_sink + window_size] | rdbuf[BLOCK_DIM]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* rdbuf  = sdata + n_sink + window_size;

    const __nv_bfloat16* q_ptr = q + ((size_t)qi * num_heads + h) * head_dim;

    // ── Step 1: QK dot products ──────────────────────────────────────────────

    // 1a. Sink tokens (physical slot == logical pos, no ring logic)
    for (int s = threadIdx.x; s < sink_len; s += blockDim.x) {
        const __nv_bfloat16* k_ptr =
            k_pool + ((size_t)s * num_kv_heads + kv_h) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        scores[s] = dot * scale;
    }

    // 1b. Window tokens (physical slot via ring formula)
    for (int w = threadIdx.x; w < window_len; w += blockDim.x) {
        int phys = n_sink + (win_offset + w) % window_size;
        const __nv_bfloat16* k_ptr =
            k_pool + ((size_t)phys * num_kv_heads + kv_h) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        scores[sink_len + w] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: softmax over scores[0..total-1] ──────────────────────────────

    // 2a. max reduction
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        local_max = fmaxf(local_max, scores[i]);
    rdbuf[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            rdbuf[threadIdx.x] = fmaxf(rdbuf[threadIdx.x], rdbuf[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = rdbuf[0];

    // 2b. exp + sum
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        float e = expf(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    rdbuf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) rdbuf[threadIdx.x] += rdbuf[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.f / rdbuf[0];

    // 2c. normalize
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        scores[i] *= inv_sum;
    __syncthreads();

    // ── Step 3: weighted sum of V ────────────────────────────────────────────
    __nv_bfloat16* out_ptr = out + ((size_t)qi * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.f;

        // 3a. Sink tokens
        for (int s = 0; s < sink_len; s++) {
            const __nv_bfloat16* v_ptr =
                v_pool + ((size_t)s * num_kv_heads + kv_h) * head_dim;
            val += scores[s] * __bfloat162float(v_ptr[d]);
        }

        // 3b. Window tokens
        for (int w = 0; w < window_len; w++) {
            int phys = n_sink + (win_offset + w) % window_size;
            const __nv_bfloat16* v_ptr =
                v_pool + ((size_t)phys * num_kv_heads + kv_h) * head_dim;
            val += scores[sink_len + w] * __bfloat162float(v_ptr[d]);
        }

        out_ptr[d] = __float2bfloat16(val);
    }
}

namespace kernels {

void attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale,
    int n_sink, int window_size
) {
    constexpr int BLOCK_DIM = 128;

    // Shared memory: scores[n_sink + window_size] + rdbuf[BLOCK_DIM]
    size_t smem = ((size_t)(n_sink + window_size) + BLOCK_DIM) * sizeof(float);

    static bool smem_configured = false;
    if (!smem_configured) {
        // SM 8.7 (Jetson Orin) allows up to 164 KB dynamic shared memory.
        // With default n_sink=4, window=2044: smem = (2048+128)*4 = 8.7 KB — no issue.
        cudaFuncSetAttribute(attn_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             163840);
        smem_configured = true;
    }

    dim3 grid(seq_len, num_heads);
    attn_kernel<<<grid, BLOCK_DIM, smem>>>(
        out, q, k_pool, v_pool,
        seq_len, cache_len,
        num_heads, num_kv_heads,
        head_dim, scale,
        n_sink, window_size);
}

} // namespace kernels
