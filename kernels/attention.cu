#include "kernels.h"
#include <cfloat>
#include <stdexcept>
#include <string>

// Paged GQA attention kernel.
//
// Grid : (seq_len, num_heads) — one block per (query_token, query_head).
// Block: BLOCK_DIM threads.
// Dynamic shared memory: (cache_len + BLOCK_DIM) floats.
//
// Causal rule: query token qi (0-indexed within current batch) may attend to
//   logical KV positions 0 .. cache_len - seq_len + qi   (inclusive).
//   attend_len = cache_len - seq_len + qi + 1
//
// KV pool layout per layer: [total_pages * block_size, num_kv_heads, head_dim]
// Q layout:                 [seq_len, num_heads, head_dim]
// Output layout:            [seq_len, num_heads, head_dim]
//
// The block_table maps logical_block → physical_page.
// Physical slot for logical position j:
//   page = block_table[j / block_size]
//   slot = j % block_size
//   row  = page * block_size + slot
__global__ void paged_attn_kernel(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
    const int32_t*       block_table,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale,
    int block_size
) {
    int qi   = blockIdx.x;
    int h    = blockIdx.y;
    int kv_h = h / (num_heads / num_kv_heads);   // GQA head mapping

    int attend_len = cache_len - seq_len + qi + 1;

    // Shared memory: scores[cache_len] | rdbuf[BLOCK_DIM]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* rdbuf  = sdata + cache_len;

    const __nv_bfloat16* q_ptr = q + ((size_t)qi * num_heads + h) * head_dim;

    // ── Step 1: QK dot products ──────────────────────────────────────────────
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x) {
        int page = block_table[j / block_size];
        int slot = j % block_size;
        const __nv_bfloat16* k_ptr =
            k_pool + ((size_t)(page * block_size + slot) * num_kv_heads + kv_h) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++)
            s += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        scores[j] = s * scale;
    }
    __syncthreads();

    // ── Step 2: softmax ──────────────────────────────────────────────────────
    // 2a. max reduction
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

    // 2b. exp + sum
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

    // 2c. normalize
    for (int j = threadIdx.x; j < attend_len; j += blockDim.x)
        scores[j] *= inv_sum;
    __syncthreads();

    // ── Step 3: weighted sum of V ────────────────────────────────────────────
    __nv_bfloat16* out_ptr = out + ((size_t)qi * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            int page = block_table[j / block_size];
            int slot = j % block_size;
            const __nv_bfloat16* v_ptr =
                v_pool + ((size_t)(page * block_size + slot) * num_kv_heads + kv_h) * head_dim;
            val += scores[j] * __bfloat162float(v_ptr[d]);
        }
        out_ptr[d] = __float2bfloat16(val);
    }
}

namespace kernels {

// SM 8.7 (Jetson Orin) supports up to 164 KB shared memory per block.
// scores[cache_len] + rdbuf[BLOCK_DIM] in float32.
// 160 KB headroom ≈ 40832 tokens of context.
static constexpr size_t ATTN_MAX_SMEM = 163840;  // 160 KB

void paged_attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
    const int32_t* block_table,
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads,
    int head_dim, float scale,
    int block_size
) {
    constexpr int BLOCK_DIM = 128;
    size_t smem = ((size_t)cache_len + BLOCK_DIM) * sizeof(float);

    if (smem > ATTN_MAX_SMEM)
        throw std::runtime_error(
            "[paged_attention] context too long for shared-memory kernel: "
            + std::to_string(cache_len) + " tokens");

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(paged_attn_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             ATTN_MAX_SMEM);
        smem_configured = true;
    }

    dim3 grid(seq_len, num_heads);
    paged_attn_kernel<<<grid, BLOCK_DIM, smem>>>(
        out, q, k_pool, v_pool, block_table,
        seq_len, cache_len,
        num_heads, num_kv_heads,
        head_dim, scale, block_size);
}

} // namespace kernels
