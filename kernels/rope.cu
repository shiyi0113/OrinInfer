#include "kernels.h"
#include <cmath>

// Each thread handles one (seq_pos, head, pair_index) triplet.
// In-place rotation: [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
// angle = abs_pos * theta^(-2*pair_idx / head_dim)
__global__ void rope_kernel(
    __nv_bfloat16* qk,      // [seq_len, num_heads, head_dim] in-place
    int seq_len,
    int num_heads,
    int head_dim,            // must be even
    int start_pos,
    float theta
) {
    int half_hd = head_dim >> 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * half_hd;
    if (tid >= total) return;

    int pair_idx = tid % half_hd;
    int head_idx = (tid / half_hd) % num_heads;
    int seq_idx  = tid / (half_hd * num_heads);

    int abs_pos  = start_pos + seq_idx;
    float angle  = abs_pos * powf(theta, -2.0f * pair_idx / (float)head_dim);
    float cos_a  = cosf(angle);
    float sin_a  = sinf(angle);

    int base = (seq_idx * num_heads + head_idx) * head_dim;
    int i0   = base + pair_idx;              // first half:  element p
    int i1   = base + pair_idx + half_hd;   // second half: element p + head_dim/2

    float x0 = __bfloat162float(qk[i0]);
    float x1 = __bfloat162float(qk[i1]);

    qk[i0] = __float2bfloat16(x0 * cos_a - x1 * sin_a);
    qk[i1] = __float2bfloat16(x0 * sin_a + x1 * cos_a);
}

namespace kernels {

void rope(__nv_bfloat16* q, __nv_bfloat16* k,
          int seq_len, int num_heads, int num_kv_heads,
          int head_dim, int start_pos, float theta)
{
    constexpr int THREADS = 256;
    int half_hd = head_dim >> 1;

    // Apply to Q
    int total_q = seq_len * num_heads * half_hd;
    rope_kernel<<<(total_q + THREADS - 1) / THREADS, THREADS>>>(
        q, seq_len, num_heads, head_dim, start_pos, theta);

    // Apply to K (same kernel, different num_heads)
    int total_k = seq_len * num_kv_heads * half_hd;
    rope_kernel<<<(total_k + THREADS - 1) / THREADS, THREADS>>>(
        k, seq_len, num_kv_heads, head_dim, start_pos, theta);
}

} // namespace kernels
