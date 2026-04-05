#include "kernels.h"
#include <cuda_bf16.h>

// Fused SwiGLU: out = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
__global__ void fused_silu_mul_kernel(__nv_bfloat16* out, const __nv_bfloat16* gate,
                                      const __nv_bfloat16* up, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);

    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2bfloat16(silu_g * u);
}

namespace kernels {
void fused_silu_mul(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                    int seq_len, int intermediate_size) {
    int total   = seq_len * intermediate_size;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    fused_silu_mul_kernel<<<blocks, threads>>>(out, gate, up, total);
}
}
