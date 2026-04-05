#include "kernels.h"

// Fused SwiGLU: out = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Fusing avoids extra memory round-trip for the intermediate SiLU result
__global__ void fused_silu_mul_kernel(half* out, const half* gate, const half* up,
                                      int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // SiLU(g) * u
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu_g * u);
}

namespace kernels {
void fused_silu_mul(half* out, const half* gate, const half* up,
                    int seq_len, int intermediate_size) {
    int total = seq_len * intermediate_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_silu_mul_kernel<<<blocks, threads>>>(out, gate, up, total);
}
}
