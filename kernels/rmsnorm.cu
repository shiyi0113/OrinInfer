#include "kernels.h"
#include <cuda_bf16.h>

// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
// Each block handles one row (one token position).
// Shared memory warp reduction for sum-of-squares.
__global__ void rmsnorm_kernel(__nv_bfloat16* out, const __nv_bfloat16* x,
                                const __nv_bfloat16* weight, int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + row * dim;
    __nv_bfloat16* o_row = out + row * dim;

    extern __shared__ float sdata[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        local_ss += val * val;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]) * rms * __bfloat162float(weight[i]);
        o_row[i] = __float2bfloat16(val);
    }
}

namespace kernels {
void rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
             int seq_len, int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    rmsnorm_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
        out, x, weight, dim, eps);
}
}
