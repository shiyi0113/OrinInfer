#include "kernels.h"

// TODO: Implement efficient RMSNorm kernel
// Key optimization: use warp-level reduction for sum-of-squares
// Formula: out = x * weight / sqrt(mean(x^2) + eps)
__global__ void rmsnorm_kernel(half* out, const half* x, const half* weight,
                                int dim, float eps) {
    // Each block handles one row (one token position)
    int row = blockIdx.x;
    const half* x_row = x + row * dim;
    half* o_row = out + row * dim;

    // 1. Compute sum of squares (shared memory reduction)
    extern __shared__ float sdata[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        local_ss += val * val;
    }
    sdata[threadIdx.x] = local_ss;
    __syncthreads();

    // Warp reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / dim + eps);

    // 2. Normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(x_row[i]) * rms * __half2float(weight[i]);
        o_row[i] = __float2half(val);
    }
}

namespace kernels {
void rmsnorm(half* out, const half* x, const half* weight,
             int seq_len, int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    rmsnorm_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
        out, x, weight, dim, eps);
}
}
