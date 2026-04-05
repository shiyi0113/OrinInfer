#include "kernels.h"
#include <cfloat>

// In-place softmax over last dimension (fp32 for numerical stability)
__global__ void softmax_kernel(float* x, int cols) {
    int row = blockIdx.x;
    float* row_ptr = x + row * cols;

    extern __shared__ float sdata[];

    // 1. Find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    // 3. Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_ptr[i] /= sum;
    }
}

namespace kernels {
void softmax(float* x, int rows, int cols) {
    int threads = (cols < 1024) ? cols : 1024;
    softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(x, cols);
}

void argmax(int32_t* d_out_token, const float* logits, int vocab_size) {
    // TODO: Implement parallel reduction argmax
    // Simple approach: use one block, each thread scans a chunk,
    //                  shared memory reduction for max
    // For now, placeholder
}

void top_p_sample(int32_t* d_out_token, float* logits, int vocab_size,
                  float temperature, float top_p, unsigned long long seed) {
    // TODO: Implement nucleus sampling on GPU
    // 1. Apply temperature: logits /= temperature
    // 2. Softmax
    // 3. Sort probabilities (use thrust::sort or bitonic sort)
    // 4. Cumulative sum, find cutoff at top_p
    // 5. Re-normalize and sample
}
}
