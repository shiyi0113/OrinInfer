#include "kernels.h"
#include <cuda_bf16.h>

__global__ void embedding_kernel(__nv_bfloat16* out, const __nv_bfloat16* table,
                                  const int32_t* ids, int hidden_size) {
    int token_idx = blockIdx.x;
    int dim_idx   = threadIdx.x + blockIdx.y * blockDim.x;
    if (dim_idx >= hidden_size) return;

    int32_t id = ids[token_idx];
    out[token_idx * hidden_size + dim_idx] = table[id * hidden_size + dim_idx];
}

namespace kernels {
void embedding(__nv_bfloat16* out, const __nv_bfloat16* table, const int32_t* token_ids,
               int seq_len, int hidden_size) {
    dim3 grid(seq_len, (hidden_size + 255) / 256);
    dim3 block(256);
    embedding_kernel<<<grid, block>>>(out, table, token_ids, hidden_size);
}
}
