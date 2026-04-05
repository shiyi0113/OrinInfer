#include "activation_pool.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

ActivationPool::~ActivationPool() {
    if (buf_a_)   cudaFree(buf_a_);
    if (buf_b_)   cudaFree(buf_b_);
    if (scratch_)  cudaFree(scratch_);
    if (logits_)   cudaFree(logits_);
}

void ActivationPool::init(const ModelConfig& config) {
    // Main buffer size: max_seq_len * max(hidden_size, q_dim, intermediate_size)
    int max_dim = std::max({config.hidden_size, config.q_dim(), config.intermediate_size});
    size_t buf_bytes = (size_t)config.max_seq_len * max_dim * sizeof(__nv_bfloat16);

    CUDA_CHECK(cudaMalloc(&buf_a_, buf_bytes));
    CUDA_CHECK(cudaMalloc(&buf_b_, buf_bytes));

    // Scratch for attention scores, intermediate results
    size_t scratch_bytes = (size_t)config.max_seq_len * config.intermediate_size * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&scratch_, scratch_bytes));

    // Logits buffer in fp32 for numerical stability
    CUDA_CHECK(cudaMalloc(&logits_, config.vocab_size * sizeof(float)));

    std::cout << "[ActivationPool] Allocated "
              << (2 * buf_bytes + scratch_bytes + config.vocab_size * 4) / (1024*1024)
              << " MB GPU memory\n";
}
