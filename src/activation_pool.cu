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

    // Scratch: reused for QKV projections and FFN gate/up.
    // QKV needs: seq * (q_dim + kv_dim + kv_dim) = seq * (2048+1024+1024) = seq * 4096
    // FFN needs: seq * 2 * intermediate_size      = seq * 6144   (larger)
    int scratch_dim = std::max(config.q_dim() + 2 * config.kv_dim(),
                               2 * config.intermediate_size);
    size_t scratch_bytes = (size_t)config.max_seq_len * scratch_dim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&scratch_, scratch_bytes));

    // Logits buffer in fp32 for numerical stability
    CUDA_CHECK(cudaMalloc(&logits_, config.vocab_size * sizeof(float)));

    size_t logits_bytes = (size_t)config.vocab_size * sizeof(float);
    size_t total = 2 * buf_bytes + scratch_bytes + logits_bytes;
    std::cout << "[ActivationPool] ping-pong=" << (2*buf_bytes>>20) << " MB"
              << "  scratch=" << (scratch_bytes>>20) << " MB"
              << "  logits=" << (logits_bytes>>20) << " MB"
              << "  total=" << (total>>20) << " MB\n";
}
