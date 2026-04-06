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
    if (pool_base_) cudaFree(pool_base_); 
}

void ActivationPool::init(const ModelConfig& config) {
    // 1. 计算各个部分的大小
    int max_dim = std::max({config.hidden_size, config.q_dim(), config.intermediate_size});
    size_t buf_elements = (size_t)config.max_seq_len * max_dim;
    size_t buf_bytes = buf_elements * sizeof(__nv_bfloat16);

    int scratch_dim = std::max(config.q_dim() + 2 * config.kv_dim(), 2 * config.intermediate_size);
    size_t scratch_elements = (size_t)config.max_seq_len * scratch_dim;
    size_t scratch_bytes = scratch_elements * sizeof(__nv_bfloat16);

    size_t logits_bytes = (size_t)config.vocab_size * sizeof(float);

    // 2. 合并计算总字节数（注意对齐，虽然目前类型自然对齐）
    size_t total_bytes = 2 * buf_bytes + scratch_bytes + logits_bytes;

    // 3. 执行唯一的一次 cudaMalloc
    CUDA_CHECK(cudaMalloc(&pool_base_, total_bytes));
    CUDA_CHECK(cudaMemset(pool_base_, 0, total_bytes));

    // 4. 按偏移量切分指针
    char* ptr = static_cast<char*>(pool_base_);
    
    buf_a_   = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr     += buf_bytes;
    
    buf_b_   = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr     += buf_bytes;
    
    scratch_ = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr     += scratch_bytes;
    
    logits_  = reinterpret_cast<float*>(ptr);

    std::cout << "[ActivationPool] Combined allocation: " << (total_bytes >> 20) << " MB\n";
}