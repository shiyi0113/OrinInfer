#include "sampler.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

Sampler::Sampler(int vocab_size, SamplerConfig config)
    : vocab_size_(vocab_size), config_(config) {
    // Allocate single int32 on device for result
    CUDA_CHECK(cudaMalloc(&d_token_, sizeof(int32_t)));
}

Sampler::~Sampler() {
    if (d_token_) cudaFree(d_token_);
}

int32_t Sampler::sample(float* d_logits) {
    // Everything happens on GPU — only the final int32 crosses PCIe
    switch (config_.method) {
        case SampleMethod::GREEDY:
            kernels::argmax(d_token_, d_logits, vocab_size_);
            break;
        case SampleMethod::TOP_P:
            kernels::top_p_sample(d_token_, d_logits, vocab_size_,
                                  config_.temperature, config_.top_p,
                                  config_.seed++);
            break;
    }

    // Single cudaMemcpy: 4 bytes, device → host
    int32_t token;
    CUDA_CHECK(cudaMemcpy(&token, d_token_, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return token;
}
