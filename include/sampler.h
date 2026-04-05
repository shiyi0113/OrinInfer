#pragma once
#include <cstdint>

enum class SampleMethod {
    GREEDY,     // argmax
    TOP_P,      // nucleus sampling
};

struct SamplerConfig {
    SampleMethod method = SampleMethod::GREEDY;
    float temperature   = 1.0f;
    float top_p         = 0.9f;
    unsigned long long seed = 42;
};

class Sampler {
public:
    explicit Sampler(int vocab_size, SamplerConfig config = {});
    ~Sampler();

    // Sample next token from logits (all on GPU).
    // logits: device pointer [vocab_size] in fp32
    // Returns: token id (copied to host, the ONLY data that crosses PCIe)
    int32_t sample(float* d_logits);

private:
    int vocab_size_;
    SamplerConfig config_;
    int32_t* d_token_;       // device-side single int32 for result
};
