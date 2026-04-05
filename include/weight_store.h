#pragma once
#include "config.h"
#include "model_loader.h"
#include <cuda_bf16.h>
#include <vector>

// Per-layer weight pointers (all on GPU, read-only after init)
struct LayerWeights {
    __nv_bfloat16* input_layernorm;     // [hidden_size]
    __nv_bfloat16* q_proj;              // [q_dim, hidden_size]
    __nv_bfloat16* q_norm;              // [head_dim]
    __nv_bfloat16* k_proj;              // [kv_dim, hidden_size]
    __nv_bfloat16* k_norm;              // [head_dim]
    __nv_bfloat16* v_proj;              // [kv_dim, hidden_size]
    __nv_bfloat16* o_proj;              // [hidden_size, q_dim]
    __nv_bfloat16* post_attn_norm;      // [hidden_size]
    __nv_bfloat16* gate_proj;           // [intermediate_size, hidden_size]
    __nv_bfloat16* up_proj;             // [intermediate_size, hidden_size]
    __nv_bfloat16* down_proj;           // [hidden_size, intermediate_size]
};

class WeightStore {
public:
    WeightStore() = default;
    ~WeightStore();

    // Load all weights from ModelLoader into GPU (one-time init)
    void load(const ModelLoader& loader, const ModelConfig& config);

    const __nv_bfloat16* embedding()  const { return embedding_; }
    const __nv_bfloat16* final_norm() const { return final_norm_; }
    const __nv_bfloat16* lm_head()    const { return lm_head_; }
    const LayerWeights& layer(int i) const { return layers_[i]; }

private:
    __nv_bfloat16* alloc_and_copy(const TensorInfo* tensor);

    __nv_bfloat16* embedding_  = nullptr;
    __nv_bfloat16* final_norm_ = nullptr;
    __nv_bfloat16* lm_head_    = nullptr;
    std::vector<LayerWeights> layers_;
    std::vector<void*> all_allocs_;
};
