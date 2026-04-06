#include "weight_store.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>  // 引入 mmap
#include <sys/stat.h>  // 引入 fstat

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

WeightStore::~WeightStore() {
    if (mapped_addr_) {
        cudaHostUnregister(mapped_addr_);
        munmap(mapped_addr_, mapped_size_);
    }
}

void WeightStore::load(const ModelLoader& loader, const ModelConfig& config) {
    int fd = loader.fd();
    size_t base_file_offset = loader.data_offset();

    // 1. 获取整个文件的真实大小
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error("fstat failed!");
    }
    mapped_size_ = sb.st_size;

    std::cout << "  -> Mapping file of " << (mapped_size_ >> 20) << " MB...\n";

    // 2. 执行 mmap，从文件头(offset=0)开始映射
    // 使用 MAP_POPULATE 强制提前触发缺页中断，消除推理时的延迟。
    mapped_addr_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (mapped_addr_ == MAP_FAILED) {
        throw std::runtime_error("mmap failed!");
    }

    // 3. 将映射的系统内存注册给 CUDA 
    CUDA_CHECK(cudaHostRegister(mapped_addr_, mapped_size_, cudaHostRegisterReadOnly));

    // 4. 获取 GPU 视角的设备指针
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_ptr, mapped_addr_, 0));

    // 5. 定位到实际权重数据的起始基地址 (绕过 Safetensors Header)
    char* base_data_ptr = static_cast<char*>(d_ptr) + base_file_offset;

    // 6. 零成本切分指针
    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        auto* t = loader.get_tensor(name);
        if (!t) throw std::runtime_error("[WeightStore] missing tensor: " + name);
        return reinterpret_cast<__nv_bfloat16*>(base_data_ptr + t->offset);
    };

    // --- 组装结构体---
    embedding_ = load_tensor("model.embed_tokens.weight");

    layers_.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        auto& L = layers_[i];
        L.input_layernorm = load_tensor(p + ".input_layernorm.weight");
        L.q_proj          = load_tensor(p + ".self_attn.q_proj.weight");
        L.q_norm          = load_tensor(p + ".self_attn.q_norm.weight");
        L.k_proj          = load_tensor(p + ".self_attn.k_proj.weight");
        L.k_norm          = load_tensor(p + ".self_attn.k_norm.weight");
        L.v_proj          = load_tensor(p + ".self_attn.v_proj.weight");
        L.o_proj          = load_tensor(p + ".self_attn.o_proj.weight");
        L.post_attn_norm  = load_tensor(p + ".post_attention_layernorm.weight");
        L.gate_proj       = load_tensor(p + ".mlp.gate_proj.weight");
        L.up_proj         = load_tensor(p + ".mlp.up_proj.weight");
        L.down_proj       = load_tensor(p + ".mlp.down_proj.weight");
    }

    final_norm_ = load_tensor("model.norm.weight");
    lm_head_    = load_tensor("lm_head.weight");

    std::cout << "[WeightStore] Load complete.\n";
}