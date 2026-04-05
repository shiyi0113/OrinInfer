#pragma once
#include "config.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "weight_store.h"
#include "activation_pool.h"
#include "kv_cache.h"
#include "model.h"
#include "sampler.h"
#include <string>
#include <functional>

struct GenerateConfig {
    int max_new_tokens       = 256;
    SamplerConfig sampler    = {};
    bool print_stats         = true;

    // Callback for streaming output (return false to stop early)
    std::function<bool(const std::string& piece)> on_token = nullptr;
};

class Engine {
public:
    // model_dir: path to HF model directory containing
    //   config.json, model.safetensors, tokenizer.json
    explicit Engine(const std::string& model_dir);
    ~Engine();

    std::string generate(const std::string& prompt, GenerateConfig config = {});
    std::string chat(const std::string& user_message, GenerateConfig config = {});
    void print_info() const;

private:
    void init(const std::string& model_dir);

    ModelConfig     config_;
    ModelLoader*    loader_    = nullptr;
    Tokenizer*      tokenizer_ = nullptr;
    WeightStore     weights_;
    ActivationPool  pool_;
    KVCache         cache_;
    Model           model_;
    int32_t*        d_tokens_  = nullptr;
};
