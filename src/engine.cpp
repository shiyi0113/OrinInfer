#include "engine.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <stdexcept>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
} while(0)

Engine::Engine(const std::string& model_dir) {
    init(model_dir);
}

Engine::~Engine() {
    if (d_tokens_) cudaFree(d_tokens_);
    delete tokenizer_;
    delete loader_;
}

void Engine::init(const std::string& model_dir) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Parse config.json + mmap safetensors + parse tokenizer.json
    loader_ = new ModelLoader(model_dir);
    config_ = loader_->config();

    // 2. Build tokenizer from parsed vocab/merges (host-only)
    tokenizer_ = new Tokenizer(loader_->tokenizer_data());

    // 3. cudaMemcpy all weights to GPU (one-time bulk transfer)
    weights_.load(*loader_, config_);

    // 4. Pre-allocate GPU activation buffers
    pool_.init(config_);

    // 5. Pre-allocate GPU KV cache ring buffers
    cache_.init(config_);

    // 6. Wire model
    model_.init(config_, weights_, pool_, cache_);

    // 7. Device buffer for input token ids
    CUDA_CHECK(cudaMalloc(&d_tokens_, config_.max_seq_len * sizeof(int32_t)));

    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Engine] Init complete in " << init_ms << " ms\n";
}

// Internal: run prefill + decode given pre-tokenized prompt ids.
std::string Engine::run_generation(const std::vector<int32_t>& prompt_tokens,
                                    GenerateConfig gen_config) {
    int prompt_len = static_cast<int>(prompt_tokens.size());

    if (prompt_len == 0) return "";
    if (prompt_len >= config_.max_seq_len) {
        std::cerr << "[Engine] Prompt too long (" << prompt_len
                  << " >= " << config_.max_seq_len << ")\n";
        return "";
    }

    cache_.reset();
    pool_.reset();

    Sampler sampler(config_.vocab_size, gen_config.sampler);

    CUDA_CHECK(cudaMemcpy(d_tokens_, prompt_tokens.data(),
                          prompt_len * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    auto t_start = std::chrono::high_resolution_clock::now();

    // ── Prefill ──────────────────────────────────────────
    float* logits = model_.forward(d_tokens_, prompt_len, 0);
    cache_.advance(prompt_len);
    int32_t next_token = sampler.sample(logits);

    auto t_prefill = std::chrono::high_resolution_clock::now();

    // ── Decode loop ──────────────────────────────────────
    std::vector<int32_t> output_tokens;
    output_tokens.push_back(next_token);

    if (gen_config.on_token) {
        std::string piece = tokenizer_->decode_token(next_token);
        if (!gen_config.on_token(piece)) goto done;
    }

    for (int step = 0; step < gen_config.max_new_tokens - 1; step++) {
        if (next_token == config_.eos_id || next_token == config_.im_end_id)
            break;

        CUDA_CHECK(cudaMemcpy(d_tokens_, &next_token, sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        int pos = prompt_len + static_cast<int>(output_tokens.size()) - 1;
        logits = model_.forward(d_tokens_, 1, pos);
        cache_.advance(1);

        next_token = sampler.sample(logits);
        output_tokens.push_back(next_token);

        if (gen_config.on_token) {
            std::string piece = tokenizer_->decode_token(next_token);
            if (!gen_config.on_token(piece)) break;
        }
    }

done:
    auto t_end = std::chrono::high_resolution_clock::now();

    if (gen_config.print_stats) {
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        double decode_ms  = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
        int gen_tokens = static_cast<int>(output_tokens.size());

        std::cerr << "\n[Stats] prompt=" << prompt_len << " tok"
                  << " | prefill=" << prefill_ms << "ms"
                  << " (" << (prompt_len * 1000.0 / prefill_ms) << " tok/s)"
                  << " | decode=" << gen_tokens << " tok in " << decode_ms << "ms"
                  << " (" << (gen_tokens * 1000.0 / decode_ms) << " tok/s)\n";
    }

    return tokenizer_->decode(output_tokens);
}

std::string Engine::generate(const std::string& prompt, GenerateConfig gen_config) {
    // Raw text prompt: encode directly (no chat template)
    return run_generation(tokenizer_->encode(prompt), gen_config);
}

std::string Engine::chat(const std::string& user_message, GenerateConfig config) {
    // Apply chat template to get token ids directly — avoids decode→encode roundtrip
    // which would silently drop special tokens (id >= 151643).
    auto tokens = tokenizer_->apply_chat_template(user_message, /*enable_thinking=*/false);
    return run_generation(tokens, config);
}

void Engine::print_info() const {
    std::cout << "╔══════════════════════════════════════╗\n"
              << "║  Qwen3-0.6B Inference Engine         ║\n"
              << "╠══════════════════════════════════════╣\n"
              << "║  Layers:       " << config_.num_layers << "                    ║\n"
              << "║  Hidden:       " << config_.hidden_size << "                  ║\n"
              << "║  Heads (Q/KV): " << config_.num_heads << "/" << config_.num_kv_heads << "                  ║\n"
              << "║  Vocab:        " << config_.vocab_size << "                ║\n"
              << "║  Max seq:      " << config_.max_seq_len << "                  ║\n"
              << "║  Format:       safetensors (FP16)    ║\n"
              << "╚══════════════════════════════════════╝\n";
}
