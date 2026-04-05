#include "engine.h"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_dir> [options]\n"
              << "  model_dir: HF directory with config.json, model.safetensors, tokenizer.json\n\n"
              << "Options:\n"
              << "  -p <prompt>      Input prompt (default: interactive mode)\n"
              << "  -n <max_tokens>  Max tokens to generate (default: 256)\n"
              << "  -t <temperature> Sampling temperature (default: 1.0)\n"
              << "  --greedy         Use greedy decoding (default)\n"
              << "  --top-p <value>  Use nucleus sampling\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::string prompt;
    GenerateConfig gen_config;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            gen_config.max_new_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            gen_config.sampler.temperature = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            gen_config.sampler.method = SampleMethod::GREEDY;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            gen_config.sampler.method = SampleMethod::TOP_P;
            gen_config.sampler.top_p = std::stof(argv[++i]);
        }
    }

    std::cout << "Loading model: " << model_dir << std::endl;
    Engine engine(model_dir);
    engine.print_info();

    gen_config.on_token = [](const std::string& piece) -> bool {
        std::cout << piece << std::flush;
        return true;
    };

    if (!prompt.empty()) {
        std::cout << "\n--- Generation ---\n";
        engine.chat(prompt, gen_config);
        std::cout << "\n";
    } else {
        std::cout << "\nInteractive mode. Type 'quit' to exit.\n";
        while (true) {
            std::cout << "\n> ";
            std::getline(std::cin, prompt);
            if (prompt == "quit" || prompt == "exit" || std::cin.eof()) break;
            if (prompt.empty()) continue;
            engine.chat(prompt, gen_config);
            std::cout << "\n";
        }
    }

    return 0;
}
