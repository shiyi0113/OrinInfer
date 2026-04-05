#include "model_loader.h"
#include "tokenizer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_tokenizer <model_dir>\n";
        return 1;
    }

    ModelLoader loader(argv[1]);
    Tokenizer tokenizer(loader.tokenizer_data());

    std::vector<std::string> tests = {
        "Hello, world!",
        "你好世界",
        "The quick brown fox jumps over the lazy dog.",
        "1+1=2",
        "def fibonacci(n):",
    };

    for (auto& text : tests) {
        auto ids = tokenizer.encode(text);
        auto decoded = tokenizer.decode(ids);
        std::cout << "Input:   \"" << text << "\"\n"
                  << "Tokens:  [";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "] (" << ids.size() << " tokens)\n"
                  << "Decoded: \"" << decoded << "\"\n"
                  << "Match:   " << (text == decoded ? "OK" : "MISMATCH") << "\n\n";
    }

    // Test chat template
    auto chat_ids = tokenizer.apply_chat_template("What is 1+1?");
    std::cout << "=== Chat template ===\n"
              << "Tokens: " << chat_ids.size() << "\n"
              << "Decoded: " << tokenizer.decode(chat_ids) << "\n";

    return 0;
}
