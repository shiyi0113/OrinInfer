#include "model_loader.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_loader <model_dir>\n"
                  << "  model_dir should contain config.json, model.safetensors, tokenizer.json\n";
        return 1;
    }

    ModelLoader loader(argv[1]);

    // Print config
    auto& c = loader.config();
    std::cout << "\n=== Model Config ===\n"
              << "hidden_size       = " << c.hidden_size << "\n"
              << "num_layers        = " << c.num_layers << "\n"
              << "num_heads         = " << c.num_heads << "\n"
              << "num_kv_heads      = " << c.num_kv_heads << "\n"
              << "head_dim          = " << c.head_dim << "\n"
              << "intermediate_size = " << c.intermediate_size << "\n"
              << "vocab_size        = " << c.vocab_size << "\n"
              << "rope_theta        = " << c.rope_theta << "\n"
              << "tie_embeddings    = " << c.tie_embeddings << "\n";

    // Print all tensors
    std::cout << "\n=== Tensors ===\n";
    for (auto& name : loader.tensor_names()) {
        auto* t = loader.get_tensor(name);
        std::cout << name << "  dtype=" << t->dtype << "  shape=[";
        for (size_t i = 0; i < t->shape.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << t->shape[i];
        }
        std::cout << "]  " << (t->byte_size / 1024) << " KB\n";
    }

    // Print tokenizer info
    auto& tok = loader.tokenizer_data();
    std::cout << "\n=== Tokenizer ===\n"
              << "vocab size = " << tok.vocab.size() << "\n"
              << "merges     = " << tok.merges.size() << "\n"
              << "bos_id     = " << tok.bos_id << "\n"
              << "eos_id     = " << tok.eos_id << "\n"
              << "im_start   = " << tok.im_start_id << "\n"
              << "im_end     = " << tok.im_end_id << "\n";

    // Print first 5 and last 5 vocab entries
    std::cout << "\nVocab sample (first 5):\n";
    for (int i = 0; i < 5 && i < (int)tok.vocab.size(); i++)
        std::cout << "  " << i << ": \"" << tok.vocab[i] << "\"\n";
    std::cout << "Vocab sample (last 5 specials):\n";
    for (int i = std::max(0, (int)tok.vocab.size() - 5); i < (int)tok.vocab.size(); i++)
        std::cout << "  " << i << ": \"" << tok.vocab[i] << "\"\n";

    return 0;
}
