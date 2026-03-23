#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

#include "llama.h"

// 测试文本 - 重复多次以获得更准确的测量
const char* TEST_TEXT =
    "The quick brown fox jumps over the lazy dog. "
    "人工智能是当今科技领域最热门的话题之一。"
    "Machine learning models train in the background processing vast amounts of data. "
    "深度学习、机器学习、神经网络这些概念已经不再陌生。";

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Simple Tokenizer Performance Test                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    const char* model_path = argv[1];

    // 加载模型
    std::cout << "Loading model: " << model_path << " ...\n";
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(model_path, params);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::cout << "Vocabulary size: " << n_vocab << "\n";
    std::cout << "Test text length: " << strlen(TEST_TEXT) << " chars\n\n";

    // 准备 token 缓冲区
    std::vector<llama_token> tokens;
    tokens.resize(strlen(TEST_TEXT) * 4);

    // 预热
    int n = llama_tokenize(vocab, TEST_TEXT, strlen(TEST_TEXT),
                           tokens.data(), tokens.size(), false, false);
    std::cout << "Warmup: " << n << " tokens\n\n";

    // 性能测试
    const int ITERATIONS = 1000;
    std::cout << "Running " << ITERATIONS << " iterations...\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        n = llama_tokenize(vocab, TEST_TEXT, strlen(TEST_TEXT),
                           tokens.data(), tokens.size(), false, false);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERATIONS;
    double tokens_per_sec = (n * ITERATIONS / total_ms) * 1000.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n";
    std::cout << "Results:\n";
    std::cout << "  Total time:      " << total_ms << " ms\n";
    std::cout << "  Average time:    " << avg_ms << " ms/tokenization\n";
    std::cout << "  Tokens/sec:      " << tokens_per_sec << "\n";
    std::cout << "  Tokens/output:   " << n << "\n";

    // Detokenize 测试
    std::cout << "\nDetokenize test:\n";
    std::string output;
    output.resize(strlen(TEST_TEXT) * 2);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        llama_detokenize(vocab, tokens.data(), n, &output[0], output.size(), false, false);
    }
    end = std::chrono::high_resolution_clock::now();

    total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Detokenize time: " << total_ms << " ms (" << ITERATIONS << " iterations)\n";
    std::cout << "  Output: " << output.substr(0, 50) << "...\n";

    llama_model_free(model);

    std::cout << "\nDone!\n";
    return 0;
}
