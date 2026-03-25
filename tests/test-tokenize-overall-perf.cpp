// Test for overall llama_tokenize Performance
// Measures the complete tokenization pipeline with flat_hash_map

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include "llama.h"
#include "unicode.h"

const char* TEST_TEXTS[] = {
    "Hello, world! This is a simple test.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "人工智能是当今科技领域最热门的话题之一。深度学习、机器学习、神经网络这些概念已经不再陌生。",
    "AI technology 人工智能的发展 has brought significant changes 对我们的社会带来了重大变化。",
    "In the heart of the city, where skyscrapers touch the clouds and neon lights illuminate the streets, a lone programmer sits at their desk, coding late into the night. The screen glows with lines of code, each character a building block of digital creation. 人工智能技术正在改变着我们的生活方式。"
};

const char* TEST_NAMES[] = {
    "短英文",
    "中等英文",
    "中文",
    "中英文混合",
    "长文本"
};

const int NUM_TESTS = sizeof(TEST_TEXTS) / sizeof(TEST_TEXTS[0]);
const int NUM_ITERATIONS = 1000;

int main(int argc, char** argv) {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     llama_tokenize Overall Performance Test              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_vocab_path>\n";
        return 1;
    }

    std::string vocab_path = argv[1];

    // Load vocab model with vocab_only flag
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;

    llama_model* model = llama_model_load_from_file(vocab_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load vocab: " << vocab_path << "\n";
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create llama_context\n";
        llama_model_free(model);
        return 1;
    }

    llama_token* tokens = new llama_token[4096];
    const int MAX_TOKENS = 4096;

    // Qwen3.5 regex patterns
    std::vector<std::string> qwen35_regex = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
        "\\p{N}|"
        " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    };

    std::cout << "测试配置:\n";
    std::cout << "  Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "  测试用例：" << NUM_TESTS << " 个\n";
    std::cout << "  使用库：absl::flat_hash_map\n\n";

    std::vector<double> total_times(NUM_TESTS, 0);
    std::vector<double> split_times(NUM_TESTS, 0);
    std::vector<double> bpe_times(NUM_TESTS, 0);
    std::vector<int> token_counts(NUM_TESTS, 0);

    // Test each text
    for (int i = 0; i < NUM_TESTS; i++) {
        std::string text = TEST_TEXTS[i];

        // Measure unicode_regex_split time
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            auto words = unicode_regex_split(text, qwen35_regex);
        }
        auto end = std::chrono::high_resolution_clock::now();
        split_times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 1000.0;

        // Measure total tokenize time
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens, MAX_TOKENS, false, false);
            if (iter == 0) token_counts[i] = n_tokens;
        }
        end = std::chrono::high_resolution_clock::now();
        total_times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 1000.0;

        // BPE merge time is the difference
        bpe_times[i] = total_times[i] - split_times[i];
    }

    // Print results
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Results (absl::flat_hash_map)                                                    ║\n";
    std::cout << "╠════════════════════╦═════════════╦═════════════╦═════════════╦═════════════╦═══════════╣\n";
    std::cout << "║ Test               │ Total (ms)  │ Split (ms)  │ BPE (ms)    │ BPE Ratio   │ Tokens    ║\n";
    std::cout << "╠════════════════════╬═════════════╬═════════════╬═════════════╬═════════════╬═══════════╣\n";

    double total_avg = 0;
    double split_avg = 0;
    double bpe_avg = 0;
    int total_tokens = 0;

    for (int i = 0; i < NUM_TESTS; i++) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(16) << TEST_NAMES[i] << " │ "
                  << std::setw(10) << total_times[i] << " │ "
                  << std::setw(10) << split_times[i] << " │ "
                  << std::setw(10) << bpe_times[i] << " │ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << (bpe_times[i] / total_times[i] * 100) << "% │ "
                  << std::setw(8) << token_counts[i] << " ║\n";
        total_avg += total_times[i];
        split_avg += split_times[i];
        bpe_avg += bpe_times[i];
        total_tokens += token_counts[i];
    }
    std::cout << "╠════════════════════╬═════════════╬═════════════╬═════════════╬═════════════╬═══════════╣\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(16) << "Average" << " │ "
              << std::setw(10) << (total_avg / NUM_TESTS) << " │ "
              << std::setw(10) << (split_avg / NUM_TESTS) << " │ "
              << std::setw(10) << (bpe_avg / NUM_TESTS) << " │ "
              << std::setw(10) << std::fixed << std::setprecision(1) << ((bpe_avg / total_avg) * 100) << "% │ "
              << std::setw(8) << (total_tokens / NUM_TESTS) << " ║\n";
    std::cout << "╚════════════════════╩═════════════╩═════════════╩═════════════╩═════════════╩═══════════╝\n";

    // Print summary
    std::cout << "\nSummary:\n";
    std::cout << "  Average total time: " << std::fixed << std::setprecision(3) << (total_avg / NUM_TESTS) << " ms\n";
    std::cout << "  Average BPE time: " << std::fixed << std::setprecision(3) << (bpe_avg / NUM_TESTS) << " ms\n";
    std::cout << "  BPE ratio: " << std::fixed << std::setprecision(1) << ((bpe_avg / total_avg) * 100) << "%\n";
    std::cout << "  Average tokens per text: " << (total_tokens / NUM_TESTS) << "\n";

    delete[] tokens;
    llama_free(ctx);
    llama_model_free(model);

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Test Complete                                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
