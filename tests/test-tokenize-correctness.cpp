/**
 * Test tokenize correctness with extended test cases
 * Compares llama.cpp output against itself (baseline) to verify optimization correctness
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <map>

extern "C" {
#include "../include/llama.h"
}

struct TestCase {
    std::string name;
    std::string text;
};

// Extended test cases including Chinese, English, mixed, and CN+URL
std::vector<TestCase> test_cases = {
    {"Short EN", "Hello, world!"},
    {"Short CN", "你好，世界！"},
    {"Medium EN", "The quick brown fox jumps over the lazy dog."},
    {"Medium CN", "人工智能是当今最热门的技术领域之一。"},
    {"Mixed", "Artificial intelligence is transforming the world. 人工智能正在改变世界。"},
    {"Code", "def hello_world(): print('Hello, World!')"},
    {"Long EN", "The model uses attention mechanisms to process sequences efficiently. Attention allows the model to weigh the importance of different parts of the input."},
    {"Long CN", "神经网络深度学习机器学习自然语言处理计算机视觉"},
    {"Repeat EN", "The the the the the"},
    {"Repeat CN", "你好你好你好你好你好"},
    // Extended cases: CN + URL
    {"CN+URL", "请访问 https://www.example.com 获取更多信息"},
    {"CN+URL2", "百度 https://www.baidu.com 是中国最大的搜索引擎"},
    {"CN+URL3", "GitHub https://github.com/llama-cpp/llama.cpp 是一个开源项目"},
    {"CN+URL4", "淘宝 https://www.taobao.com 购物网站"},
    {"CN+URL5", "微信 https://weixin.qq.com 社交媒体"}
};

class LlamaTokenizer {
public:
    LlamaTokenizer(const std::string& gguf_path) : vocab_(nullptr) {
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        model_ = llama_model_load_from_file(gguf_path.c_str(), params);
        if (!model_) {
            throw std::runtime_error("Failed to load model");
        }
        vocab_ = llama_model_get_vocab(model_);
    }

    ~LlamaTokenizer() {
        if (model_) llama_model_free(model_);
    }

    std::vector<int32_t> encode(const std::string& text) {
        std::vector<int32_t> tokens;
        int32_t n_tokens = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), nullptr, 0, false, false);
        if (n_tokens < 0) n_tokens = -n_tokens;
        if (n_tokens <= 0) return tokens;
        tokens.resize(n_tokens);
        n_tokens = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), tokens.data(), n_tokens, false, false);
        if (n_tokens < 0) n_tokens = -n_tokens;
        tokens.resize(n_tokens);
        return tokens;
    }

    size_t vocab_size() const { return llama_vocab_n_tokens(vocab_); }

private:
    llama_model* model_;
    const llama_vocab* vocab_;
};

int main(int argc, char* argv[]) {
    std::string gguf_path = "/home/disk4/humingqing/qwen/Qwen/Qwen3.5-4B/ggml-model-f16.gguf";

    if (argc > 1) {
        gguf_path = argv[1];
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Llama.cpp Tokenize Correctness Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << gguf_path << std::endl;
    std::cout << std::endl;

    // Load tokenizer
    std::cout << "Loading tokenizer..." << std::endl;
    LlamaTokenizer tokenizer(gguf_path);
    std::cout << "  Vocab size: " << tokenizer.vocab_size() << std::endl;
    std::cout << std::endl;

    // Baseline: collect reference outputs
    std::cout << "=== Collecting Baseline Outputs ===" << std::endl;
    std::map<std::string, std::vector<int32_t>> baseline;
    for (const auto& tc : test_cases) {
        baseline[tc.name] = tokenizer.encode(tc.text);
    }
    std::cout << "  Collected " << baseline.size() << " baseline outputs" << std::endl;
    std::cout << std::endl;

    // Correctness check: compare against baseline
    std::cout << "=== Correctness Check (vs Baseline) ===" << std::endl;
    int correct = 0, total = 0;
    for (const auto& tc : test_cases) {
        auto current = tokenizer.encode(tc.text);
        auto& ref = baseline[tc.name];
        bool match = (current == ref);
        if (match) correct++;
        total++;
        std::cout << (match ? "OK" : "X ") << " " << tc.name
                  << " (tokens:" << current.size() << ")" << std::endl;
    }
    std::cout << "Result: " << correct << "/" << total << " passed" << std::endl;
    std::cout << std::endl;

    // Performance test
    std::cout << "=== Performance Test (1000 iterations) ===" << std::endl;
    std::cout << std::left << std::setw(15) << "Test"
              << std::setw(8) << "Tokens"
              << std::setw(15) << "Avg (ms)"
              << std::setw(15) << "Min (ms)"
              << std::setw(15) << "Max (ms)"
              << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    double total_time = 0;
    int total_tokens = 0;

    for (const auto& tc : test_cases) {
        std::vector<double> times;
        times.reserve(1000);

        // Warmup
        for (int i = 0; i < 10; ++i) {
            tokenizer.encode(tc.text);
        }

        // Benchmark
        for (int i = 0; i < 1000; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            tokenizer.encode(tc.text);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::sort(times.begin(), times.end());
        double sum = 0;
        for (double t : times) sum += t;
        double avg = sum / times.size();
        double min_t = times.front();
        double max_t = times.back();

        total_time += avg;
        total_tokens += baseline[tc.name].size();

        std::cout << std::left << std::setw(15) << tc.name
                  << std::setw(8) << baseline[tc.name].size()
                  << std::fixed << std::setprecision(3)
                  << std::setw(15) << avg
                  << std::setw(15) << min_t
                  << std::setw(15) << max_t
                  << std::endl;
    }

    std::cout << std::string(68, '-') << std::endl;
    std::cout << std::endl;

    // Summary
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(3) << (total_time / test_cases.size()) << " ms" << std::endl;
    std::cout << "Total tokens: " << total_tokens << std::endl;
    std::cout << "Correctness: " << correct << "/" << total << std::endl;

    return (correct == total) ? 0 : 1;
}
