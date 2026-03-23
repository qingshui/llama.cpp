#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "llama.h"

//
// 测试文本
//

const char* TEST_TEXTS[] = {
    // 短文本
    "Hello, world! This is a simple test.",

    // 中等文本
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow.",

    // 长文本（英文）
    "In the heart of the city, where skyscrapers touch the clouds and neon lights "
    "illuminate the streets, a lone programmer sits at their desk, coding late into "
    "the night. The screen glows with lines of code, each character a building block "
    "of digital creation. Machine learning models train in the background, their "
    "neural networks processing vast amounts of data to find patterns hidden within. "
    "Artificial intelligence, once a distant dream, now powers our daily lives from "
    "smartphones to self-driving cars. The future is here, and it is written in code.",

    // 中文文本
    "人工智能是当今科技领域最热门的话题之一。从智能手机到自动驾驶汽车，从语音识别到图像识别，"
    "人工智能技术正在改变着我们的生活方式。深度学习、机器学习、神经网络这些概念已经不再陌生。"
    "在中国，人工智能产业发展迅速，许多科技公司都在这个领域投入了大量资源。未来，人工智能将"
    "在医疗、教育、金融等更多领域发挥重要作用。",

    // 混合文本
    "The development of AI technology 人工智能的发展 has brought significant changes "
    "to our society 对我们的社会带来了重大变化。Machine learning algorithms 机器学习算法 "
    "can now process vast amounts of data 现在可以处理大量数据 to find patterns and make predictions."
};

const int NUM_TEXTS = sizeof(TEST_TEXTS) / sizeof(TEST_TEXTS[0]);

//
// 性能测试结果
//

struct TokenizerBenchmark {
    std::string name;
    int text_length;
    int num_tokens;
    double tokenize_time_ms;
    double detokenize_time_ms;
    double tokens_per_second;
};

//
// 基准测试函数
//

void run_benchmark(const llama_vocab* vocab, const char* text, TokenizerBenchmark& result) {
    std::string input_text(text);
    result.text_length = input_text.length();

    std::vector<llama_token> tokens;
    tokens.resize(input_text.length() * 2);  // 预分配空间

    // Tokenize 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    int num_tokens = llama_tokenize(vocab, text, input_text.length(),
                                    tokens.data(), tokens.size(),
                                    false, false);
    auto end = std::chrono::high_resolution_clock::now();

    result.num_tokens = num_tokens;
    result.tokenize_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.tokens_per_second = (num_tokens / result.tokenize_time_ms) * 1000.0;

    // Detokenize 性能测试
    std::string output_text;
    output_text.resize(input_text.length() * 2);

    start = std::chrono::high_resolution_clock::now();
    int output_len = llama_detokenize(vocab, tokens.data(), num_tokens,
                                      &output_text[0], output_text.length(),
                                      false, false);
    end = std::chrono::high_resolution_clock::now();

    result.detokenize_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
}

//
// 打印测试结果
//

void print_results(const std::vector<TokenizerBenchmark>& results) {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Tokenizer Performance Benchmark Results                 ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════╣\n";

    double total_tokenize_time = 0;
    double total_tokens = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "║ Test " << (i + 1) << ": " << r.name << std::string(50 - r.name.length(), ' ') << "║\n";
        std::cout << "║   Text length: " << r.text_length << " chars, Tokens: " << r.num_tokens
                  << std::string(30, ' ') << "║\n";
        std::cout << "║   Tokenize:  " << r.tokenize_time_ms << " ms (" << r.tokens_per_second
                  << " tokens/sec)" << std::string(20, ' ') << "║\n";
        std::cout << "║   Detokenize: " << r.detokenize_time_ms << " ms"
                  << std::string(35, ' ') << "║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════╣\n";

        total_tokenize_time += r.tokenize_time_ms;
        total_tokens += r.num_tokens;
    }

    double avg_tokens_per_sec = (total_tokens / total_tokenize_time) * 1000.0;

    std::cout << "║ Summary:                                                             ║\n";
    std::cout << "║   Total tokens processed: " << static_cast<int>(total_tokens)
              << std::string(24, ' ') << "║\n";
    std::cout << "║   Average throughput: " << avg_tokens_per_sec << " tokens/sec"
              << std::string(26, ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
}

//
// 主函数
//

int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              llama.cpp String View Tokenizer Performance Test              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";

    // 检查命令行参数
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] << " <vocab_file.gguf>\n";
        std::cout << "\nNote: This test requires a GGUF model file with BPE vocabulary.\n";
        std::cout << "You can use the vocabulary-only GGUF files from llama.cpp/models/\n";
        return 1;
    }

    const char* vocab_file = argv[1];

    // 加载词表
    std::cout << "\nLoading vocabulary from: " << vocab_file << " ...\n";

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(vocab_file, model_params);
    if (model == nullptr) {
        std::cerr << "Failed to load model/vocabulary from: " << vocab_file << "\n";
        return 1;
    }

    std::cout << "Model loaded successfully!\n";

    // 获取词表信息
    const llama_vocab* vocab = llama_model_get_vocab(model);
    uint32_t n_tokens = llama_vocab_n_tokens(vocab);

    std::cout << "Vocabulary size: " << n_tokens << "\n";
    std::cout << "Vocabulary type: BPE (assumed)\n\n";

    // 运行基准测试
    std::vector<TokenizerBenchmark> results(NUM_TEXTS);

    const char* test_names[] = {
        "Short English text",
        "Medium English text (pangrams)",
        "Long English text",
        "Chinese text",
        "Mixed English-Chinese text"
    };

    for (int i = 0; i < NUM_TEXTS; ++i) {
        results[i].name = test_names[i];
        run_benchmark(vocab, TEST_TEXTS[i], results[i]);
        std::cout << "Test " << (i + 1) << " completed: " << results[i].name << "\n";
    }

    // 打印结果
    print_results(results);

    // 清理
    llama_model_free(model);

    return 0;
}
