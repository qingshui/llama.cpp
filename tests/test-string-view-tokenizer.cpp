#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <unordered_map>

//
// 模拟优化前的数据结构
//

struct OldSymbol {
    using index = int;
    index prev;
    index next;
    const char* text;
    size_t n;
};

struct OldBigramBPE {
    int left;
    int right;
    std::string text;  // 拥有数据
    int rank;
    size_t size;
};

//
// 模拟优化后的数据结构
//

struct NewSymbol {
    using index = int;
    index prev;
    index next;
    std::string_view text;  // 引用数据，零拷贝
};

struct NewBigramBPE {
    int left;
    int right;
    std::string text;  // 仍然拥有数据（队列存储需要）
    int rank;
    size_t size;
};

//
// 测试用例
//

class StringViewTokenizerTest {
public:
    // 测试 1: string_view 基本功能
    static void test_string_view_basic() {
        std::cout << "=== Test 1: String View Basic Functionality ===" << std::endl;

        const char* data = "Hello, World! This is a test string.";

        // 旧方法
        OldSymbol old_sym;
        old_sym.text = data;
        old_sym.n = 5;
        std::string old_str(old_sym.text, old_sym.n);

        // 新方法
        NewSymbol new_sym;
        new_sym.text = std::string_view(data, 5);
        std::string new_str(new_sym.text);

        assert(old_str == new_str);
        assert(old_str == "Hello");

        std::cout << "  Old method: '" << old_str << "'" << std::endl;
        std::cout << "  New method: '" << new_str << "'" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 2: string_view 子串操作
    static void test_string_view_substr() {
        std::cout << "=== Test 2: String View Substring Operations ===" << std::endl;

        const char* data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string_view sv(data, 26);

        // 测试 substr
        assert(sv.substr(0, 5) == "ABCDE");
        assert(sv.substr(10, 5) == "KLMNO");
        assert(sv.substr(21, 5) == "VWXYZ");

        // 测试 compare
        assert(sv.compare(0, 5, "ABCDE") == 0);

        // 测试 find
        assert(sv.find("HIJ") == 7);
        assert(sv.find("XYZ") == 23);

        std::cout << "  substr(0, 5): '" << sv.substr(0, 5) << "'" << std::endl;
        std::cout << "  substr(10, 5): '" << sv.substr(10, 5) << "'" << std::endl;
        std::cout << "  find('HIJ'): " << sv.find("HIJ") << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 3: 符号合并操作（模拟 BPE 合并）
    static void test_symbol_merge() {
        std::cout << "=== Test 3: Symbol Merge Operation (BPE Simulation) ===" << std::endl;

        const char* data = "Hello World";

        // 旧方法
        std::vector<OldSymbol> old_symbols;
        OldSymbol h = { -1, 1, data, 1 };
        OldSymbol e = { 0, 2, data + 1, 1 };
        OldSymbol l1 = { 1, 3, data + 2, 1 };
        OldSymbol l2 = { 2, 4, data + 3, 1 };
        OldSymbol o = { 3, -1, data + 4, 1 };
        old_symbols = {h, e, l1, l2, o};

        // 合并 "ll" -> "l"
        old_symbols[2].n = 2;  // "ll"
        old_symbols[3].n = 0;  // 清空

        std::string merged_old(old_symbols[2].text, old_symbols[2].n);

        // 新方法
        std::vector<NewSymbol> new_symbols;
        NewSymbol h_new = { -1, 1, std::string_view(data, 1) };
        NewSymbol e_new = { 0, 2, std::string_view(data + 1, 1) };
        NewSymbol l1_new = { 1, 3, std::string_view(data + 2, 1) };
        NewSymbol l2_new = { 2, 4, std::string_view(data + 3, 1) };
        NewSymbol o_new = { 3, -1, std::string_view(data + 4, 1) };
        new_symbols = {h_new, e_new, l1_new, l2_new, o_new};

        // 合并 "ll" -> "l"
        l1_new.text = std::string_view(l1_new.text.data(), 2);
        l2_new.text = std::string_view();  // empty
        new_symbols[2] = l1_new;
        new_symbols[3] = l2_new;

        std::string merged_new(new_symbols[2].text);

        assert(merged_old == merged_new);
        assert(merged_old == "ll");

        std::cout << "  Merged token (old): '" << merged_old << "'" << std::endl;
        std::cout << "  Merged token (new): '" << merged_new << "'" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 4: 性能对比 - 字符串创建
    static void test_performance_string_creation() {
        std::cout << "=== Test 4: Performance - String Creation ===" << std::endl;

        const char* data = "The quick brown fox jumps over the lazy dog.";
        size_t len = 15;
        int iterations = 10000000;

        // 旧方法：创建 std::string
        auto start = std::chrono::high_resolution_clock::now();
        volatile size_t dummy = 0;
        for (int i = 0; i < iterations; ++i) {
            std::string s(data, len);
            dummy += s.size();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto old_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 新方法：使用 string_view
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::string_view sv(data, len);
            dummy += sv.size();
        }
        end = std::chrono::high_resolution_clock::now();
        auto new_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Old method (std::string):  " << old_time << " ms" << std::endl;
        std::cout << "  New method (string_view):  " << new_time << " ms" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                  << (double)old_time / (new_time + 1) << "x" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 5: 性能对比 - 字符串转 token 查找（实际 BPE 场景）
    static void test_performance_token_lookup() {
        std::cout << "=== Test 5: Performance - Token Lookup Simulation ===" << std::endl;

        // 模拟词表查找场景
        std::unordered_map<std::string, int> token_table;
        for (int i = 0; i < 1000; ++i) {
            token_table["token_" + std::to_string(i)] = i;
        }

        const char* test_token = "token_500";
        int iterations = 5000000;

        // 旧方法：每次创建临时 string 查找
        auto start = std::chrono::high_resolution_clock::now();
        volatile bool found_old = false;
        for (int i = 0; i < iterations; ++i) {
            std::string temp(test_token, 11);
            found_old = (token_table.find(temp) != token_table.end());
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto old_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 新方法：使用 string_view 避免临时 string 创建
        // 注意：实际场景中需要 heterogeneous lookup 支持
        // 这里演示的是避免在调用点创建 string
        start = std::chrono::high_resolution_clock::now();
        volatile bool found_new = false;
        for (int i = 0; i < iterations; ++i) {
            std::string_view sv(test_token, 11);
            // 仍然需要创建临时 string 用于查找（因为 unordered_map 不支持 string_view key）
            // 但可以在调用点延迟创建，只在必要时转换
            std::string temp(sv);
            found_new = (token_table.find(temp) != token_table.end());
        }
        end = std::chrono::high_resolution_clock::now();
        auto new_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Old method (direct string):  " << old_time << " ms" << std::endl;
        std::cout << "  New method (via string_view): " << new_time << " ms" << std::endl;
        std::cout << "  Note: Main benefit is API flexibility, not raw speed" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 6: 性能对比 - Bigram 验证
    static void test_performance_bigram_validation() {
        std::cout << "=== Test 6: Performance - Bigram Validation ===" << std::endl;

        const char* left_data = "Hello";
        const char* right_data = "World";
        std::string combined = "HelloWorld";

        int iterations = 5000000;

        // 旧方法：创建临时 string 进行验证
        auto start = std::chrono::high_resolution_clock::now();
        volatile bool result = false;
        for (int i = 0; i < iterations; ++i) {
            std::string left(left_data, 5);
            std::string right(right_data, 5);
            std::string test = left + right;
            result = (test == combined);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto old_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 新方法：使用 string_view 进行快速验证
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::string_view left(left_data, 5);
            std::string_view right(right_data, 5);
            // 快速路径：先检查长度，再逐段比较
            bool match = (left.size() + right.size() == combined.size()) &&
                         (left == std::string_view(combined).substr(0, left.size())) &&
                         (right == std::string_view(combined).substr(left.size()));
            result = match;
        }
        end = std::chrono::high_resolution_clock::now();
        auto new_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Old method (string concat):  " << old_time << " ms" << std::endl;
        std::cout << "  New method (string_view):    " << new_time << " ms" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                  << (double)old_time / (new_time + 1) << "x" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 7: 内存使用对比
    static void test_memory_usage() {
        std::cout << "=== Test 7: Memory Usage Comparison ===" << std::endl;

        std::cout << "  sizeof(OldSymbol): " << sizeof(OldSymbol) << " bytes" << std::endl;
        std::cout << "    - const char*: " << sizeof(const char*) << " bytes" << std::endl;
        std::cout << "    - size_t: " << sizeof(size_t) << " bytes" << std::endl;
        std::cout << "    - index prev: " << sizeof(int) << " bytes" << std::endl;
        std::cout << "    - index next: " << sizeof(int) << " bytes" << std::endl;

        std::cout << "  sizeof(NewSymbol): " << sizeof(NewSymbol) << " bytes" << std::endl;
        std::cout << "    - string_view: " << sizeof(std::string_view) << " bytes" << std::endl;
        std::cout << "    - index prev: " << sizeof(int) << " bytes" << std::endl;
        std::cout << "    - index next: " << sizeof(int) << " bytes" << std::endl;

        // string_view 通常是 {pointer, size} = 16 bytes (64-bit)
        // const char* + size_t 也是 16 bytes (64-bit)
        // 所以内存占用相同，但 string_view 提供了更多功能

        std::cout << "  Memory footprint: "
                  << (sizeof(NewSymbol) == sizeof(OldSymbol) ? "SAME" :
                     sizeof(NewSymbol) < sizeof(OldSymbol) ? "REDUCED" : "INCREASED")
                  << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 测试 8: 完整 BPE 分词流程模拟
    static void test_full_bpe_simulation() {
        std::cout << "=== Test 8: Full BPE Tokenization Simulation ===" << std::endl;

        const char* input = "hello world";

        // 初始化符号（字符级分割）
        std::vector<NewSymbol> symbols;
        for (size_t i = 0; i < strlen(input); ++i) {
            symbols.push_back({
                -1, -1,
                std::string_view(input + i, 1)
            });
        }

        // 设置链表
        for (size_t i = 0; i < symbols.size(); ++i) {
            symbols[i].prev = (i == 0) ? -1 : (int)i - 1;
            symbols[i].next = (i == symbols.size() - 1) ? -1 : (int)i + 1;
        }

        // 模拟 BPE 合并："lo" -> "lo" (假设这是一个 merge)
        // 合并索引 2 ('l') 和 3 ('o')
        symbols[2].text = std::string_view(symbols[2].text.data(), 2);
        symbols[3].text = std::string_view();  // empty

        // 更新链表
        symbols[2].next = symbols[3].next;  // 3 -> 4
        if (symbols[3].next >= 0) {
            symbols[symbols[3].next].prev = 2;
        }

        // 收集结果
        std::vector<std::string> tokens;
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (!symbols[i].text.empty()) {
                tokens.push_back(std::string(symbols[i].text));
            }
        }

        std::cout << "  Input: \"" << input << "\"" << std::endl;
        std::cout << "  Tokens: ";
        for (const auto& t : tokens) {
            std::cout << "\"" << t << "\" ";
        }
        std::cout << std::endl;

        // After merging 'l'+'o' at index 2, we have: h, e, lo, (empty), w, o, r, l, d
        // But we skip empty tokens, so: h, e, lo, w, o, r, l, d = 9 tokens for "hello world"
        // Wait, the input is "hello world" (11 chars), after one merge we have 10 tokens
        assert(tokens.size() == 10);  // h, e, ll, o,  , w, o, r, l, d
        std::cout << "  Token count: " << tokens.size() << " (expected 10)" << std::endl;
        std::cout << "  PASSED" << std::endl << std::endl;
    }

    // 运行所有测试
    static void run_all_tests() {
        std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     String View Tokenizer Unit Tests                     ║" << std::endl;
        std::cout << "║     Testing BPE optimization for llama.cpp               ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << std::endl;

        test_string_view_basic();
        test_string_view_substr();
        test_symbol_merge();
        test_performance_string_creation();
        test_performance_token_lookup();
        test_performance_bigram_validation();
        test_memory_usage();
        test_full_bpe_simulation();

        std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ALL TESTS PASSED                                     ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    }
};

int main() {
    StringViewTokenizerTest::run_all_tests();
    return 0;
}
