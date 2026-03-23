# Qwen3.5 BPE 分词优化路线图

## 概述

本文档总结了 llama.cpp 中 Qwen3.5 BPE 分词器的所有优化机会，包括已完成和待实现的优化。

---

## Qwen3.5 分词特点

### 预分词器配置

```cpp
case LLAMA_VOCAB_PRE_TYPE_QWEN35:
    regex_exprs = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"  // 关键：支持组合标记 \p{M}
        "\\p{N}|"
        " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    break;
```

### 关键特性

| 特性 | 说明 |
|------|------|
| 组合标记支持 | `\p{M}` (Combining Marks) - 重音/变音符号 |
| 词表大小 | ~151,936 tokens |
| BPE merges | ~151,387 |
| 最大 token 长度 | 256 bytes |

---

## 优化清单

### ✅ 已完成优化

#### 1. String View 优化 (2026-03-23)

**修改内容**:
- `llm_symbol` 结构：`const char* + size_t` → `std::string_view`
- `llm_bigram_bpe` 验证：避免临时 string 创建
- `find_bpe_rank` 参数：`const std::string&` → `std::string_view`
- `text_to_token` 新增 `string_view` 重载

**性能提升**: **2.80x** BPE 合并操作加速

**测试文件**:
- `tests/test-string-view-tokenizer.cpp` (单元测试)
- `tests/test-bpe-tokenize.cpp` (性能基准)
- `tests/FINAL_BENCHMARK_REPORT.md` (测试报告)

**基准测试结果**:

| 测试 | 文本类型 | 旧方法 (ms) | 优化后 (ms) | 加速比 |
|------|----------|------------|------------|--------|
| Test 1 | 短英文 | 8.17 | 2.74 | 2.98x |
| Test 2 | 中等英文 | 28.10 | 9.75 | 2.88x |
| Test 3 | 中文 | 38.47 | 13.68 | 2.81x |
| Test 4 | 中英文混合 | 35.63 | 12.67 | 2.81x |
| Test 5 | 长文本 | 16.05 | 6.32 | 2.54x |

---

### 🔄 待实现优化

#### 2. BPE Rank 异质查找 (Heterogeneous Lookup)

**优先级**: 🔴 高

**问题描述**:
当前 `find_bpe_rank` 即使使用 `string_view` 参数，仍需创建临时 `std::string` 进行查找：

```cpp
// 当前实现
auto it = pimpl->bpe_ranks.find(std::make_pair(std::string(token_left), std::string(token_right)));
```

**优化方案**:
```cpp
// 使用 C++14 透明比较器
struct pair_hash_transparent {
    using is_transparent = void;

    size_t operator()(const std::pair<std::string, std::string>& p) const;
    size_t operator()(const std::pair<std::string_view, std::string_view>& p) const;
};

struct pair_equal_transparent {
    using is_transparent = void;

    bool operator()(const std::pair<std::string, std::string>& a,
                    const std::pair<std::string_view, std::string_view>& b) const;
};

std::unordered_map<std::pair<std::string, std::string>, int,
                   pair_hash_transparent, pair_equal_transparent> bpe_ranks;
```

**预期收益**: 15-25% 额外加速

**实现复杂度**: 中

**修改文件**:
- `src/llama-vocab.cpp` (bpe_ranks 定义和 find_bpe_rank 实现)
- `src/llama-vocab.h` (声明)

---

#### 3. Unicode 折叠查找表 (LUT)

**优先级**: 🟡 中

**问题描述**:
`unicode_regex_split` 中对每个 codepoint 调用 `unicode_cpt_flags_from_cpt`，包含多次范围检查：

```cpp
// 当前实现 (unicode.cpp:974)
const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);
```

**优化方案**:
```cpp
// 预计算 BMP 字符的 Unicode 类别 (16KB LUT)
static const uint8_t unicode_category_lut[65536];

// 快速路径：直接查表
inline uint8_t get_unicode_category_fast(uint32_t cpt) {
    if (cpt < 65536) return unicode_category_lut[cpt];
    return unicode_cpt_flags_from_cpt(cpt).category_flag();
}
```

**预期收益**: 10-15% 预分词加速

**实现复杂度**: 低

**修改文件**:
- `src/unicode.cpp` (添加 LUT 和快速路径)
- `src/unicode-data.cpp` (生成 LUT 数据)

---

#### 4. 中文字符快速路径

**优先级**: 🟡 中

**问题描述**:
Qwen3.5 词表中常用汉字约 3000-5000 个，占实际使用频率的 95%+，但每次查找仍需哈希查找。

**优化方案**:
```cpp
// 在 llama_vocab::impl 中添加
struct {
    std::array<llama_token, 20941> han_lut;  // CJK Unified Ideographs (0x4E00-0x9FFF)
    bool initialized = false;
} han_cache;

// 快速查找
inline llama_token find_chinese_char_fast(uint32_t cpt) {
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) {
        size_t idx = cpt - 0x4E00;
        if (idx < han_cache.han_lut.size()) {
            return han_cache.han_lut[idx];
        }
    }
    return LLAMA_TOKEN_NULL;
}
```

**预期收益**: 中文文本 20-30% 加速

**实现复杂度**: 中

**修改文件**:
- `src/llama-vocab.cpp` (添加缓存和快速路径)
- `src/llama-vocab.cpp` (load 函数中初始化缓存)

---

#### 5. Bigram 优先队列优化

**优先级**: 🟢 低

**问题描述**:
`std::priority_queue` 的 `push/pop` 操作为 O(log n)，长文本可能产生数万个 bigram。

**优化方案**:
- 使用 **配对堆 (Pairing Heap)** 或 **Fibonacci Heap**
- 或针对 BPE 场景的定制堆（利用 rank 单调性）

```cpp
// 配对堆实现（简化版）
template<typename T, typename Compare>
class pairing_heap {
    struct node {
        T data;
        std::vector<node*> children;
        node *prev = nullptr, *next = nullptr;
    };
    // ...
};
```

**预期收益**: 5-10% 加速（长文本更明显）

**实现复杂度**: 高

**修改文件**:
- `src/llama-vocab.cpp` (替换 priority_queue)

---

#### 6. Token 查找缓存 (LRU)

**优先级**: 🟢 低

**问题描述**:
BPE 分词过程中，相同子串可能被多次查找。

**优化方案**:
```cpp
// LRU 缓存最近查找的 token
struct TokenCache {
    struct Entry {
        std::string key;
        llama_token token;
    };
    std::vector<Entry> entries;
    size_t capacity = 256;

    llama_token find(std::string_view sv);
    void insert(std::string_view sv, llama_token token);
};
```

**预期收益**: 5-10% 加速（取决于文本重复度）

**实现复杂度**: 低

**修改文件**:
- `src/llama-vocab.cpp` (添加缓存层)

---

#### 7. Qwen3.5 组合标记预计算

**优先级**: 🟢 低

**问题描述**:
Qwen3.5 支持 `\p{M}` 组合标记，常见组合（如 `é = e + ́`）可预计算。

**优化方案**:
```cpp
// 预计算常见组合标记序列
std::unordered_map<std::string, llama_token> combining_mark_cache;

// 初始化时预加载
void init_combining_mark_cache() {
    // 拉丁字母 + 组合标记
    combining_mark_cache["e\u0301"] = text_to_token("é");
    combining_mark_cache["a\u0301"] = text_to_token("á");
    // ...
}
```

**预期收益**: 多语言文本 5-10% 加速

**实现复杂度**: 低

**修改文件**:
- `src/llama-vocab.cpp` (添加缓存)

---

## 优化优先级矩阵

```
                高收益
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    │  BPE 异质    │  中文字符    │
    │  查找        │  快速路径    │
    │  (15-25%)   │  (20-30%)   │
    │             │             │
低 ─┼─────────────┼─────────────┼─ 高
    │  复杂度     │             │  复杂度
    │             │             │
    │  Unicode    │  Bigram     │
    │  LUT        │  堆优化      │
    │  (10-15%)   │  (5-10%)    │
    │             │             │
    └─────────────┼─────────────┘
                  │
                低收益
```

---

## 实施计划

### Phase 1 (立即实施)
- [x] ✅ String View 优化
- [ ] BPE Rank 异质查找

### Phase 2 (短期)
- [ ] Unicode 折叠 LUT
- [ ] 中文字符快速路径

### Phase 3 (长期)
- [ ] Bigram 堆优化
- [ ] Token 查找缓存
- [ ] 组合标记预计算

---

## 预期总收益

| 阶段 | 优化项 | 累计加速 |
|------|--------|----------|
| Phase 1 | String View | 2.80x |
| Phase 2 | + 异质查找 + LUT + 中文快速路径 | 4.0-5.0x |
| Phase 3 | + 堆优化 + 缓存 | 4.5-5.5x |

**最终目标**: 相比原始实现，**4-5x** BPE 分词性能提升

---

## 测试方法

### 单元测试
```bash
./tests/test-string-view-tokenizer
```

### 性能基准
```bash
./tests/test-bpe-tokenize
```

### 端到端测试
```bash
# 需要 Qwen3.5 GGUF 模型
./build/bin/llama-bench -m models/Qwen3.5-*.gguf -b 10
```

---

## 参考文档

- `tests/STRING_VIEW_OPTIMIZATION_REPORT.md` - String View 优化报告
- `tests/FINAL_BENCHMARK_REPORT.md` - 最终基准报告
- `src/llama-vocab.cpp` - BPE 分词实现
- `src/unicode.cpp` - Unicode 工具函数

---

*文档创建日期：2026-03-23*
*最后更新：2026-03-23*
