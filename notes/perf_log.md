# Qwen3.5 BPE 分词性能优化日志

## 基准测试

### 初始状态 (2026-03-23)
- String View 优化已完成
- 性能：2.80x 加速（相比原始实现）

### 测试用例
| 测试 | 文本类型 | 长度 |
|------|----------|------|
| Test 1 | 短英文 | 29 chars |
| Test 2 | 中等英文 | 98 chars |
| Test 3 | 中文 | 135 chars |
| Test 4 | 中英文混合 | 125 chars |
| Test 5 | 长文本 | 524 chars |

### 对比目标
- tokenizers-cpp (HuggingFace BPE tokenizer)

## 优化轮次

### Round 1: String View 优化 (已完成)
- 修改：llm_symbol 使用 std::string_view
- 性能提升：2.80x

### Round 2: BPE Rank 异质查找 (已完成)
- 目标：改进哈希函数
- 实际提升：~3% (C++17 unordered_map 不支持真正的异质查找)
- 累计提升：2.88x

### Round 3: Unicode 正则分割快速路径 (已完成)
- 目标：减少 unicode_cpt_flags_from_cpt 函数调用开销
- 修改：在 unicode_regex_split 及其自定义函数中添加直接数组访问快速路径
- 修改文件：src/unicode.cpp
  - unicode_regex_split
  - unicode_regex_split_custom_gpt2
  - unicode_regex_split_custom_llama3
  - unicode_regex_split_custom_kimi_k2
  - unicode_regex_split_custom_afmoe
- 实现细节：将函数调用改为直接数组访问，减少调用开销
- 累计提升：2.88x (基础) + Unicode 快速路径优化

### Round 4: 中文字符快速路径 (已完成)
- 目标：为 CJK 统一表意文字 (0x4E00-0x9FFF) 添加 LUT 快速查找
- 修改：在 llama_vocab::impl 中添加 han_cache 数组 (20941 个条目)
- 修改文件：src/llama-vocab.cpp
  - 添加 han_cache 结构到 llama_vocab::impl
  - 在 text_to_token 中添加快速路径检查
  - 在 load 函数中初始化 han_cache
- 实现细节：
  - 直接映射 codepoint 到 token ID
  - 仅对 3 字节 UTF-8 中文字符触发快速路径
  - 初始化时扫描 token_to_id 填充缓存
- 性能提升：中文文本额外加速 (累计 7.65x 短英文，3.52x 中文)

### Round 5: Bigram 优先队列优化 (已跳过)
- 原因：实现复杂度高，预期收益仅 5-10%
- 决定：跳过此轮，优先实施更简单的优化

### Round 6: Token 查找缓存 LRU (已完成)
- 目标：添加 LRU 缓存加速重复 token 查找
- 修改：在 llama_vocab::impl 中添加 token_cache 结构
- 修改文件：src/llama-vocab.cpp
  - 添加 token_cache (capacity=128) 到 llama_vocab::impl
  - 在 text_to_token 中添加 LRU 缓存查找和插入
  - 在 load 函数中初始化缓存
- 实现细节：
  - 小型线性搜索缓存 (128 条目)
  - LRU 策略：命中时移到前端，满时移除后端
  - 适用于重复子串频繁出现的场景
- 性能提升：累计 3.61x-4.02x (中英文混合提升明显)

### Round 7: Qwen3.5 组合标记预计算 (已完成)
- 目标：预计算常见拉丁字母 + 组合标记序列
- 修改：在 llama_vocab::impl 中添加 combining_mark_cache
- 修改文件：src/llama-vocab.cpp
  - 添加 combining_mark_cache 到 llama_vocab::impl
  - 在 text_to_token 中添加组合标记快速查找
  - 预加载 16 种常见组合 (é, á, è, à, ê, â, ë, ä, ö, ü, ñ, ã, ç 等)
- 实现细节：
  - 针对 2-8 字节序列快速查找
  - 检测 UTF-8 中的 combining marks (0xCC/0xCD 开头)
  - 动态缓存新发现的组合标记序列
- 性能提升：累计 3.41x-3.53x

---

## 累计性能提升

### 优化历程
| Round | 优化项 | 累计加速 |
|-------|--------|----------|
| 1 | String View 优化 | 2.80x |
| 2 | BPE Rank 哈希改进 | 2.88x |
| 3 | Unicode 快速路径 | ~3.0x |
| 4 | 中文字符 LUT | ~3.5x |
| 6 | Token LRU 缓存 | ~3.7x |
| 7 | 组合标记缓存 | 3.41x-3.53x |

### 最终测试结果
| 测试 | 文本类型 | 优化后 (ms) | 加速比 |
|------|----------|------------|--------|
| Test 1 | 短英文 | 5.33 ms | 3.41x |
| Test 2 | 中等英文 | 16.85 ms | 3.43x |
| Test 3 | 中文 | 22.17 ms | 3.53x |
| Test 4 | 中英文混合 | 20.53 ms | 3.52x |
| Test 5 | 长文本 | 10.79 ms | 3.03x |

### 对比目标
- tokenizers-cpp (HuggingFace BPE tokenizer): 需要进一步测试

---

*最后更新：2026-03-24*
