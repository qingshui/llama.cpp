#!/usr/bin/env python3
"""
Performance benchmark: llama.cpp vs tokenizers-cpp (Qwen3.5-4B)
对比 llama.cpp 优化后的分词性能与 HuggingFace tokenizers-cpp
"""

import subprocess
import time
import re

# 测试文本
TEST_TEXTS = [
    ("Short English", "短英文", "Hello, world! This is a simple test."),
    ("Medium English", "中等英文", "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."),
    ("Chinese", "中文", "人工智能是当今科技领域最热门的话题之一。深度学习、机器学习、神经网络这些概念已经不再陌生。"),
    ("Mixed English-Chinese", "中英文混合", "AI technology 人工智能的发展 has brought significant changes 对我们的社会带来了重大变化。"),
    ("Long text", "长文本", "In the heart of the city, where skyscrapers touch the clouds and neon lights illuminate the streets, a lone programmer sits at their desk, coding late into the night. The screen glows with lines of code, each character a building block of digital creation. 人工智能技术正在改变着我们的生活方式。")
]

ITERATIONS = 1000

def run_llama_benchmark():
    """运行 llama.cpp 的 BPE 分词基准测试"""
    print("运行 llama.cpp 基准测试...")
    result = subprocess.run(
        ["./tests/test-bpe-tokenize"],
        capture_output=True,
        text=True,
        cwd="/home/disk4/humingqing/work/llama.cpp"
    )

    # 解析输出 - 查找 "Optimized:" 行
    llama_results = {}

    patterns = [
        ("Short English", "短英文"),
        ("Medium English", "中等英文"),
        ("Chinese", "中文"),
        ("Mixed English-Chinese", "中英文混合"),
        ("Long text", "长文本"),
    ]

    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        for en_name, cn_name in patterns:
            if en_name in line or cn_name in line:
                # 查找后面的 Optimized 行
                for j in range(i+1, min(i+4, len(lines))):
                    if "Optimized:" in lines[j]:
                        match = re.search(r'Optimized:\s+([\d.]+)\s*ms', lines[j])
                        if match:
                            llama_results[cn_name] = float(match.group(1))
                        break

    return llama_results

def run_tokenizers_benchmark():
    """运行 HuggingFace tokenizers 基准测试 (使用本地 Qwen3.5-4B)"""
    print("运行 tokenizers-cpp (HuggingFace) 基准测试...")
    print("使用模型：/home/disk4/humingqing/qwen/Qwen/Qwen3.5-4B/tokenizer.json")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('/home/disk4/humingqing/qwen/Qwen/Qwen3.5-4B/tokenizer.json')

    results = {}

    for en_name, cn_name, text in TEST_TEXTS:
        # Warmup
        tokenizer.encode(text)

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            tokenizer.encode(text)
        end = time.perf_counter()

        time_ms = (end - start) * 1000
        results[cn_name] = time_ms
        print(f"  {cn_name}: {time_ms:.2f} ms")

    return results

def main():
    print("=" * 70)
    print("llama.cpp vs tokenizers-cpp (Qwen3.5-4B) 性能对比")
    print("=" * 70)
    print()

    # 运行 llama.cpp 测试
    llama_results = run_llama_benchmark()
    print(f"llama.cpp 结果：{llama_results}")
    print()

    # 运行 tokenizers 测试
    tokenizer_results = run_tokenizers_benchmark()
    print()

    # 输出对比结果
    print("=" * 70)
    print("性能对比结果")
    print("=" * 70)
    print(f"{'测试':<15} {'llama.cpp (ms)':<15} {'tokenizers (ms)':<15} {'对比':<20}")
    print("-" * 70)

    ratios = []
    for _, cn_name, _ in TEST_TEXTS:
        llama_time = llama_results.get(cn_name, 0)
        tok_time = tokenizer_results.get(cn_name, 0)

        if tok_time > 0 and llama_time > 0:
            ratio = tok_time / llama_time
            comparison = f"llama {ratio:.2f}x 快" if ratio > 1 else f"tokenizers {1/ratio:.2f}x 快"
            ratios.append(ratio)
        else:
            comparison = "N/A"

        print(f"{cn_name:<15} {llama_time:<15.2f} {tok_time:<15.2f} {comparison}")

    print("=" * 70)

    # 计算平均加速比
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"平均加速比：{avg_ratio:.2f}x (llama.cpp 相对于 tokenizers)")
        print("=" * 70)

        if avg_ratio >= 1.0:
            print("✓ llama.cpp 性能超过 tokenizers-cpp!")
            print(f"  平均快 {avg_ratio:.2f}x")
            return True
        else:
            gap = (1/avg_ratio - 1) * 100
            print(f"✗ llama.cpp 还需要优化 {gap:.1f}% 才能追平 tokenizers")
            print(f"  当前 tokenizers 比 llama.cpp 快 {1/avg_ratio:.2f}x")
            return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
