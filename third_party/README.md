# Third Party Dependencies

本目录包含 llama.cpp 编译所需的第三方依赖。

## 快速开始 (Quick Start)

只需一行命令，下载所有依赖并开始编译：

```bash
# 一键下载依赖并编译
bash third_party/download-deps.sh && mkdir build && cd build && cmake .. && make -j$(nproc)
```

**就这么简单！** 所有其他依赖都已包含在 `vendor` 目录中。

## 依赖列表

### 1. abseil-cpp (必需)

用于 BPE  tokenizer 优化的 flat_hash_map 实现。

**下载方式**:

```bash
# 方式 1: 使用 git clone (推荐)
cd third_party
git clone --depth 1 --branch 20240722.0 https://github.com/abseil/abseil-cpp.git abseil

# 方式 2: 下载 tarball
cd third_party
curl -L https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.tar.gz | tar xz
mv abseil-cpp-20240722.0 abseil
```

**版本要求**: 20240722.0 或更高版本

**用途**:
- `absl::flat_hash_map` 用于加速 BPE merge 查找
- 位于 `src/llama-vocab.cpp` 中的 `find_bpe_rank` 函数使用

## 可选依赖 (vendor 目录)

以下依赖已包含在 `vendor` 目录中，无需额外下载：

- **nlohmann/json** - JSON 解析库 (single-header)
- **cpp-httplib** - HTTP 服务器库
- **miniaudio** - 音频处理库
- **stb** - 图像处理库
- **shereman** - 单元测试库

## 系统依赖 (通过包管理器安装)

编译时会自动检测并使用以下系统库（如已安装）：

- **OpenSSL** - HTTPS 支持 (可选)
- **BLAS** - 线性代数加速 (可选)
- **OpenMP** - 并行计算 (可选)
- **CUDA** - GPU 加速 (可选)
- **Vulkan** - GPU 加速 (可选)

## 快速开始

### Ubuntu/Debian

```bash
# 安装基本编译依赖
sudo apt-get install -y build-essential cmake

# 安装 abseil-cpp (或使用上面的 git clone 方式)
sudo apt-get install -y libabsl-dev

# 编译 llama.cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### CentOS/RHEL

```bash
# 安装基本编译依赖
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake

# 下载 abseil-cpp
cd third_party
git clone --depth 1 --branch 20240722.0 https://github.com/abseil/abseil-cpp.git abseil

# 编译 llama.cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# 使用 Homebrew 安装依赖
brew install cmake

# 下载 abseil-cpp
cd third_party
git clone --depth 1 --branch 20240722.0 https://github.com/abseil/abseil-cpp.git abseil

# 编译 llama.cpp
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

## 一键下载脚本

运行以下脚本自动下载所有依赖：

```bash
bash download-deps.sh
```

## 目录结构

```
third_party/
├── abseil/              # abseil-cpp 库 (必需)
└── README.md            # 本文件

vendor/
├── nlohmann/            # JSON 库 (已包含)
├── cpp-httplib/         # HTTP 库 (已包含)
├── miniaudio/           # 音频库 (已包含)
├── stb/                 # 图像库 (已包含)
└── sheredom/            # 测试库 (已包含)
```
