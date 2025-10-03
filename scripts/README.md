# KTransformers 开发环境构建脚本

本目录包含了用于 KTransformers 开发环境配置和编译的脚本集合，特别针对 balance_serve 功能进行了优化。

## 脚本概览

### 1. `build_ktransformers.sh` - 完整编译脚本
**推荐用于首次安装和生产环境**

功能特点：
- 完整的系统环境检查和依赖验证
- 支持 Ubuntu 22.04 LTS + W3-2423 CPU (AMX 指令)
- 包含 balance_serve 多并发功能支持
- 生产级别的错误处理和诊断
- 详细的构建日志和进度反馈
- 自动生成验证脚本

使用方法：
```bash
# 标准编译（推荐）
./scripts/build_ktransformers.sh

# 清理重编译
./scripts/build_ktransformers.sh --clean

# 启用 NUMA 支持（双 CPU 系统）
./scripts/build_ktransformers.sh --numa

# 详细输出模式
./scripts/build_ktransformers.sh --verbose

# 跳过依赖检查（快速重编译）
./scripts/build_ktransformers.sh --skip-deps
```

### 2. `quick_build.sh` - 快速编译脚本
**推荐用于开发迭代**

功能特点：
- 简化的编译流程，适合快速迭代
- 最小化的环境检查
- 快速的增量编译支持
- 适用于开发环境的日常使用

使用方法：
```bash
# 快速编译
./scripts/quick_build.sh

# 清理后快速编译
./scripts/quick_build.sh --clean

# 启用 NUMA 支持
./scripts/quick_build.sh --numa

# 使用 CPU 后端
./scripts/quick_build.sh --cpu
```

### 3. `verify_ktransformers.sh` - 功能验证脚本
**用于验证安装和功能测试**

功能特点：
- 全面的功能验证测试
- balance_serve 后端可用性检查
- 系统资源和兼容性检查
- 详细的测试报告和建议

使用方法：
```bash
# 运行完整验证
python scripts/test_balance_serve.py
```

## 环境要求

### 系统要求
- **操作系统**: Ubuntu 22.04 LTS（推荐）
- **CPU**: 支持 AMX 指令的处理器（如 W3-2423）
- **内存**: 最少 32GB，推荐 512GB（用于大模型）
- **存储**: 至少 50GB 可用空间

### 软件依赖
- **CUDA**: 12.1 或更高版本
- **Python**: 3.11（推荐使用 conda 环境）
- **GCC/G++**: 11 或更高版本
- **CMake**: 3.25 或更高版本

### 开发容器配置
项目包含了优化的 `.devcontainer/Dockerfile`，包含：
- 所有 balance_serve 相关依赖
- 更新的工具链（支持 CUDA20 语言方言）
- 开发和调试工具
- 预配置的环境变量

## 编译配置

### 环境变量
```bash
USE_BALANCE_SERVE=1          # 启用 balance_serve 功能
USE_NUMA=0                   # NUMA 支持（双 CPU 系统设为 1）
KTRANSFORMERS_FORCE_BUILD=TRUE
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX"
DEV_BACKEND=cuda             # 后端类型：cuda 或 cpu
```

### Balance_Serve 特定依赖
脚本会自动安装以下依赖：
```bash
# 系统库
libtbb-dev libssl-dev libcurl4-openssl-dev 
libaio1 libaio-dev libfmt-dev libgflags-dev 
zlib1g-dev patchelf libnuma-dev

# Python 包
aiohttp zmq openai packaging ninja 
cpufeature numpy torch flash-attn
```

## 故障排除

### 常见问题

1. **GLIBCXX_3.4.32 not found**
   ```bash
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt-get update
   sudo apt-get install --only-upgrade libstdc++6
   ```

2. **CMake 版本过低**
   ```bash
   # 脚本会自动添加 Kitware APT 仓库
   # 或手动安装更新版本的 CMake
   ```

3. **CUDA 编译错误**
   ```bash
   # 检查 CUDA 环境变量
   echo $CUDA_HOME
   echo $PATH | grep cuda
   
   # 重新设置环境变量
   export CUDA_HOME=/usr/local/cuda
   export PATH=$PATH:/usr/local/cuda/bin
   ```

4. **内存不足**
   ```bash
   # 减少并行编译进程
   export MAX_JOBS=4
   
   # 或使用交换文件
   sudo fallocate -l 32G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### 日志和诊断
- 完整构建日志：`build_YYYYMMDD_HHMMSS.log`
- 验证报告：运行 `verify_ktransformers.sh`
- 详细错误信息：使用 `--verbose` 选项

## 开发工作流

### 推荐的开发流程
1. **首次设置**：
   ```bash
   ./scripts/build_ktransformers.sh --clean
   ./scripts/verify_ktransformers.sh
   ```

2. **日常开发**：
   ```bash
   # 修改代码后快速重编译
   ./scripts/quick_build.sh
   
   # 验证更改
   ./scripts/verify_ktransformers.sh
   ```

3. **测试 balance_serve**：
   ```bash
   python ktransformers/server/main.py \
     --model_path <model_config_path> \
     --gguf_path <gguf_files_path> \
     --backend_type balance_serve \
     --max_batch_size 4 \
     --cache_lens 32768 \
     --port 10002
   ```

### 性能优化建议
- 使用 `--numa` 选项（双 CPU 系统）
- 设置合适的 `--cpu_infer` 参数
- 根据 VRAM 调整 `--max_batch_size` 和 `--cache_lens`

## 贡献指南

如需改进脚本或报告问题：
1. 运行 `verify_ktransformers.sh` 获取详细环境信息
2. 提供完整的构建日志
3. 说明具体的错误现象和复现步骤

---

**注意**: 这些脚本专门针对 KTransformers 的 balance_serve 功能进行了优化，确保在 Ubuntu 22.04 LTS 环境中的稳定性和性能。
