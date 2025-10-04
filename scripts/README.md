# KTransformers 开发环境构建指南

本目录包含了用于 KTransformers 开发环境配置和编译的脚本集合，特别针对 balance_serve 功能进行了优化。

## 脚本概览

### 1. `initial_build.sh` - 完整编译脚本
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
./scripts/initial_build.sh

# 清理重编译
./scripts/initial_build.sh --clean

# 启用 NUMA 支持（双 CPU 系统）
./scripts/initial_build.sh --numa

# 详细输出模式
./scripts/initial_build.sh --verbose

# 跳过依赖检查（快速重编译）
./scripts/initial_build.sh --skip-deps
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

### 3. `test_balance_serve.py` - 功能验证脚本
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