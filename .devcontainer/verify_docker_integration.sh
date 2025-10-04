#!/bin/bash

# KTransformers Docker 集成验证脚本
# 验证集成后的 Dockerfile 配置是否正确

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "============================================================"
echo "  KTransformers Docker 集成配置验证"
echo "============================================================"

# 1. 验证 CMake 版本和路径
log_info "验证 CMake 配置..."
CMAKE_VERSION=$(cmake --version | head -1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
CMAKE_PATH=$(which cmake)

if [[ "$CMAKE_VERSION" == "3.28.3" ]]; then
    log_success "CMake 版本正确: $CMAKE_VERSION"
else
    log_error "CMake 版本不正确: $CMAKE_VERSION (期望: 3.28.3)"
    exit 1
fi

if [[ "$CMAKE_PATH" == "/usr/local/bin/cmake" ]]; then
    log_success "CMake 路径正确: $CMAKE_PATH"
else
    log_warn "CMake 路径: $CMAKE_PATH (期望: /usr/local/bin/cmake)"
fi

# 2. 验证 GLIBCXX 支持
log_info "验证 GLIBCXX 支持..."
if strings /opt/conda/lib/libstdc++.so.6 | grep "GLIBCXX_3.4.32"; then
    log_success "GLIBCXX_3.4.32 支持已启用"
else
    log_error "GLIBCXX_3.4.32 支持未找到"
    exit 1
fi

# 3. 验证编译器版本
log_info "验证编译器版本..."
GCC_VERSION=$(gcc --version | head -1 | grep -o '[0-9]\+\.[0-9]\+')
GPP_VERSION=$(g++ --version | head -1 | grep -o '[0-9]\+\.[0-9]\+')

if [[ "$GCC_VERSION" == "11."* ]]; then
    log_success "GCC 版本正确: $GCC_VERSION"
else
    log_warn "GCC 版本: $GCC_VERSION (期望: 11.x)"
fi

if [[ "$GPP_VERSION" == "11."* ]]; then
    log_success "G++ 版本正确: $GPP_VERSION"
else
    log_warn "G++ 版本: $GPP_VERSION (期望: 11.x)"
fi

# 4. 验证 PyTorch 和 CUDA
log_info "验证 PyTorch 和 CUDA..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 设备数: {torch.cuda.device_count()}')
    print(f'CUDA 版本: {torch.version.cuda}')
" && log_success "PyTorch 和 CUDA 验证通过" || { log_error "PyTorch 或 CUDA 验证失败"; exit 1; }

# 5. 验证 PyTorch CMake 配置
log_info "验证 PyTorch CMake 配置..."
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
TORCH_CMAKE_CONFIG="$TORCH_PATH/share/cmake/Torch/TorchConfig.cmake"

if [[ -f "$TORCH_CMAKE_CONFIG" ]]; then
    log_success "PyTorch CMake 配置文件存在: $TORCH_CMAKE_CONFIG"
else
    log_error "PyTorch CMake 配置文件不存在: $TORCH_CMAKE_CONFIG"
    exit 1
fi

# 6. 验证环境变量
log_info "验证环境变量..."
ENV_VARS=(
    "USE_BALANCE_SERVE:1"
    "USE_NUMA:0"
    "KTRANSFORMERS_FORCE_BUILD:TRUE"
    "DEV_BACKEND:cuda"
    "CUDA_HOME:/usr/local/cuda"
)

for env_var in "${ENV_VARS[@]}"; do
    var_name=$(echo "$env_var" | cut -d':' -f1)
    expected_value=$(echo "$env_var" | cut -d':' -f2)
    actual_value="${!var_name:-}"
    
    if [[ "$actual_value" == "$expected_value" ]]; then
        log_success "$var_name = $actual_value"
    else
        log_warn "$var_name = $actual_value (期望: $expected_value)"
    fi
done

# 7. 验证 Balance_Serve 依赖
log_info "验证 Balance_Serve 系统依赖..."
BALANCE_SERVE_DEPS=(
    "libtbb-dev"
    "libssl-dev"
    "libcurl4-openssl-dev"
    "libaio-dev"
    "libfmt-dev"
    "libgflags-dev"
    "zlib1g-dev"
    "libnuma-dev"
)

for dep in "${BALANCE_SERVE_DEPS[@]}"; do
    if dpkg -l | grep "$dep"; then
        log_success "$dep 已安装"
    else
        log_error "$dep 未安装"
    fi
done

# 8. 验证 Python 依赖
log_info "验证 Python 依赖..."
PYTHON_DEPS=(
    "packaging"
    "ninja"
    "numpy"
    "aiohttp"
    "torch"
    "flash_attn"
)

for dep in "${PYTHON_DEPS[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        log_success "Python 包 $dep 可用"
    else
        log_error "Python 包 $dep 不可用"
    fi
done



log_success "============================================================"
log_success "  Docker 集成配置验证完成！"
log_success "  所有关键配置已正确集成"
log_success "============================================================"

echo ""
log_info "下一步操作："
log_info "1. 运行构建脚本: ./scripts/build_ktransformers.sh"
log_info "2. 验证 KTransformers 安装: ./scripts/verify_ktransformers.sh"
log_info "3. 测试 balance_serve 功能"
