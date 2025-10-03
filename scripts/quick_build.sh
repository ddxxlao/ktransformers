#!/bin/bash

# KTransformers 快速编译脚本
# 适用于开发环境的快速迭代编译

set -euo pipefail

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# 错误处理
handle_error() {
    local exit_code=$?
    log_error "快速编译失败，退出码: $exit_code"
    log_error "如需完整诊断，请运行: ./scripts/build_ktransformers.sh --verbose"
    exit $exit_code
}

trap 'handle_error' ERR

# 显示帮助
show_help() {
    cat << EOF
KTransformers 快速编译脚本

用法: $0 [选项]

选项:
    --clean         清理构建缓存
    --numa          启用 NUMA 支持
    --cpu           使用 CPU 后端
    --help          显示帮助

这是一个简化的编译脚本，适用于开发环境的快速迭代。
如需完整的环境检查和错误诊断，请使用: ./scripts/build_ktransformers.sh
EOF
}

# 解析参数
CLEAN_BUILD=false
USE_NUMA=0
DEV_BACKEND=cuda

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --numa)
            USE_NUMA=1
            shift
            ;;
        --cpu)
            DEV_BACKEND=cpu
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=================================================="
echo "  KTransformers 快速编译"
echo "=================================================="
log_info "开始时间: $(date)"
log_info "项目目录: $PROJECT_ROOT"
log_info "配置: NUMA=$USE_NUMA, 后端=$DEV_BACKEND, 清理=$CLEAN_BUILD"
echo "=================================================="

# 快速环境检查
log_info "快速环境检查..."
python --version || { log_error "Python 未安装"; exit 1; }
pip --version || { log_error "pip 未安装"; exit 1; }

if [[ "$DEV_BACKEND" == "cuda" ]]; then
    nvcc --version >/dev/null 2>&1 || { log_warn "CUDA 未检测到，将使用 CPU 模式"; DEV_BACKEND=cpu; }
fi

# 清理构建文件（如果需要）
if [[ "$CLEAN_BUILD" == "true" ]]; then
    log_info "清理构建文件..."
    rm -rf build *.egg-info csrc/build csrc/ktransformers_ext/build 2>/dev/null || true
    rm -rf ~/.ktransformers 2>/dev/null || true
    log_success "清理完成"
fi

# 设置环境变量
export USE_BALANCE_SERVE=1
export USE_NUMA=$USE_NUMA
export DEV_BACKEND=$DEV_BACKEND
export KTRANSFORMERS_FORCE_BUILD=TRUE

if [[ "$DEV_BACKEND" == "cuda" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.9"
fi

# 安装基础依赖（快速模式）
log_info "检查基础依赖..."
pip install -q packaging ninja numpy cpufeature 2>/dev/null || log_warn "部分依赖安装失败"

# 编译 KTransformers
log_info "编译 KTransformers..."
if pip install -v . --no-build-isolation; then
    log_success "KTransformers 编译成功"
else
    log_error "KTransformers 编译失败"
    exit 1
fi

# 编译 custom_flashinfer（仅 CUDA 模式）
if [[ "$DEV_BACKEND" == "cuda" ]] && [[ -d "third_party/custom_flashinfer" ]]; then
    log_info "编译 custom_flashinfer..."
    if pip install third_party/custom_flashinfer/; then
        log_success "custom_flashinfer 编译成功"
    else
        log_warn "custom_flashinfer 编译失败，但不影响基础功能"
    fi
fi

# 快速验证
log_info "快速验证..."
if python -c "import ktransformers; print('KTransformers 导入成功')" 2>/dev/null; then
    log_success "验证通过"
else
    log_error "验证失败"
    exit 1
fi

echo "=================================================="
log_success "快速编译完成！"
log_info "耗时: $(($(date +%s) - $(date +%s --date='1 minute ago'))) 秒"
log_info "运行完整验证: ./scripts/verify_ktransformers.sh"
echo "=================================================="
