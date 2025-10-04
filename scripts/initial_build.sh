#!/bin/bash

# KTransformers 开发环境编译脚本
# 支持 balance_serve 功能，包含完整的错误处理和验证
# 适用于 Ubuntu 22.04 LTS + W3-2423 CPU (AMX 支持)

set -euo pipefail  # 严格错误处理

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/build_$(date +%Y%m%d_%H%M%S).log"
BUILD_START_TIME=$(date +%s)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# 错误处理函数
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "脚本在第 $line_number 行失败，退出码: $exit_code"
    log_error "查看完整日志: $LOG_FILE"
    
    # 提供诊断信息
    log_error "=== 诊断信息 ==="
    log_error "当前目录: $(pwd)"
    log_error "Python 版本: $(python --version 2>&1 || echo '未找到 Python')"
    log_error "CUDA 版本: $(nvcc --version 2>&1 | grep release || echo '未找到 CUDA')"
    log_error "CMake 版本: $(cmake --version 2>&1 | head -1 || echo '未找到 CMake')"
    log_error "GCC 版本: $(gcc --version 2>&1 | head -1 || echo '未找到 GCC')"
    
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# 显示脚本信息
show_banner() {
    echo "=================================================="
    echo "  KTransformers 开发环境编译脚本"
    echo "  支持 balance_serve 多并发功能"
    echo "  适用于 Ubuntu 22.04 LTS"
    echo "=================================================="
    echo "开始时间: $(date)"
    echo "项目根目录: $PROJECT_ROOT"
    echo "日志文件: $LOG_FILE"
    echo "=================================================="
}

# 解析命令行参数
parse_arguments() {
    # 默认配置
    export USE_BALANCE_SERVE=${USE_BALANCE_SERVE:-1}
    export USE_NUMA=${USE_NUMA:-0}
    export DEV_BACKEND=${DEV_BACKEND:-cuda}
    export KTRANSFORMERS_FORCE_BUILD=${KTRANSFORMERS_FORCE_BUILD:-TRUE}
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}
    CLEAN_BUILD=false
    SKIP_DEPS=false
    ENABLE_NUMA=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --numa)
                ENABLE_NUMA=true
                export USE_NUMA=1
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dev)
                export DEV_BACKEND="$2"
                shift 2
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
    
    log_info "编译配置:"
    log_info "  USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    log_info "  USE_NUMA: $USE_NUMA"
    log_info "  DEV_BACKEND: $DEV_BACKEND"
    log_info "  CLEAN_BUILD: $CLEAN_BUILD"
    log_info "  SKIP_DEPS: $SKIP_DEPS"
    log_info "  VERBOSE: $VERBOSE"
}

show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    --clean         清理所有构建文件后重新编译
    --skip-deps     跳过依赖检查和安装
    --numa          启用 NUMA 支持（适用于双 CPU 系统）
    --verbose       显示详细编译输出
    --dev BACKEND   指定后端类型 (cuda|cpu，默认: cuda)
    --help          显示此帮助信息

环境变量:
    USE_BALANCE_SERVE    启用 balance_serve 功能 (默认: 1)
    USE_NUMA            启用 NUMA 支持 (默认: 0)
    TORCH_CUDA_ARCH_LIST CUDA 架构列表 (默认: "8.9")

示例:
    $0                    # 标准编译
    $0 --clean --numa     # 清理重编译并启用 NUMA
    $0 --skip-deps        # 跳过依赖检查快速编译
EOF
}

# 系统环境检查
check_system_requirements() {
    log_info "检查系统环境..."
    
    # 检查操作系统
    if [[ ! -f /etc/os-release ]]; then
        log_error "无法确定操作系统版本"
        return 1
    fi
    
    local os_info=$(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)
    log_info "操作系统: $os_info"
    
    # 检查 Ubuntu 版本
    if grep -q "Ubuntu 22.04" /etc/os-release; then
        log_success "检测到 Ubuntu 22.04 LTS"
    else
        log_warn "未检测到 Ubuntu 22.04，可能存在兼容性问题"
    fi
    
    # 检查 CPU 特性（AMX 支持）
    if grep -q "amx" /proc/cpuinfo; then
        log_success "检测到 AMX 指令支持"
    else
        log_warn "未检测到 AMX 指令支持"
    fi
    
    # 检查内存
    local total_mem=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}')
    log_info "系统内存: ${total_mem}GB"
    
    if [[ $total_mem -lt 32 ]]; then
        log_warn "系统内存较少 (${total_mem}GB)，编译可能较慢"
    fi
}

# 检查必需工具
check_required_tools() {
    log_info "检查必需工具..."
    
    local required_tools=(
        "python:Python 解释器"
        "pip:Python 包管理器"
        "gcc:GNU C 编译器"
        "g++:GNU C++ 编译器"
        "cmake:CMake 构建系统"
        "ninja:Ninja 构建工具"
        "git:Git 版本控制"
    )
    
    for tool_info in "${required_tools[@]}"; do
        local tool=$(echo "$tool_info" | cut -d':' -f1)
        local desc=$(echo "$tool_info" | cut -d':' -f2)
        
        if command -v "$tool" >/dev/null 2>&1; then
            local version=$($tool --version 2>&1 | head -1 || echo "未知版本")
            log_success "$desc: $version"
        else
            log_error "缺少必需工具: $desc ($tool)"
            return 1
        fi
    done
}

# 检查 CUDA 环境
check_cuda_environment() {
    if [[ "$DEV_BACKEND" != "cuda" ]]; then
        log_info "跳过 CUDA 检查 (后端: $DEV_BACKEND)"
        return 0
    fi
    
    log_info "检查 CUDA 环境..."
    
    # 检查 CUDA 安装
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version | grep release | sed 's/.*release \([0-9.]*\).*/\1/')
        log_success "CUDA 版本: $cuda_version"
        
        # 检查版本要求
        if [[ $(echo "$cuda_version >= 12.1" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
            log_success "CUDA 版本满足要求 (>= 12.1)"
        else
            log_warn "CUDA 版本可能过低，建议使用 12.1 或更高版本"
        fi
    else
        log_error "未找到 CUDA 安装，请安装 CUDA 12.1 或更高版本"
        return 1
    fi
    
    # 检查 GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | while read line; do
            log_info "  $line"
        done
    else
        log_warn "无法获取 GPU 信息"
    fi
    
    # 检查环境变量
    local cuda_vars=("CUDA_HOME" "CUDA_PATH")
    for var in "${cuda_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            log_success "$var: ${!var}"
        else
            log_warn "$var 未设置"
        fi
    done
}

# 检查 Python 环境
check_python_environment() {
    log_info "检查 Python 环境..."

    # 检查 Python 版本
    local python_version=$(python --version 2>&1 | sed 's/Python //')
    log_info "Python 版本: $python_version"

    # 检查是否在虚拟环境中
    if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        log_success "检测到虚拟环境: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}"
    else
        log_warn "未检测到虚拟环境，建议使用 conda 或 venv"
    fi

    # 检查 PyTorch
    if python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        local torch_cuda=$(python -c "import torch; print('CUDA available:' if torch.cuda.is_available() else 'CUDA not available')" 2>/dev/null)
        log_success "PyTorch 已安装，$torch_cuda"
    else
        log_error "PyTorch 未安装或无法导入"
        return 1
    fi

    # 检查 GLIBCXX 版本
    if python -c "import ctypes; ctypes.CDLL('libstdc++.so.6')" 2>/dev/null; then
        log_success "libstdc++ 可用"

        # 检查 GLIBCXX 版本
        local glibcxx_versions=$(strings $(g++ -print-file-name=libstdc++.so.6) 2>/dev/null | grep GLIBCXX | tail -3 || echo "无法检测")
        log_info "GLIBCXX 版本: $glibcxx_versions"
    else
        log_warn "libstdc++ 检查失败"
    fi
}

# 修复构建环境（合并自 fix_build_environment.sh）
fix_build_environment() {
    log_info "=== 开始构建环境修复 ==="

    # 1. 检查 Python 环境
    log_info "1. 验证 Python 环境..."
    local python_path=$(which python3)
    if [[ -z "$python_path" ]]; then
        log_error "无法找到 Python3 可执行文件"
        return 1
    fi
    log_success "Python 路径: $python_path"

    # 2. 检查 PyTorch 安装和路径
    log_info "2. 验证 PyTorch 安装..."
    local torch_path
    if ! torch_path=$($python_path -c "import torch; print(torch.__path__[0])" 2>/dev/null); then
        log_error "无法找到 PyTorch 安装"
        log_error "请先安装 PyTorch: pip install torch"
        return 1
    fi
    log_success "PyTorch 路径: $torch_path"

    # 3. 检查 PyTorch CMake 配置
    log_info "3. 验证 PyTorch CMake 配置..."
    local torch_cmake_config="$torch_path/share/cmake/Torch/TorchConfig.cmake"
    if [[ ! -f "$torch_cmake_config" ]]; then
        log_error "PyTorch CMake 配置文件不存在: $torch_cmake_config"
        log_error "请重新安装 PyTorch 或使用包含开发文件的版本"
        return 1
    fi
    log_success "PyTorch CMake 配置: $torch_cmake_config"

    # 4. 设置构建环境变量
    log_info "4. 设置构建环境变量..."
    export PYTHON_EXECUTABLE="$python_path"
    export TORCH_INSTALL_PREFIX="$torch_path"

    # 安全地设置 CMAKE_PREFIX_PATH，处理变量可能未定义的情况
    local torch_cmake_path="$torch_path/share/cmake/Torch"
    if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
        export CMAKE_PREFIX_PATH="$torch_cmake_path:$CMAKE_PREFIX_PATH"
    else
        export CMAKE_PREFIX_PATH="$torch_cmake_path"
    fi

    log_success "环境变量设置完成:"
    log_info "  PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE"
    log_info "  TORCH_INSTALL_PREFIX: $TORCH_INSTALL_PREFIX"
    log_info "  CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

    # 5. 清理之前的构建文件（如果需要）
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        log_info "5. 清理构建文件..."
        local clean_targets=(
            "build/"
            "csrc/ktransformers_ext/build/"
            "csrc/balance_serve/build/"
            "*.egg-info/"
        )

        for target in "${clean_targets[@]}"; do
            if ls $target 2>/dev/null; then
                log_info "删除: $target"
                rm -rf $target
            fi
        done
        log_success "构建文件清理完成"
    fi

    # 6. 验证关键编译标志
    log_info "6. 验证编译标志..."
    log_info "  USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    log_info "  USE_NUMA: $USE_NUMA"
    log_info "  DEV_BACKEND: $DEV_BACKEND"
    log_info "  KTRANSFORMERS_FORCE_BUILD: $KTRANSFORMERS_FORCE_BUILD"
    log_info "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

    log_success "=== 构建环境修复完成 ==="
    return 0
}

# 清理构建文件
clean_build_files() {
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        log_info "清理构建文件..."

        local clean_dirs=(
            "build"
            "*.egg-info"
            "csrc/build"
            "csrc/ktransformers_ext/build"
            "csrc/ktransformers_ext/cuda/build"
            "csrc/ktransformers_ext/cuda/dist"
            "csrc/ktransformers_ext/cuda/*.egg-info"
            "third_party/custom_flashinfer/build"
            "third_party/custom_flashinfer/*.egg-info"
        )

        for pattern in "${clean_dirs[@]}"; do
            if ls $pattern 2>/dev/null; then
                log_info "删除: $pattern"
                rm -rf $pattern
            fi
        done

        # 清理 Python 缓存
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true

        # 清理用户配置
        if [[ -d "$HOME/.ktransformers" ]]; then
            log_info "清理用户配置: $HOME/.ktransformers"
            rm -rf "$HOME/.ktransformers"
        fi

        log_success "构建文件清理完成"
    fi
}

# 安装 Python 依赖
install_python_dependencies() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log_info "跳过依赖安装"
        return 0
    fi

    log_info "安装 Python 依赖..."

    # 检查并安装 requirements 文件
    local req_files=(
        "requirements.txt"
    )

    for req_file in "${req_files[@]}"; do
        if [[ -f "$req_file" ]]; then
            log_info "安装依赖: $req_file"
            pip install -r "$req_file" || {
                log_error "安装 $req_file 失败"
                return 1
            }
        else
            log_warn "依赖文件不存在: $req_file"
        fi
    done

    # 安装额外的开发依赖
    local dev_deps=(
        "pytest"
        "black"
        "flake8"
        "mypy"
        "ipython"
    )

    log_info "安装开发依赖..."
    for dep in "${dev_deps[@]}"; do
        pip install "$dep" || log_warn "安装 $dep 失败"
    done

    log_success "Python 依赖安装完成"
}

# 编译 KTransformers
compile_ktransformers() {
    log_info "开始编译 KTransformers..."

    # 设置编译选项
    local pip_args="-v --no-build-isolation"
    if [[ "$VERBOSE" == "true" ]]; then
        pip_args="$pip_args --verbose"
    fi

    # 显示编译环境
    log_info "编译环境变量:"
    log_info "  USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    log_info "  USE_NUMA: $USE_NUMA"
    log_info "  KTRANSFORMERS_FORCE_BUILD: $KTRANSFORMERS_FORCE_BUILD"
    log_info "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    log_info "  DEV_BACKEND: $DEV_BACKEND"

    # 执行编译
    log_info "执行: pip install $pip_args ."
    if pip install $pip_args . 2>&1 | tee -a "$LOG_FILE"; then
        log_success "KTransformers 编译成功"
    else
        log_error "KTransformers 编译失败"
        return 1
    fi
}

# 编译 custom_flashinfer
compile_custom_flashinfer() {
    if [[ "$DEV_BACKEND" != "cuda" ]]; then
        log_info "跳过 custom_flashinfer 编译 (后端: $DEV_BACKEND)"
        return 0
    fi

    log_info "编译 custom_flashinfer..."

    local flashinfer_dir="third_party/custom_flashinfer"
    if [[ ! -d "$flashinfer_dir" ]]; then
        log_error "custom_flashinfer 目录不存在: $flashinfer_dir"
        return 1
    fi

    if pip install "$flashinfer_dir/" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "custom_flashinfer 编译成功"
    else
        log_error "custom_flashinfer 编译失败"
        return 1
    fi
}

# 验证安装
verify_installation() {
    log_info "验证安装..."

    # 检查 KTransformers 导入
    if python -c "import ktransformers; print(f'KTransformers 版本: {ktransformers.__version__}')" 2>/dev/null; then
        log_success "KTransformers 导入成功"
    else
        log_error "KTransformers 导入失败"
        return 1
    fi

    # 检查 balance_serve 支持
    if [[ "$USE_BALANCE_SERVE" == "1" ]]; then
        if python -c "from ktransformers.server.backend.interfaces.balance_serve import BalanceServeInterface" 2>/dev/null; then
            log_success "balance_serve 后端可用"
        else
            log_warn "balance_serve 后端导入失败"
        fi
    fi

    # 检查 CUDA 支持（如果适用）
    if [[ "$DEV_BACKEND" == "cuda" ]]; then
        if python -c "import torch; assert torch.cuda.is_available(); print('CUDA 支持正常')" 2>/dev/null; then
            log_success "CUDA 支持验证通过"
        else
            log_warn "CUDA 支持验证失败"
        fi
    fi

    log_success "安装验证完成"
}

# 生成验证脚本
generate_verification_script() {
    local verify_script="$PROJECT_ROOT/scripts/verify_ktransformers.sh"

    log_info "生成验证脚本: $verify_script"

    cat > "$verify_script" << 'EOF'
#!/bin/bash

# KTransformers 功能验证脚本

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

echo "=================================================="
echo "  KTransformers 功能验证"
echo "=================================================="

# 基础导入测试
log_info "测试基础模块导入..."
python -c "
import ktransformers
import torch
import numpy as np
print(f'KTransformers: {ktransformers.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
" && log_success "基础模块导入成功" || { log_error "基础模块导入失败"; exit 1; }

# balance_serve 后端测试
log_info "测试 balance_serve 后端..."
python -c "
try:
    from ktransformers.server.backend.balance_serve import BalanceServeBackend
    print('balance_serve 后端可用')
except ImportError as e:
    print(f'balance_serve 后端不可用: {e}')
    exit(1)
" && log_success "balance_serve 后端可用" || log_warn "balance_serve 后端不可用"

# 优化规则测试
log_info "测试优化规则加载..."
python -c "
import os
import yaml
from pathlib import Path

rules_dir = Path('ktransformers/optimize/optimize_rules')
if rules_dir.exists():
    yaml_files = list(rules_dir.glob('*.yaml'))
    print(f'找到 {len(yaml_files)} 个优化规则文件')

    # 测试加载一个规则文件
    for yaml_file in yaml_files:
        if 'serve' in yaml_file.name:
            with open(yaml_file, 'r') as f:
                rules = yaml.safe_load(f)
            print(f'成功加载规则文件: {yaml_file.name}')
            break
else:
    print('优化规则目录不存在')
    exit(1)
" && log_success "优化规则测试通过" || log_warn "优化规则测试失败"

# 内存和性能检查
log_info "系统资源检查..."
python -c "
import psutil
import os

# 内存检查
mem = psutil.virtual_memory()
print(f'总内存: {mem.total // (1024**3)} GB')
print(f'可用内存: {mem.available // (1024**3)} GB')
print(f'内存使用率: {mem.percent:.1f}%')

# CPU 检查
cpu_count = os.cpu_count()
print(f'CPU 核心数: {cpu_count}')

# 检查 AMX 支持
try:
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    if 'amx' in cpuinfo:
        print('检测到 AMX 指令支持')
    else:
        print('未检测到 AMX 指令支持')
except:
    print('无法检查 CPU 特性')
"

log_success "验证完成！"
echo "=================================================="
EOF

    chmod +x "$verify_script"
    log_success "验证脚本已生成: $verify_script"
}

# 显示构建总结
show_build_summary() {
    local build_end_time=$(date +%s)
    local build_duration=$((build_end_time - BUILD_START_TIME))
    local build_minutes=$((build_duration / 60))
    local build_seconds=$((build_duration % 60))

    echo "=================================================="
    echo "  构建总结"
    echo "=================================================="
    log_success "构建完成时间: $(date)"
    log_success "总耗时: ${build_minutes}分${build_seconds}秒"
    echo ""
    log_info "配置信息:"
    log_info "  USE_BALANCE_SERVE: $USE_BALANCE_SERVE"
    log_info "  USE_NUMA: $USE_NUMA"
    log_info "  DEV_BACKEND: $DEV_BACKEND"
    echo ""
    log_info "下一步操作:"
    log_info "  1. 运行验证脚本: ./scripts/verify_ktransformers.sh"
    log_info "  2. 查看完整日志: $LOG_FILE"
    log_info "  3. 开始使用 KTransformers 进行开发"
    echo ""

    if [[ "$USE_BALANCE_SERVE" == "1" ]]; then
        log_info "balance_serve 使用示例:"
        echo "  python ktransformers/server/main.py \\"
        echo "    --model_path <model_config_path> \\"
        echo "    --gguf_path <gguf_files_path> \\"
        echo "    --backend_type balance_serve \\"
        echo "    --max_batch_size 4 \\"
        echo "    --cache_lens 32768 \\"
        echo "    --port 10002"
    fi

    echo "=================================================="
}

# 主函数
main() {
    # 切换到项目根目录
    cd "$PROJECT_ROOT"

    # 显示横幅
    show_banner

    # 解析参数
    parse_arguments "$@"

    # 系统检查
    log_info "开始系统环境检查..."
    check_system_requirements
    check_required_tools
    check_cuda_environment
    check_python_environment

    # 修复构建环境
    fix_build_environment

    # 清理构建文件
    clean_build_files

    # 安装依赖
    install_python_dependencies

    # 编译 KTransformers
    compile_ktransformers

    # 编译 custom_flashinfer
    compile_custom_flashinfer

    # 验证安装
    verify_installation

    # 显示构建总结
    show_build_summary

    log_success "KTransformers 编译完成！"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
