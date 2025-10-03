FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel as ktransformers_dev

# ============================================================================
# 基础环境配置
# ============================================================================

# 设置工作目录
WORKDIR /workspace

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# 设置 KTransformers 编译环境变量
ENV USE_BALANCE_SERVE=1
ENV USE_NUMA=0
ENV KTRANSFORMERS_FORCE_BUILD=TRUE
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX"
ENV DEV_BACKEND=cuda
ENV CLEAN_BUILD=true
ENV SKIP_DEPS=false
ENV VERBOSE=false

# ============================================================================
# 系统仓库和工具链配置
# ============================================================================

# 添加更新的工具链仓库以获取 GLIBCXX_3.4.32 支持
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --only-upgrade libstdc++6

# ============================================================================
# 系统依赖安装
# ============================================================================

# 安装基础开发工具和编译工具链
RUN apt-get install -y --no-install-recommends \
    # 基础开发工具
    build-essential \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    bc \
    # 编译工具链
    gcc-11 \
    g++-11 \
    ninja-build \
    patchelf \
    # Balance_Serve 核心依赖
    libtbb-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libaio1 \
    libaio-dev \
    libfmt-dev \
    libgflags-dev \
    zlib1g-dev \
    # NUMA 支持（可选）
    libnuma-dev \
    # 调试和开发工具
    gdb \
    valgrind \
    strace \
    lsof \
    net-tools \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# CMake 3.28.3 升级配置
# ============================================================================

# 下载并安装 CMake 3.28.3 以解决版本冲突问题
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.28.3-linux-x86_64.tar.gz && \
    # 安装到 /usr/local
    cp -r cmake-3.28.3-linux-x86_64/bin/* /usr/local/bin/ && \
    cp -r cmake-3.28.3-linux-x86_64/share/* /usr/local/share/ && \
    # 彻底移除旧版本 cmake，避免路径冲突
    rm -f /opt/conda/bin/cmake /usr/bin/cmake && \
    # 创建符号链接到新版本
    ln -sf /usr/local/bin/cmake /opt/conda/bin/cmake && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake && \
    # 更新 PATH 环境变量，确保 /usr/local/bin 优先
    echo 'export PATH="/usr/local/bin:$PATH"' >> /etc/environment && \
    # 清理下载文件
    rm -rf /tmp/cmake-3.28.3-linux-x86_64* && \
    # 验证安装和路径
    echo "CMake 版本验证:" && \
    /usr/local/bin/cmake --version && \
    echo "CMake 路径验证:" && \
    which cmake


# ============================================================================
# 编译器配置
# ============================================================================

# 设置默认编译器版本
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# ============================================================================
# Python 环境配置
# ============================================================================

# 升级 pip 并安装基础构建依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    # 基础构建依赖
    packaging \
    ninja \
    pyproject \
    numpy \
    cpufeature \
    # Balance_Serve 相关依赖
    aiohttp \
    zmq \
    openai \
    # 开发和调试工具
    ipython \
    jupyter \
    pytest \
    black \
    flake8 \
    mypy

# 安装 PyTorch（确保 CUDA 12.1 兼容性）
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 flash-attention
RUN pip install --no-cache-dir flash-attn

# ============================================================================
# C++ 标准库兼容性修复
# ============================================================================

# 修复 C++ 标准库兼容性，确保 GLIBCXX_3.4.32 支持
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/ && \
    # 验证 GLIBCXX 版本
    echo "验证 GLIBCXX 版本支持:" && \
    strings /opt/conda/lib/libstdc++.so.6 | grep GLIBCXX | tail -5

# ============================================================================
# PyTorch CMake 集成配置
# ============================================================================

# 设置 PyTorch CMake 配置环境变量
RUN TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])") && \
    echo "export PYTHON_EXECUTABLE=$(which python3)" >> /etc/environment && \
    echo "export TORCH_INSTALL_PREFIX=$TORCH_PATH" >> /etc/environment && \
    echo "export CMAKE_PREFIX_PATH=$TORCH_PATH/share/cmake/Torch:\$CMAKE_PREFIX_PATH" >> /etc/environment && \
    # 验证 PyTorch CMake 配置文件存在
    echo "验证 PyTorch CMake 配置:" && \
    ls -la "$TORCH_PATH/share/cmake/Torch/TorchConfig.cmake" && \
    echo "PyTorch CMake 配置路径: $TORCH_PATH/share/cmake/Torch"

# 设置运行时环境变量
ENV PYTHON_EXECUTABLE=/opt/conda/bin/python3

# ============================================================================
# 开发环境准备
# ============================================================================

# 创建开发脚本目录和工作空间
RUN mkdir -p /workspace/scripts /workspace/build

# 注意：环境初始化功能已合并到 scripts/build_ktransformers.sh 中

# ============================================================================
# 容器启动配置
# ============================================================================

# 设置默认 shell 为 bash
SHELL ["/bin/bash", "-c"]

# 环境初始化功能已合并到构建脚本中，无需自动加载

# 默认命令
CMD ["/bin/bash"]
