# Ktransformer Redevelop for Hybrid CPU+GPU Scheduling
This project aims to build a hybrid CPU+GPU scheduling framework, based KTransformer, to optimize large Mixture-of-Experts (MoE) model inference on consumer hardware with limited VRAM.

**Problem:** The primary bottleneck in running large MoE models on consumer hardware is the latency caused by offloading expert weights to system DRAM. During the autoregressive decoding phase, the default strategy is using CPU to compute all experts, which leads to high latency due to the limited CPU compute power. On the other hand, offloading experts to GPU VRAM can significantly speed up computation, but the limited VRAM capacity means that not all experts can fit simultaneously. This results in frequent PCIe transfers when an expert is not in the GPU cache, which adds substantial latency.

**Proposed Solution:** We propose a hybrid scheduling system that dynamically allocates expert computations between the CPU and GPU to minimize end-to-end latency. The core idea is to treat the problem as a real-time optimization task.

The key components of our proposed solution are:

- **Mathematical Cost Model**: We've formulated a model to calculate the "cost" of running an expert on either the CPU or the GPU. The GPU cost includes the potential PCIe transfer time if the expert is not already in the VRAM cache.
- **Dynamic Real-Time Scheduler**: Based on the cost model, a greedy scheduling algorithm will decide, on a per-token basis, where to execute each of the selected top-k experts. It will prioritize moving an expert to the GPU only if the performance gain outweighs the transfer latency.
- **PCIe-Aware Caching**: The limited GPU VRAM is treated as a cache for experts. We plan to implement a value-driven eviction policy that, unlike simple LRU/LFU, decides which expert to remove from the cache based on a prediction of its future usefulness, thus maximizing the chances of a "cache hit" and avoiding PCIe transfers.

The ultimate goal is to create a practical and adaptive framework that intelligently uses all available compute resources (CPU and GPU) to make inference for large MoE models on consumer-grade hardware as fast as possible.

# How to Build and Develop Ktransformers in Devcontainer
This is simplified KTransformer inference, and mainly support for Ubuntu 22/24.
In here will provide you some basic instruction about how to build the Ktransformer in devcontainer. I improved `.devcontainer` to build a image environment with all essential tools. In `scripts` folder, I provided several useful scripts to initial build and develop build.
这是简化版的KTransformer，对于开发环境和脚本等进行了大面积重构，保持简洁与稳定性。使用 `.devcontainer` 作为基础开发镜像环境，并在 `scripts` 目录下提供了多种实用脚本。

## 1. Using devcontainer
You can use `devcontainer` to build a image with all essential tools. Just open this folder in VSCode and it will prompt you to reopen in container. If you don't have docker installed, please install docker first.
你可以使用 `devcontainer` 来构建一个包含所有必要工具的镜像。只需在 VSCode 中打开此文件夹，它会提示你重新在容器中打开。如果你还没有安装 Docker，请先安装 Docker。

In devcontainer, I set a lot of environment variables to make sure the build is consistent. 
在 devcontainer 中，我设置了许多环境变量以确保构建的一致性。

```bash
# KTransformers Build Environment Variables
USE_BALANCE_SERVE=1
USE_NUMA=0
KTRANSFORMERS_FORCE_BUILD=TRUE
TORCH_CUDA_ARCH_LIST="8.9" # Changed to support 8.9 only
DEV_BACKEND=cuda

# CUDA Configuration
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda

# PyTorch CMake Configuration
PYTHON_EXECUTABLE=/opt/conda/bin/python3
TORCH_INSTALL_PREFIX=[PyTorch Installation Path]
CMAKE_PREFIX_PATH=[PyTorch CMake Configuration Path]:$CMAKE_PREFIX_PATH
```
Please note that `TORCH_CUDA_ARCH_LIST` has been changed to support only 8.9 to simplify the build process.
请注意，`TORCH_CUDA_ARCH_LIST` 已更改为仅支持 8.9，以简化构建过程。

After the container is built, you can run the `verify_docker_integration.sh` script to check if everything is set up correctly.
容器构建完成后，你可以运行 `verify_docker_integration.sh` 脚本来检查是否一切设置正确。

- - -
Errors I Met:
1. in devcontainer, you have to check `mounts` in `devcontainer.json`, make sure you mount the current folder to `/workspace` in container and the model data and gguf files.

## 2. Useful Scripts in `scripts` Folder
In the `scripts` folder, I provided several useful scripts to help you build and develop Ktransformers. Here are the main scripts:
在 `scripts` 目录下，我提供了几种实用脚本来帮助你构建和开发 Ktransformers。以下是主要脚本：

### First time build
Before build the Ktransformer, you must install submoudle of git. This is crucial for total process.
在构建 Ktransformer 之前，您必须安装 git 的子模块。这对于整个过程至关重要。
```bash
# Install git submodule
git submodule update --init --recursive
```

using `initial_build.sh` to initial build the Ktransformers. This procedure will cost a lot of time based on your CPU performance, which always about 20 mins in a normal machine.
第一次构建使用 `initial_build.sh` 来初始化构建 Ktransformers。此过程将根据您的 CPU 性能花费大量时间，在普通机器上通常约为 20 分钟。
Besides, this script will give you a detailed log about the build process, and you can check the log to find out if there are any errors.
此外，此脚本将为您提供有关构建过程的详细日志，您可以检查日志以找出是否有任何错误。
```bash
# Initial build
./scripts/initial_build.sh
```
After the initial build, you can use `test_balance_serve.py` to do verify the situation of KTransformers, especially the balance_serve backend.
初始构建完成后，您可以使用 `test_balance_serve.py` 来验证 KTransformers 的情况，特别是 balance_serve 后端。

```bash
# Test balance serve
python ./scripts/test_balance_serve.py
```
### Develop build
After the initial build, you can use `quick_build.sh` to do incremental build. This script will only build the changed files, which will save a lot of time.
初始构建完成后，您可以使用 `quick_build.sh` 进行增量构建。此脚本将仅构建更改的文件，从而节省大量时间。
```bash
# Quick build
./scripts/quick_build.sh
```

- - -
Errors I Met:
1. Forget to clear all submodule of git, which cause lack of a build_info.cmake file. So every time when you clone the repo, remember to use `git submodule update --init --recursive` to install all submodules.
2. If you cannot run the test_balance_serve.py and lack of some module, please check the requirements.txt and install the modules by pip.

## 3. Install models and gguf files
You can download the models and gguf files from the official website or other sources. After downloading, you can put them in the `models` folder. Make sure to create the `models` folder if it does not exist.
您可以从官方网站或其他来源下载模型和 gguf 文件。下载后，您可以将它们放在 `models` 文件夹中。如果不存在，请确保创建 `models` 文件夹。

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ~/models/qwen3moe

huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir ~/models/qwen3moe/

huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --local-dir ~/models/qwen3moe-gguf/2507/bf16 --include "BF16/*.gguf" 

huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --local-dir ~/models/qwen3moe-gguf/2507/q8 --include Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
```

## 4. Run the server
After building and installing the models, you can run the server by using the following command:
构建和安装模型后，您可以使用以下命令运行服务器：

```bash
# Standard multi-concurrency backend (non-AMX)
export TORCH_CUDA_ARCH_LIST="8.9"
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /workspace/data/models/qwen3moe \
  --gguf_path /workspace/data/models/qwen3moe-gguf/2507/q8 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml \
  --backend_type balance_serve

# AMX optimized backend
# Only support Intel 4th Gen Xeon (Sapphire Rapids) and later CPUs with AMX support
# Only support BF16 gguf files
export TORCH_CUDA_ARCH_LIST="8.9"
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /workspace/data/models/qwen3moe \
  --gguf_path /workspace/data/models/qwen3moe-gguf/BF16 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve-amx.yaml \
  --backend_type balance_serve
```

## 5. Measure Qwen3 performance with `test_speed.py`
Use the async benchmarking script in `ktransformers/tests/test_speed.py` to stream tokens from the running server and collect prefill / decode speeds for Qwen3.
使用 `ktransformers/tests/test_speed.py` 中的异步基准测试脚本从正在运行的服务器流式传输令牌，并收集 Qwen3 的预填充/解码速度。

1. **Confirm the chat server is running.** Launch the server with the desired backend using the commands in the previous section and note the listening URL (default `http://localhost:10002/v1/chat/completions`) and the exposed model name.
2. **Install Python dependencies.** Make sure the runtime environment already has the packages from `requirements.txt` installed (in particular `aiohttp`, `numpy`, and `yaml`).
3. **Run the benchmark script.** Invoke the tester with the server URL, model name, prompt length, and concurrency that you want to measure:

```bash
python ktransformers/tests/test_speed.py \
    --api_url http://localhost:10002/v1/chat/completions \
    --model Qwen3-30B-A3B-Instruct-2507 \
    --prompt_lens 1024 \
    --max_tokens 512 \
    --concurrent 4
```

   - `--prompt_lens` chooses one of the built-in prompts (1024, 2048, or 4096 tokens) to stress the prefill stage.
   - `--max_tokens` limits the streamed completion length; raise this to study longer decode phases.
   - `--concurrent` controls how many simultaneous requests are issued to probe throughput.
   - Ensure `--model` matches the identifier exported by your server (for example the name you register via `--architectures`).

4. **Inspect the output.** The script prints the streamed text for each request and reports `prefill speed` and `decode speed` (tokens/s). When all requests finish it also aggregates the total prefill and decode speeds so you can compare different server settings quickly.