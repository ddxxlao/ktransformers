# KTransformers 推理接口与后端架构分析

## 概述

KTransformers 提供了三种不同的推理方式，分别适用于不同的使用场景。本文档详细分析它们的区别、特点和适用场景。

---

## 一、三种推理模式对比

### 1. local_chat（命令行交互工具）

**文件位置**: `ktransformers/local_chat.py`

**特点**:
- ✅ 独立的命令行交互式聊天工具
- ✅ 直接使用 `prefill_and_generate()` 函数进行推理
- ✅ 支持 CUDA Graph 优化
- ✅ 支持 chunk prefill
- ✅ 适合本地快速测试和开发调试
- ❌ **不是服务器，无法通过 HTTP API 访问**
- ❌ **无法使用 test_speed.py 等基于 HTTP 的性能测试脚本**

**启动方式**:
```bash
python ktransformers/local_chat.py \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path <path_to_yaml> \
  --max_new_tokens 1000 \
  --chunk_size 8192
```

**核心实现**:
- 使用 `prefill_and_generate()` 函数（位于 `ktransformers/util/utils.py`）
- 直接调用模型进行推理，无服务器层
- 交互式输入输出

---

### 2. ktransformers 后端（单并发 HTTP 服务）

**文件位置**: `ktransformers/server/backend/interfaces/ktransformers.py`

**特点**:
- ✅ 提供 OpenAI 兼容的 HTTP API 接口
- ✅ 支持 CUDA Graph 优化
- ✅ 支持 chunk prefill
- ✅ **可以使用 test_speed.py 进行性能测试**
- ⚠️ **单并发处理**（使用 `asyncio.Lock` 确保同一时间只处理一个请求）
- ⚠️ 适合轻量级单用户场景
- ❌ 不支持 continuous batching
- ❌ 多用户并发时性能受限

**启动方式**:
```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path <path_to_yaml> \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --backend_type ktransformers
```

**核心实现**:
```python
class KTransformersInterface(TransformersInterface):
    def __init__(self, args: ConfigArgs = default_args):
        # ... 初始化模型和 tokenizer
        self._infer_lock = asyncio.Lock()  # 推理锁，确保单并发
    
    async def inference(self, ...):
        async with self._infer_lock:  # 同一时间只处理一个请求
            async for v in super().inference(...):
                yield v
```

**关键特性**:
- 使用 `StaticCache` 进行 KV cache 管理
- 继承自 `TransformersInterface`
- 提供性能指标（prefill/decode time, token/s）

---

### 3. balance_serve 后端（多并发 HTTP 服务）

**文件位置**: `ktransformers/server/backend/interfaces/balance_serve.py`

**特点**:
- ✅ **多并发支持**（可同时处理多个请求）
- ✅ Continuous batching（连续批处理）
- ✅ Chunked prefill
- ✅ 基于 FlashInfer 的高性能实现
- ✅ 支持 variable batch size CUDA Graph
- ✅ **可以使用 test_speed.py 进行性能测试**
- ✅ **生产环境推荐，性能最优**
- ✅ **内置 Profiling Bridge**：自动从 `CpuProfilingRuntime` 排空捕获并写入 NDJSON，保证 balance_serve 模式下 `tokens.jsonl` 不再缺失。
- ⚠️ 架构复杂，需要更多资源

**启动方式**:
```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

**核心架构**:
```
┌─────────────────────────────────────────┐
│         FastAPI Server (main.py)        │
│    (OpenAI Compatible API Interface)    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      BalanceServeInterface              │
│  - 多进程管理                             │
│  - Token 队列分发                         │
└─────┬───────────────────────┬───────────┘
      │                       │
      │                       │
┌─────▼─────┐          ┌──────▼──────────┐
│  Engine   │◄───────► │   Scheduler     │
│  Process  │  ZMQ/RPC │   (sched_rpc)   │
│           │          │                 │
│ - Model   │          │ - Query Mgr     │
│ - Sampler │          │ - Batch Mgr     │
│ - Runner  │          │ - Continuous    │
└───────────┘          │   Batching      │
                       └─────────────────┘
```

**核心实现**:
- **Engine 进程**: 负责模型推理、采样
- **Scheduler 进程**: 负责请求调度、批处理管理
- **多进程通信**: 使用 ZMQ 和多进程队列
- **KV Cache**: 使用 `KDeepSeekV3Cache` 或 `KGQACache`（支持分页）

**性能提升**:
- 4 并发下总吞吐量提升约 **130%**（根据官方文档）
- 支持 GPU 共享，提高资源利用率


