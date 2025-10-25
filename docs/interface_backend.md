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

# Prompt

## YAML && Operators

- **Q1**
  我正在实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 。再之后，请你阅读我们的实施计划 @/docs/Implement.md 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。
  我已经完成了一份基本的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml 我根据 Qwen2-57B-A14B-Instruct.yaml（这一份专门为单线程后端设计的规则）和Qwen3Moe-serve.yaml、Qwen2Moe-serve.yaml（这两份专门为多线程后端设计的规则）进行合并，划分出了几个部分，你可以在Qwen3Moe-ktransformers.yaml中看到。
  现在我需要你做的是：

1. 对于Operators both have in Qwen2-57B-A14B-Instruct.yaml and Qwen3Moe-serve.yaml, but the version is different.的算子, 我需要你先专注于"^model\\.layers\\..\*\\.mlp$"这个算子
2. 我们知道，在 Qwen2-57B-A14B-Instruct.yaml中使用的是 ktransformers.operators.experts.KQwen2MoeSparseMoeBlock。
3. 我们知道，在 Qwen2-serve.yaml中使用的是 ktransformers.operators.experts.KQwen2MoeSparseMoeBlockV2。
4. 我们知道，在 Qwen3Moe-serve.yaml中使用的是 ktransformers.operators.experts.KQwen3MoeSparseMoeBlockV2。
   所以我怀疑带V2后缀的是专门为多线程后端适配的算子，而且由于KQwen3Moe也进行了更新，所以我认为Qwen3的mlp架构进行了更新，所以才需要更新算子。请你判断是否我们需要开发一个支持Qwen3的单线程新mlp算子，叫做KQwen3MoeSparseMoeBlock

**A1** 无需新增操作符。现有的 KQwen3MoeSparseMoeBlockV2 已经完全复现了 Qwen3 的架构（无共享专家路径），并且保持了 Qwen3MoeDecoderLayer 所期望的 HF 前向调用契约——路由器 logits 为可选参数，调用方能优雅处理其返回的 None 值（参见 ktransformers/models/modeling_qwen3_moe.py:376 和 ktransformers/operators/experts.py:1581）。在运行时层面，所有受影响的依赖项均接受 bsz_tensor=None；无论是门控机制（KTransformersLinear.forward）还是专家执行器（KTransformersExpertsV2.forward），都会自动内部构造一个 batch-size 张量，如 ktransformers/operators/linear.py:924 和 ktransformers/operators/experts.py:300 所示。我们自己的需求文档甚至明确列出这一对组合在单后端工作流中可以安全重用（docs/requirement.md:55-57）。因此，ktransformers 后端可以继续使用 V2 块——只需确保 YAML 配置选择了合适的专家实现（例如，若希望纯 GPU 运行，可将 generate_op 改为 KExpertsTorch），即可完成配置。

是的。我走了一遍单后端流程以进行确认：

- 预填充和解码都使用原始的 HuggingFace 签名调用模型
  —— 参见 ktransformers/server/backend/interfaces/ktransformers.py:235-247 以及
  ktransformers/server/backend/interfaces/ktransformers.py:130-137。在此路径上，任何地方都没有构造 bsz_tensor。
- 一旦 YAML 文件切换为 KQwen3MoeSparseMoeBlockV2，其自身入口点将默认设置可选参数（bsz_tensor=None），并在该值保持为 None 时通过普通线性门控路由（ktransformers/operators/experts.py:1582-1605）。后续对专家执行器的调用持续传递相同的 None。
- KTransformersExpertsV2 仅仅转发接收到的任何内容（ktransformers/operators/experts.py:1295-1307），而默认加载设置使其为 InferenceState.GENERATE，因此在单线程后端中仅 CPU 支持的专家处于激活状态（通过 module.load() 调用 ktransformers/util/utils.py:120-138）。
- 激活的后端（KExpertsCPU.forward）明确处理缺失的 batch-size 张量，并在该值为 None 时动态生成一个（ktransformers/operators/experts.py:300-305），因此执行默认仍处于“HF 兼容”分支。
- 辅助操作符（例如 ktransformers/operators/linear.py:924-931 中的 KTransformersLinear.forward）也具有相同的可选参数，并且在 None 路径下能正确运行。

因此，在当前的单线程设置中，所有注入的模块均接收 bsz_tensor=None，而 KQwen3MoeSparseMoeBlockV2 和 KTransformersExpertsV2 都会自动回退到标准的 HuggingFace 风格行为。如果你日后启用按层预填充（threshold > 0）或切换至 GPU 专家实现，我们需要重新审视调用方签名；但对于当前工作流，无需额外更改。

目前，单线程路径在未传入第四个参数的情况下从不会调用 KTransformersExpertsV2.forward。当我们切换到 KQwen3MoeSparseMoeBlockV2 时，其 forward 方法始终会调用专家执行器，如下所示：

self.experts(hidden_states, selected_experts, routing_weights, bsz_tensor,
cuda_graph_idx)

即使 bsz_tensor 为 None 时也是如此（参见 ktransformers/operators/experts.py:1624-1627）。由于该参数被传入，ktransformers/operators/experts.py:1298-1306 处的方法签名依然满足要求。之后调用会进入活跃后端（默认为 KExpertsCPU），该后端会显式检查 bsz_tensor 是否为 None，并动态生成一个单元素张量（参见 ktransformers/operators/experts.py:300-305）。正是通过这种方式，我们保持了与单线程后端的兼容性——后者从不创建批大小张量。

只有在我们引入 KTransformersExpertsV2 但保留原始 HuggingFace 模块（仅使用三个参数调用 self.experts(...)）的情况下，才会出现 TypeError: missing required positional argument 'bsz_tensor'。而我们计划进行的配对替换——KQwen3MoeSparseMoeBlockV2 与 KTransformersExpertsV2 一同使用——将使调用签名完全匹配，无需任何更改。

如果你仍希望增加一种防御性调整，我们可以在 KTransformersExpertsV2.forward 中添加一个默认值（bsz_tensor=None），以应对未来可能的混合搭配场景。但从功能上看，当前流程已正确处理了 bsz_tensor 为 None 的情况。

- **Q2**
  我正在实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 明白单线程后端加载模型和推理运算的全部流程。再之后，请你阅读我们的实施计划 @/docs/Implement.md 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。
  我已经完成了一份基本的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml 我根据 Qwen2-57B-A14B-Instruct.yaml（这一份专门为单线程后端设计的规则）和Qwen3Moe-serve.yaml、Qwen2Moe-serve.yaml（这两份专门为多线程后端设计的规则）进行合并，划分出了几个部分，你可以在Qwen3Moe-ktransformers.yaml中看到。

  现在我需要你做的是：

1. 对于Operators only have in Qwen3Moe-serve.yaml and Qwen2-serve.yaml (both is balance_serve used optimize rules), So we need to check if it is only for balance_server attention operator.(In single concurrency, we use flash attention2)
   我需要你先专注于 ktransformers.models.modeling_qwen3_moe.Qwen3MoeRMSNorm和 ktransformers.models.modeling_qwen3_moe.Qwen3MoeMLP这两个被替换的算子
2. 这两个被替换的算子都是qwen3 moe的模块的部分，应该是更加细粒度的replace替换目标。
3. 我们知道，ktransformers.models.modeling_qwen3_moe.Qwen3MoeRMSNorm 被替换成为了ktransformers.operators.layernorm.KQwen3MoeRMSNorm 在Qwen3Moe-serve.yaml 中被替换
4. 我们知道，ktransformers.models.modeling_qwen3_moe.Qwen3MoeMLP 被替换成为了ktransformers.operators.mlp.KQwen2MoeMLP 在Qwen3Moe-serve.yaml 中被替换
5. 我们知道，ktransformers.models.modeling_qwen2_moe.Qwen2MoeRMSNorm 被替换成为了ktransformers.operators.layernorm.KQwen2MoeRMSNorm 在Qwen2-serve.yaml 中被替换
6. 我们知道，ktransformers.models.modeling_qwen2_moe.Qwen2MoeMLP 被替换成为了ktransformers.operators.mlp.KQwen2MoeMLP 在Qwen2-serve.yaml 中被替换

这次替换是替换的ktransformer已经注入后的模块，我是否可以认为这是两个是为多线程后端专门开发的替换模块。如果我们使用单线程后端，则可以不使用这两个模块

- **A2**
  可以把这两个替换理解成 balance_serve 多线程后端特化的算子，单线程 ktransformers
  后端不需要（而且会打坏流程）。
  - KQwen3MoeRMSNorm（ktransformers/operators/layernorm.py:128）在 forward 开头直
    接假设输入是 [N, hidden] 的二维张量，在单线程路径里 Qwen3MoeDecoderLayer 依旧以
    HuggingFace 的三维张量调用 self.input_layernorm(hidden_states)（ktransformers/
    models/modeling_qwen3_moe.py:378 左右），一旦替换就会因形状解包失败。
  - KQwen2MoeMLP（ktransformers/operators/mlp.py:19）新增了 bsz_tensor 这
    一 balance_serve 专用参数；单线程静态封装 KQwen3MoeForCausalLMStatic
    仍旧沿用原生调用 self.mlp(hidden_states)（ktransformers/models/
    modeling_qwen3_moe.py:405），没有办法提供第二个实参，会直接抛出 “missing
    required positional argument”。
  - 这些接口正是为 balance_serve 自定义前向（ktransformers/models/
    custom_modeling_qwen3_moe.py）服务：那里所有层都带着 bsz_tensor、
    cuda_graph_idx 等额外上下文。对比单线程已经在用的 Qwen2 规则
    Qwen2‑57B‑A14B‑Instruct.yaml（ktransformers/optimize/optimize_rules/Qwen2-57B-
    A14B-Instruct.yaml:1），也可以看到并未替换 RMSNorm/MLP，证明单线程不依赖这些
    算子。
  - 继续在单线程规则里保留它们只会导致崩溃或错误的 shape/参数处理；把 attention 换
    成 FA2（你已经做了）就够了，其余保持默认实现即可。

  所以，单线程 ktransformers 后端可以放心不加载 KQwen3MoeRMSNorm 和 KQwen2MoeMLP，
  只在 balance_serve 路径里使用它们。后续步骤就是在 Qwen3Moe-ktransformers.yaml 中
  移除这两条 match/replace 规则，并验证推理是否正常。

## KV Cache 与静态缓存适配

### Q1:

我正在实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 明白单线程后端加载模型和推理运算的全部流程。再之后，请你阅读我们的实施计划 @/docs/Implement.md 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。我已经完成了一份针对单线程后端的优化规则的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml
由于ktransformer单线程后端使用StaticCache ，请你判断是否能够正确地为 Qwen3MoE 模型分配和管理缓存。

- 问题: StaticCache 的实现需要确认能处理 Qwen3MoE 的特定配置（如 KV 头数量、维度等）。
- 修改内容:
  1.  检查缓存形状: 在 ktransformers/models/custom_cache.py 中，检查 StaticCache 的 cache_shape 计算逻辑是否能正确处理 Qwen3MoE 的 config.num_key_value_heads 和
      head_dim。如果需要，添加一个专门的分支来处理 Qwen3 的特殊情况。
  2.  确保接口兼容: 确认新的模型封装和算子在 forward 过程中调用的是 past_key_values.update 方法来写入缓存，而不是 balance_serve 特有的接口。

## Debugging

Q1: 我正在实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 明白单线程后端加载模型和推理运算的全部流程。再之后，请你阅读我们的实施计划 @/docs/Implement.md 了解我们已经进行的改动 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。我已经完成了一份针对单线程后端的优化规则的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml。现在，我进入了debug阶段，需要开始对代码进行整体调试，我的运行命令如下：
python ktransformers/server/main.py \
 --model_name Qwen3MoeForCausalLM \
 --model_path /workspace/data/models/qwen3moe \
 --gguf_path /workspace/data/models/qwen3moe-gguf/2507/q8 \
 --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml \
 --backend_type ktransformers
我们服务器已经能够成功启动，进入监听状态，但是在有输入token后，持续出现重复乱码，无法实际进行使用。我们使用python ktransformers/tests/test_speed.py --api_url http://localhost:10002/v1/chat/completions --model Qwen3-30B-A3B-Instruct-2507 --prompt_lens 1024 --max_tokens 512 --concurrent 1 进行输入token

输出如下：

.
the the the the the the, the the the the the. the the.
the the the the the the.

The the the.

.2
, the.
.
this

The the the.

' . light
the this.
the them
heart the the the the the the the the the what
the the! the some
the the the.. I

请你定位和分析错误来源，告诉我原因

Q2: Runtime Fix
我正在实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 明白单线程后端加载模型和推理运算的全部流程。再之后，请你阅读我们的实施计划 @/docs/Implement.md 了解我们已经进行的改动 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。我已经完成了一份针对单线程后端的优化规则的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml。现在，我进入了debug阶段，需要开始对代码进行整体调试，我的运行命令如下：
python ktransformers/server/main.py \
 --model_name Qwen3MoeForCausalLM \
 --model_path /workspace/data/models/qwen3moe \
 --gguf_path /workspace/data/models/qwen3moe-gguf/2507/q8 \
 --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml \
 --backend_type ktransformers
我们服务器已经能够成功启动，进入监听状态，但是在有输入token后，持续出现重复乱码，无法实际进行使用。

输出如下：

{"id":"03c175bb57084185bb825b32cc96c722","object":"chat.completion","created":1761316474,"model":"Qwen3MoeForCausalLM","choices":[{"index":0,"message":{"role":"assistant","content":" is,\n\nHello\n\nI\n\n\\\n\n\\\n\n\n\n\\\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\n\n\n\\\n\n\\\n\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\\\n\n\n\n\n\\\n\n\\\n\n\n\n\\\n\n\\\n\n\\\n\n\\\n\n"},"finish_reason":"length"}],"usage":{"prompt_tokens":14,"completion_tokens":98,"total_tokens":112,"prompt_tokens_details":null,"completion_tokens_details":null},"system_fingerprint":"fp_c76e25e0035c"}

我已经请你修复过一次bug，但是仍然存在问题，上一次修复的记录如下，但是仍然出现类似的重复乱码循环问题：

- 预填充完成后，KQwen3MoeSparseMoeBlockV2 切换到CPU卸载分支（self.experts 为 KExpertsCPU），因此每个解码步骤都会通过 KExpertsCPU.forward 调用 cpuinfer（ktransformers/operators/experts.py:134-341）。

- KExpertsCPU.load linearly hands the raw GGUF memmap buffers to the C++ kernel (see the pointer writes around ktransformers/operators/experts.py:173-215). That code assumes the expert weights are laid out [num_experts, in_dim, out_dim], exactly how Qwen2 GGUF files were organised.
- KExpertsCPU.load 以线性方式将原始的 GGUF memmap 缓冲区传递给 C++ 内核（参见 ktransformers/operators/experts.py 文件中第 173 至 215 行附近的指针写入）。该代码假设专家权重的布局为 [num_experts, in_dim, out_dim]，与 Qwen2 GGUF 文件的组织方式完全一致。

- Qwen3 的 2507 GGUF 改变了打包方式：专家轴现在位于最后（[in_dim, out_dim, num_experts]）。你可以直接从加载器元数据中看到这一点——例如，blk.0.ffn_gate_exps.weight 的形状为 [2048, 768, 128]（通过 GGUFLoader.tensor_info 在 ktransformers/util/custom_loader.py 中）。当这些缓冲区被直接传递时，cpuinfer 会交错读取不同专家的权重，导致路由专家实际上输出垃圾数据。预填充阶段看起来仍合理（因为它使用 GPU 内核），但一旦我们在 CPU 上解码，logits 就会崩溃，采样会退化为你所看到的重复“the … the”流。

- balance_serve 构建版本从未将不支持的 gguf 布局传递给 KExpertsCPU。其部署流程在解码时仍使用相同的 KExpertsCPU 类（ktransformers/operators/experts.py:134-341），但传递给 cpuinfer 的权重来自我们自己的转换路径（SafeTensorLoader.load_experts，ktransformers/util/custom_loader.py:74-166）。该加载器将专家权重以 [num_experts, in_dim, out_dim] 的形状堆叠，因此当 KExpertsCPU.load() 将每个专家流式传输到 C++ 运行时，这些切片恰好与 cpuinfer 所期望的格式对齐。

- 在你单线程测试中，你将 gguf_path 指向了厂商发布的 Qwen3Moe 量化版本（/workspace/data/models/qwen3moe-gguf/2507/q8）。我们在 GGUFLoader.tensor_info 中读取的元数据显示，这些张量现在以 [in_dim, out_dim, num_experts] 的形式存储。我们共享的 GGUF 加载器（ktransformers/util/custom_loader.py:402-436）仍然假设旧的顺序，仅按每个专家连续推进一个窗口。一旦解码切换到 CPU，KQwen3MoeSparseMoeBlockV2（ktransformers/operators/experts.py:1581-1643）中的 self.experts(...) 就会将这些错位的缓冲区传递给 cpuinfer，从而产生你观察到的无意义输出流。

- 因为 balance_serve 要么（a）通过 SafeTensorLoader 加载浮点 safetensors，要么（b）使用我们自己的量化脚本生成的 GGUF 包（这些包仍以 [num_experts, …] 形式输出），所以它永远不会遇到这种布局不匹配问题。如果你今天把相同的 2507 gguf 文件放入 balance_serve，将会触发完全相同的回归——该后端并不存在额外的修复机制。

这种差异并不在于调度或多线程逻辑，而完全在于哪个加载器管道向 KExpertsCPU 提供数据。继续使用 HF safetensors 或用我们的脚本重新量化，两个后端都会正常工作。只要你在不转置专家轴的情况下消费新的厂商 GGUF 文件（即在 GGUFLoader.load_expert_tensor 中未进行转置），任何将专家卸载到 CPU 的后端都会开始输出垃圾数据。

修复方案： - Updated ktransformers/util/custom_loader.py:402-459 so load_expert_tensor
now detects interleaved expert layouts and pulls GGUF blocks correctly before
dequantising, keeping the legacy contiguous path intact.

- Despite the loader fix, your rerun still yields repeated gibberish, so there’s
  likely another latent issue along the decode path.

修复未成功，大概率这是因为隐性错误导致很难修复，所以我们需要让程序给出更多信息，来帮助我们来定位错误，所以，我需要你仔细思考，在哪里会出现可能的错误点，并将该错误点附近如何添加信息输出，输出什么信息这些内容。

日志文件在 ~/.ktransformers/logs/lexllama.log

## Documentation:

我已经实施让单线程后端 ktransformer 支持最新的Qwen3MoE系列模型的重构，首先我想让你阅读我们总结的工作流 @/docs/single_workflow.md 明白单线程后端加载模型和推理运算的全部流程。再之后，请你阅读我们的实施计划 @/docs/Implement.md 了解我们已经进行的改动 。我们已经完成了KQwen3MoeForCausalLMStatic和KQwen3MoeModel类，并在@ktransformers/local_chat.py中实现了对Qwen3MoeForCausalLM的注册。我已经完成了一份针对单线程后端的优化规则的Qwen3Moe-ktransformers.yaml文件，位置在 @ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml。现在我需要你进行Decode阶段对于Experts专家的数据流追踪：GPU→CPU→GPU 时序与接口梳理，形成一份详尽且完善的文档，并且有对应的代码位置，下面请你开始，并将结果输出到 @/docs/dataflow.md
（请注意，你的数据流应该根据Yaml配置文件的区别而呈现不同路径，例如如果generate_device: "cpu" generate_op: "KExpertsCPU"，就是使用CPU进行推理，如果generate_device: "cuda" generate_op: "KExpertsTorch"就是使用GPU进行推理。请你根据不同的推理路径进行不同的文档整理）
