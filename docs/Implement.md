# Implement

我们需要让Ktransformers后端支持Qwen3MoE模型功能。并且同时保留CPU卸载的功能。

## 0 Environment

1. Git 分支
   我们需要新建一个分支来实现支持Qwen3MoE模型功能

```bash
git checkout -b support-qwen3moe
```

## 1. 模型注册与初始化链路修改

这是让 ktransformers 后端“认识” Qwen3MoE 模型的第一步。

- 问题: 当前的 ktransformers 后端在代码层面（ktransformers/local_chat.py）没有注册 Qwen3MoE 模型。当加载 Qwen3 模型时，系统会回退到标准的 HuggingFace
  实现，这会导致无法使用 GGUF 量化、自定义算子和多设备切分等核心功能。
- 修改内容:
  1.  **创建新的模型封装**:
      在 `ktransformers/models/custom_modeling_qwen3_moe_static.py:1` 新增 `KQwen3MoeForCausalLMStatic`，仅覆写 `_supports_static_cache = True`，以便保留 HuggingFace 原始 forward 的同时允许 StaticCache 初始化。
  - 在 `ktransformers/operators/models.py:500` 引入 `KQwen3MoeModel`，完全仿照 `KQwen2MoeModel`：
    - 构造函数接受 `per_layer_prefill_intput_threshold`/`transfer_map` 等参数，保持 YAML 可控。
    - `forward` 中沿用 Qwen3 原生计算流程，但在与 Qwen2 相同的位置调用 `load_layer_to(..., InferenceState.*)`，确保预填时加载 GPU 专家、结束后可卸载回 CPU。
    - `load_layer_to` 针对注意力投影、MoE Gate/Experts、线性层、LayerNorm 分别调用 `set_inference_mode` 或 `.to(device)`，从而复用了 `KTransformersLinear` / `KTransformersExpertsV2` 的 CPU/GPU 切换能力。
      整体工作流：
    1. 后端先实例化 `KQwen3MoeForCausalLMStatic` 并创建 `StaticCache`。
    2. 执行 `optimize_and_load_gguf`，按照 YAML 将各层替换成 `KTransformersLinear`、`KTransformersExpertsV2` 等注入算子，并把 `model` 换成 `KQwen3MoeModel`。
    3. 运行时所有 `self.model(...)` 调用都会落到 `KQwen3MoeModel.forward()`，它负责 per-layer `set_inference_mode`，从而触发 CPU/GPU 专家装载与卸载。
    4. 因为规则和调度逻辑都与 Qwen2 保持一致，只要在 YAML 中写入与 Qwen2 相同的 `generate_device`/`generate_op` 等参数，就能和之前一样用 CPU 解码并把专家常驻 CPU RAM。
  2.  注册新模型: 注册改动
  - ktransformers/local_chat.py:31-125 引入 KQwen3MoeForCausalLMStatic，把
    Qwen3MoeForCausalLM 注册到 custom_models，并在 default_optimize_rules 里为其预
    留 Qwen3Moe-ktransformers.yaml；同一段逻辑把 “Qwen2/3Moe” 的注意力实现统一切换
    为 flash_attention_2。
  - ktransformers/local_chat_test.py:31-108 同步测试版 CLI 的自定义模型表与默
    认规则，确保测试入口与正式入口一致，也在长文本提示里提醒 Qwen MoE 默认用
    flash_attention_2。
  - ktransformers/server/backend/interfaces/ktransformers.py:36-90 服务端接口加载时
    识别 Qwen3MoeForCausalLM，沿用自定义建模类并强制 flash_attention_2，保证推理端
    与 CLI 的注册保持一致。

  单元测试
  - ktransformers/tests/unit/test_qwen3moe_backend.py:35-389 新增 7 个单测，分别
    校验运行态/测试态注册表、接口初始化时的注意力实现，以及 KQwen3MoeModel 在缓
    存推进、阈值卸载和 load_layer_to 切换推理模式时的行为；文件顶部通过桩模块隔离
    flashinfer、prefill wrapper 与用户目录依赖，方便在 CI/CPU 环境执行。

  验证与后续
  - 已在本地执行 pytest ktransformers/tests/unit/test_qwen3moe_backend.py，所有用例
    通过。
  - 仍需后续补充真实的 ktransformers/optimize/optimize_rules/Qwen3Moe-
    ktransformers.yaml 以及结合真算子/权重的端到端验证。

## 2. GGUF 注入与算子替换

为了让模型能加载 GGUF 格式的量化权重并使用优化的算子，你需要一套新的规则。

- 问题: balance_serve 使用的 Qwen3Moe-serve.yaml 规则文件包含了与调度器和 KGQACache 强绑定的算子（如 balance_serve_attention），这些算子在单进程后端无法工作。
- 修改内容:
  1.  创建新的优化规则:
      已创建新的优化规则Qwen3Moe-ktransformers.yaml
  2.  注册新规则: 在 ktransformers/local_chat.py 文件中，更新 default_optimize_rules 字典，将 "Qwen3MoeForCausalLM" 映射到你新创建的 Qwen3Moe-ktransformers.yaml 文件。

## 3. KV Cache 与静态缓存适配

确保 StaticCache 能够正确地为 Qwen3MoE 模型分配和管理缓存。

- 问题: StaticCache 的实现需要确认能处理 Qwen3MoE 的特定配置（如 KV 头数量、维度等）。
- 修改内容:
  1.  检查缓存形状: 在 ktransformers/models/custom_cache.py 中，检查 StaticCache 的 cache_shape 计算逻辑是否能正确处理 Qwen3MoE 的 config.num_key_value_heads 和
      head_dim。如果需要，添加一个专门的分支来处理 Qwen3 的特殊情况。
  2.  确保接口兼容: 确认新的模型封装和算子在 forward 过程中调用的是 past_key_values.update 方法来写入缓存，而不是 balance_serve 特有的接口。

## 4. Token 推理流程改造 (Prefill & Decode)

确保在预填充（Prefill）和解码（Decode）阶段，数据能够正确流经新的模型。

- 问题: prefill() 和 decode_one_tokens() 函数的调用方式必须与新模型的 forward 签名完全匹配。
- 修改内容:
  1.  Prefill 阶段:
      - 新的 KQwen3MoeForCausalLMStatic 模型必须支持 inputs_embeds 和 cache_position 参数的组合调用，以对接 chunk_prefill 逻辑。
      - 模型需要能处理 self.cache.remove_suffix 操作，以便在复用历史对话时正确管理缓存。
  2.  Decode 阶段:
      - 新模型的 forward 方法必须支持仅传入 input_ids 和 cache_position 的最小化调用，这是 decode_one_tokens 和 CUDA Graph 捕获所必需的。

## 5. 接口、文档与验证

最后，完成周边的适配和验证工作。

- 修改内容:
  1.  CLI 接口: 在 ktransformers/server/args.py 中，将新的模型类名（如 KQwen3MoeForCausalLMStatic）添加到架构列表中，以确保命令行参数能够正确识别模型并计算显存。
  2.  文档更新: 在 README.md 或 docs/interface_backend.md 中，补充关于如何加载和运行 Qwen3MoE 模型的说明，包括新的架构名称和所需的规则文件。
  3.  添加测试:
      - 在 ktransformers/local_chat_test.py 中添加单元测试，验证新模型的加载和解析。
      - 编写一个简单的推理脚本，执行一次完整的 prefill + decode 流程，以确保端到端的功能正确无误。
