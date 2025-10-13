# Requirements: Enabling Qwen3MoE on the `ktransformers` Backend

## Goal

Make the single-process `ktransformers` HTTP backend load and serve Qwen3 MoE models (e.g. `Qwen3MoeForCausalLM`) so the team can focus on operator profiling without the extra complexity of the `balance_serve` multi-process workflow.
也就是我们想要让Ktransformers后端支持Qwen3MoE模型。但是，由于ktransformer后端长时间没有维护，导致添加支持会非常困难。

## Current Gaps

- `ktransformers/server/backend/interfaces/ktransformers.py` imports `custom_models` / `default_optimize_rules` from `ktransformers/local_chat.py`. That mapping historically只列出 DeepSeek、Qwen2、Llama、Mixtral，因此传入 Qwen3 检查点时会触发回退；现在需要将 Qwen3MoE 的映射补齐并长期维护。
- 当 `config.architectures[0]` 解析为 `Qwen3MoeForCausalLM` 时，必须命中新的注册表条目，否则会退回到 HuggingFace 实现（无法加载 GGUF 与自定义算子，也没有 CPU/GPU 切分）。
- Qwen3 专用 YAML 规则（例如 `ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml`）已经存在，需要确保默认配置能识别并使用它。
- 同时，对于Cache机制，`ktransformers`后端使用 `StaticCache`，需要确认它能接受 Qwen3 的配置（KV形状、头维度等）。如果 `config._attn_implementation` 需要设置（例如 `"flash_attention_2"`），则需要模仿 `balance_serve`中的逻辑。


## Required Changes

1. **模型注册与初始化链路**
   - `KTransformersInterface.__init__`（`ktransformers/server/backend/interfaces/ktransformers.py:28-81`）会先解析 `AutoConfig` 与 `GenerationConfig`，随后通过 `custom_models` 实例化模型、挑选默认 YAML，并创建 `StaticCache`。要支持 Qwen3-MoE：
     - 在 `ktransformers/local_chat.py:36-53`（以及 `ktransformers/local_chat_test.py`）补充 `from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM`，并在 `custom_models` / `default_optimize_rules` 注册 `"Qwen3MoeForCausalLM"` 对应的新模型与规则文件，否则 `self.model = custom_models[...]` 会直接抛 KeyError。
     - 参照 Qwen2 分支，在 `ktransformers/server/backend/interfaces/ktransformers.py:44-47` 为 Qwen3 设置 `_attn_implementation="flash_attention_2"`、检查 `num_key_value_heads`、`use_sliding_window` 等配置，让后续静态缓存形状与推理逻辑匹配。
     - 若希望沿用 balance_server 的 `KQwen3MoeForCausalLM` 封装，则需要额外实现一个基于 `StaticCache` 的变体，或确保现有封装能够在缺少 `KGQACache`、`ForwardBatchInput` 的情况下退化运行。当前计划是参照 Hugging Face `Qwen3MoeForCausalLM` 编写一份新的 `KQwen3MoeForCausalLMStatic`（命名待定），保持标准 `forward(input_ids, attention_mask, past_key_values, cache_position, …)` 签名，并只搭配能在 `bsz_tensor=None` 下正常工作的线性/MLP/LayerNorm 算子，以匹配 `StaticCache` 的工作方式。

在单进程 ktransformers 后端里，KQwen3MoeForCausalLM 这套 balance_server 封装确实无法直接复用，原因主要在于它与调度器强绑定的几件事：

输入结构完全不同：forward() 收到的是 ForwardBatchInput，里面带 q_indptr/kv_indptr/page_idx/page_offset 等分页信息；单进程后端只有 input_ids、attention_mask、cache_position，没有这些批调度张量。
缓存实现不兼容：balance_server 依赖的是 KGQACache（分页 KV、FlashInfer 需要的索引表），而简单后端只有 StaticCache，接口集中在 past_key_values.update。
层内算子调用方式：算子被替换成 ktransformers.operators.balance_serve_*，默认会访问 bsz_tensor、FlashInfer wrapper；单 token decode、无 wrapper 的场景根本跑不起来。
所以要让 ktransformers 后端加载 Qwen3-MoE，有两条路：

方案 A：新封装单独实现
复制 Hugging Face 版本
以 transformers 的 Qwen3MoeForCausalLM 为模板，在仓库中新建类似 KQwen3MoeForCausalLMStatic 的类，保留 Hugging Face 接口和 StaticCache 流程（past_key_values.update, cache_position 等），只针对量化权重/自定义算子做修改。
最小改造点
初始化阶段仍然创建 Qwen3MoeModel、lm_head。
在 forward 中继续用 cache_position + past_key_values.update，确保 StaticCache 可用。
只要自定义的线性层 / RMSNorm / MoE block 支持 bsz_tensor=None，就可以直接替换；注意不要替换成 KQwen3MoeSparseMoeBlockV2 那种需要 CPU 专家或 FlashInfer 的版本。
新 YAML 文件
配套创建 Qwen3Moe-ktransformers.yaml，只替换能够在标准 forward 下工作的算子（RotaryEmbedding、KTransformersLinear、KQwen3MoeRMSNorm、纯 GPU 的 MoE 模块等），不要注入 balance_server 的注意力和专家算子。
方案 B：对现有 balance_server 封装做大幅“降级”
如果想沿用当前文件里的类，也需要完成以下重写：

修改 __init__ 和 forward()，改成接受 input_ids/attention_mask/past_key_values，内部不再访问 ForwardBatchInput 或 FlashInfer。
删掉 KGQACache 相关逻辑，改用 StaticCache，即调用 past_key_values.update(...) 而不是 self.cache.get_page_table()。
关联的算子要换成标准实现：例如 attention 使用 Hugging Face 自带、MoE block 重写为直接调用专家列表，不访问 flashInferAttn。
这个改动其实等同于重新写一个“StaticCache 版”封装，工作量与方案 A 基本一致，因此更推荐直接参照 Hugging Face 版本去写一份新的类，避免和 balance_server 特定逻辑纠缠在一起。

总结：是的，需要实现一个“不会依赖 ForwardBatchInput/KGQACache”的自定义封装，结构类似 m 个现有 custom_modeling_* 文件，但 forward 签名要完全保持 Hugging Face 标准，这样才能被 ktransformers 后端在 prefill() / decode_one_tokens() 里调用。


2. **GGUF 注入与算子替换**
   - `optimize_and_load_gguf` 在 `ktransformers/server/backend/interfaces/ktransformers.py:55-74` 被调用，负责读取 YAML 后执行 `gen_optimize_config` → `inject` → `load_weights`。当前的 `Qwen3Moe-serve.yaml` 旨在给 balance_server 使用，`self_attn` / `mlp` / `experts` 都替换成 `ktransformers.operators.balance_serve_*`（依赖 FlashInfer wrapper、`KGQACache`、`ForwardBatchInput`）。为单进程后端适配需要：
     - 复制并改编一份仅依赖标准 HF `forward` 签名的规则（可命名为 `Qwen3Moe-ktransformers.yaml`），改用 `ktransformers.operators.attention`、`operators.experts`、`operators.mlp` 中不绑定 balance_server 的实现，并确认线性层仍使用 `KTransformersLinear`。
     - 为新的规则设置纯 GPU 的设备映射（balance_server 版本把专家生成阶段放在 CPU，见 `Qwen3Moe-serve.yaml:56-58`，单进程后端无法复用）。
     - 在 `default_optimize_rules` 中把 `"Qwen3MoeForCausalLM"` 指向新规则，同时验证 `ModelLoaderFactory` 能根据 Qwen3 GGUF 的张量命名找到全部权重。
     - 模块可迁移性调研结果：
       - ✅ 可直接复用：`RotaryEmbedding`（`ktransformers/operators/RoPE.py:17`）、`KTransformersLinear`（`ktransformers/operators/linear.py:895`）、`KQwen3MoeRMSNorm`（`ktransformers/operators/layernorm.py:128`）、`KQwen3MoeSparseMoeBlockV2` 与 `KTransformersExpertsV2`（`ktransformers/operators/experts.py:1243`、`:1581`）。这些算子仅依赖标准的 `hidden_states` / `bsz_tensor` 入参，在 `bsz_tensor=None` 时会自动回退到 HuggingFace 的默认路径，适合在简单后端继续使用。
       - ⚠️ 需调整或谨慎使用：`KTransformersExpertsV2` 默认在生成阶段调用 CPU 推理内核（`generate_device="cpu"`），如需纯 GPU，可在 YAML 中把 `generate_op` 改为 `"KExpertsTorch"` 并验证显存占用。
       - ❌ 不可迁移：`ktransformers.operators.balance_serve_attention.KQwen3MoeAttention`（`ktransformers/operators/balance_serve_attention.py:206`）强依赖 FlashInfer wrapper、`page_idx/page_offset` 等调度数据，只有 balance_server 的 `ForwardBatchInput` 能提供；自定义模型 `KQwen3MoeForCausalLM`（`ktransformers/models/custom_modeling_qwen3_moe.py:27`）同样依赖 `KGQACache` 与多流逻辑，无法直接接入 `StaticCache`。
       - 额外注意：仓库中唯一的 Qwen3 注意力替换是 XPU 版 `KQwen3MoeAttentionIPEXLLM`（`ktransformers/operators/attention.py:876`），GPU 场景需要保留 HuggingFace 默认注意力或自行实现新的 GPU Kernel。

3. **KV Cache 与静态缓存适配**
   - `StaticCache` 的构造/更新逻辑位于 `ktransformers/models/custom_cache.py:12-120`。为 Qwen3-MoE 保证缓存有效需完成：
     - 检查 `config.num_key_value_heads` 和 `hidden_size // num_attention_heads` 是否满足默认的 `cache_shape`，必要时像 DeepSeek 分支一样添加 Qwen3 专用分支（例如 `head_dim` 不等于 `hidden_size // num_attention_heads` 时的处理）。
     - 注意 `ktransformers/models/modeling_qwen3_moe.py:560-563` 把 `_supports_static_cache` 设为 `False`。在新封装（计划命名为 `KQwen3MoeForCausalLMStatic`）中要显式将 `_supports_static_cache = True`，否则 Hugging Face 会拒绝使用 `StaticCache`。建议在新类定义里直接覆写该 class attribute，同时确认 `forward()` 始终走 `cache_position`/`past_key_values.update` 的标准流程。
     - 确保优化规则注入后的注意力实现仍调用 `past_key_values.update`（而不是 balance_server 特定的 `kv_cache.get_k_cache` 等接口），以便 `prefill()`/`decode_one_tokens()` 传入的 `cache_position` 生效。

4. **Token 流程（Prefill → Decode）改造点**
   - Prefill 阶段：`prefill()`（`ktransformers/server/backend/interfaces/ktransformers.py:132-224`）会做前缀对齐、扩容 `generated_ids`、写入 KV，并调用 `self.model(...)`。为 Qwen3-MoE：
     - 保持 YAML 中 `per_layer_prefill_intput_threshold=0`（见现有规则的根级配置），否则分段预填会强制卸载/重载层，拉长单进程延迟。
     - 新封装必须保留 `.model.embed_tokens` 属性，并在 `forward()` 支持 `inputs_embeds` + `cache_position` 组合调用，才能与 `chunk_prefill`（`ktransformers/server/backend/interfaces/ktransformers.py:204-223`）对接。调用路径为：先用 `embed_tokens` 取 `inputs_embeds`，再在每个 chunk 内调用 `self.model(inputs_embeds=..., cache_position=..., past_key_values=self.cache, return_dict=False, use_cache=True)`。
     - 当 `is_new=True` 时，prefill 会通过 `self.cache.remove_suffix(same_prefix)` 与 `cache_position = torch.arange(...)`（`ktransformers/server/backend/interfaces/ktransformers.py:166-195`）回收历史前缀。StaticCache 版模型需支持这一流程（即 `remove_suffix` 后立即继续写入）。
     - 如果后续希望支持 `config.use_sliding_window=True` 的配置，需在 `chunk_prefill` 里补齐 `attention_mask`/`position_ids` 传参；否则默认行为依赖 Hugging Face 内部根据 `cache_position` 构造掩码，应在长上下文场景中验证无误。
   - Decode 阶段：`decode_one_tokens()`（行82-127）分别处理 CUDA Graph 与常规推理，需要验证新 attention / MLP / 专家模块在单 token 输入下接口一致；若引入 FlashInfer，还要在无 warmup 时初始化 wrapper，否则需保证保守实现也能与 `StaticCache.change_seq_length(1)` 兼容。
     - `decode_one_tokens` 默认调用 `self.model(self.current_ids.to(device), cache_position=self.active_cache_position, past_key_values=self.cache, return_dict=False, use_cache=True)[0]`。新的封装必须支持这种最小签名（仅有 `input_ids` + `cache_position`），否则 CUDA Graph 捕获（`CUDAGraphRunner.capture`）无法复用。
     - 完成一次解码后会调用 `self.cache.change_seq_length(1)`（`decode_one_tokens` 中 CUDA Graph 分支）同步 StaticCache 的内部计数。因此 StaticCache 版本模型不需要额外自增，只要保证 `past_key_values.update(...)` 针对单 token 写入即可。

5. **接口、文档与验证**
   - CLI：`ktransformers/server/args.py:115-146` 会把 GGUF 配置中的 `architectures[0]` 写回 `args.architectures` 并推导显存开销。新增静态封装后需在同一分支登记 `"Qwen3MoeForCausalLMStatic"`（或最终类名），确保命令行 `--architectures` 能映射到新实现且 `gpu_memory_size` 计算正确。
   - 文档：更新 `README.md` / `docs/interface_backend.md`，说明如何选择新的规则文件、对应的架构名称（如 `Qwen3MoeForCausalLMStatic`）、GGUF 要求与 `balance_serve` 差异；必要时补充 `scripts/` 下的启动示例。
   - 测试：在 `ktransformers/local_chat_test.py` 或新增用例里覆盖 `"Qwen3MoeForCausalLMStatic"` 的解析；准备最小推理脚本执行一次 `prefill+decode` 验证 StaticCache 写入，并在 CI 中运行相关 `pytest` 子集（至少 `ktransformers/tests/unit`）。

## Implementation Checklist

为便于后续排期，把关键改动整理如下：

1. **新封装类**
   - 新增 `KQwen3MoeForCausalLMStatic`（命名可调整）以 Hugging Face 基类为骨架，覆写 `_supports_static_cache = True`，沿用 `cache_position + past_key_values.update`。

2. **优化规则**
   - 复制改写 `Qwen3Moe-serve.yaml`，产出 `ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml`，仅保留无需 FlashInfer 的算子（RotaryEmbedding、KTransformersLinear、KQwen3MoeRMSNorm、纯 GPU MoE 等）。

3. **映射注册**
   - 更新 `ktransformers/local_chat.py` 及其测试版，注册新的封装类与默认规则路径，确保 `custom_models` / `default_optimize_rules` 能解析。

4. **后端接入**
   - 在 `ktransformers/server/backend/interfaces/ktransformers.py` 中接入新封装，prefill/decode 流程沿用 `StaticCache`。

5. **启动参数**
   - 扩展 `ktransformers/server/args.py`，支持 `--architectures Qwen3MoeForCausalLMStatic` 并正确计算显存预算。

6. **文档同步**
   - 更新 `README.md`、`docs/interface_backend.md` 与相关脚本示例，说明新规则文件、封装类与使用方式。

7. **验证**
   - 补充最小推理脚本与单元/集成测试，覆盖 StaticCache 的 `prefill + decode` 流程。
