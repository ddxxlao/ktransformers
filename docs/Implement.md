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

1. cache_position参数支持 - ✅ 已实现

- Qwen3MoeForCausalLM.forward (modeling_qwen3_moe.py:1086): ✅
  包含cache_position: Optional[torch.LongTensor] = None参数
- Qwen3MoeModel.forward (modeling_qwen3_moe.py:690): ✅ 包含cache_position:
  Optional[torch.LongTensor] = None参数
- 参数传递 (modeling_qwen3_moe.py:1143): ✅
  cache_position=cache_position正确传递给model

2. ktransformers调用模式 - ✅ 完全匹配

Prefill阶段调用 (ktransformers.py:241-247):
logits = self.model(
inputs_embeds=inputs_embeds, # ✅ Qwen3MoE支持
cache_position=cache_position, # ✅ Qwen3MoE支持
past_key_values=self.cache, # ✅ 类型兼容
return_dict=False,
use_cache=True,
)

Decode阶段调用 (ktransformers.py:131-137):
logits = self.model(
self.current_ids.to(torch_device), # ✅ input_ids参数支持
cache_position=self.active_cache_position, # ✅ cache_position支持
past_key_values=self.cache, # ✅ past_key_values支持StaticCache
return_dict=False,
use_cache=True,
)

3. past_key_values类型兼容性 - ✅ 运行时正确

- 类型声明: Qwen3MoE的past_key_values: Optional[List[torch.FloatTensor]] =
  None确实是HF上游的旧标注
- 实际运行:
  模型正确接受StaticCache（继承自Cache基类）并调用past_key_value.update()
- Cache更新 (modeling_qwen3_moe.py): ✅
  正确调用past_key_value.update(key_states, value_states, self.layer_idx,
  cache_kwargs)

4. 设备映射和算子支持 - ✅ 完全兼容

- 设备传递 (operators/models.py:619-623): ✅ cache_position正确传递到目标设备
- StaticCache操作: ✅ remove_suffix()和update()方法已实现并测试
- KQwen3MoeModel: ✅ 已实现，支持CPU/GPU设备切换

## 5. 接口、文档与验证

最后，完成周边的适配和验证工作。

- 修改内容:
  1.  CLI 接口: 在 ktransformers/server/args.py 中，将新的模型类名（如 KQwen3MoeForCausalLMStatic）添加到架构列表中，以确保命令行参数能够正确识别模型并计算显存。
  2.  文档更新: 在 README.md 或 docs/interface_backend.md 中，补充关于如何加载和运行 Qwen3MoE 模型的说明，包括新的架构名称和所需的规则文件。
  3.  添加测试:
      - 在 ktransformers/local_chat_test.py 中添加单元测试，验证新模型的加载和解析。
      - 编写一个简单的推理脚本，执行一次完整的 prefill + decode 流程，以确保端到端的功能正确无误。

# Debug

## Bug1:StaticCache初始化错误

问题根源

错误发生在ktransformers/models/custom_cache.py:38行，自定义StaticCache类错误
地调用了Cache.**init**(self)而没有提供transformers
Cache基类要求的必需参数layers或layer_class_to_replicate。

修复方案

按照您的要求，我已完善修复：

1. ✅ 在计算self.max_cache_len之后调用super().init：
   - 先计算self.max_cache_len = config.max_position_embeddings if

max_cache_len is None else max_cache_len - 然后调用super().**init**(config=config, max_cache_len=self.max_cache_len,
**kwargs) 2. ✅ 方法签名接受kwargs并转发**：- 修改方法签名为def **init**(self, ..., **kwargs) -> None: - 将kwargs转发给父类：super().**init**(config=config,
max_cache_len=self.max_cache_len, **kwargs) 3. ✅ 保持自定义逻辑不变：- super()调用后继续执行原有的key_cache/value_cache张量分配逻辑 - 保持设备字典处理和其他自定义功能

修复后的调用链

KTransformersInterface.**init**()
→ StaticCache(config, max_batch_size, max_cache_len, device, dtype)
→ super().**init**(config=config, max_cache_len=self.max_cache_len,
**kwargs)
→ transformers.StaticCache.**init**(config, max_cache_len, **kwargs)
→ 创建layers并调用Cache.**init**(layers=layers, ...)
→ ✅ 成功，不再报错

现在StaticCache应该能够正确初始化，满足新的transformers
Cache协议要求，同时保持所有自定义功能。

## Bug2: Qwen3MoeAttention.forward lack of position_ids

I inspected ktransformers/models/modeling_qwen3_moe.py and the important part
of your diff is the updated attention signature: Qwen3MoeAttention.forward now
accepts position_ids, output_attentions, use_cache, and \*\*kwargs (ktransformers/
models/modeling_qwen3_moe.py:213). With this change the call coming from
Qwen3MoeDecoderLayer.forward is satisfied, so the earlier TypeError will no
longer occur. The rest of the diff is purely formatting (import wrapping, line
reflows, etc.) and does not affect behaviour.

## Bug3: Model MLA judgement lack

1. 问题定位

- 错误位置: /opt/conda/lib/python3.11/site-packages/ktransformers/server/back
  end/interfaces/transformers.py:408
- 错误代码: 尝试访问 self.model.config.kv_lora_rank 属性
- 错误类型: AttributeError: 'Qwen3MoeConfig' object has no attribute
  'kv_lora_rank'

2. 问题根源

这个错误发生在 generate() 方法的 decode 循环中。当 flashinfer_enabled 为 True
时，代码会调用 MLAWrapperSingleton.plan_all() 方法，这个方法是为 DeepSeek
模型的 MLA（Multi-head Latent Attention）架构设计的，需要以下 DeepSeek
特有的配置参数：

- kv_lora_rank: DeepSeek MLA 的 KV 压缩维度
- qk_rope_head_dim: DeepSeek 的 RoPE 头维度

然而，Qwen3MoE 模型使用的是标准的注意力机制，不是 MLA 架构，因此：

- Qwen3MoeConfig 没有 kv_lora_rank 属性
- Qwen3MoeConfig 没有 qk_rope_head_dim 属性
- Qwen3MoE 不需要也不应该使用 MLAWrapperSingleton.plan_all()

3. 解决方案

- ktransformers/server/backend/interfaces/transformers.py:36 and 408 now gate
  FlashInfer MLA planning behind _supports_flashinfer_mla, logging once when a
  config (e.g. Qwen3Moe) lacks kv_lora_rank/qk_\*\_head_dim.
- ktransformers/local_chat.py:150 applies the same attribute check before
  enabling FlashInfer-specific kwargs, so the CLI won’t reference undefined
  config fields.
- ktransformers/local_chat_test.py:134 mirrors the guard used in production to
  keep the test harness aligned.

## Bug 4: Qwen3MoE 设备不匹配错误修复总结

错误: RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, cpu and cuda:0!

位置: ktransformers/operators/models.py:568 - self.embed_tokens(input_ids)
根本原因

- input_ids 在 decode 阶段位于 CUDA 设备
- model.embed_tokens 权重根据 YAML 配置在 CPU 设备
- KQwen3MoeModel.forward() 直接调用 embedding，未进行设备转换

修复方案

在 KQwen3MoeModel.forward() 中添加设备转换逻辑，与 Qwen2 保持一致。
文件: ktransformers/operators/models.py:567-574

    修改前:
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    修改后:
    if inputs_embeds is None:
        # The embedding weights stay on CPU, so stage ids on CPU and send the
    result back.
        input_ids_device = input_ids.device
        if input_ids_device.type != "cpu":
            input_ids = input_ids.to("cpu")
        inputs_embeds = self.embed_tokens(input_ids)
        if input_ids_device.type != "cpu":
            inputs_embeds = inputs_embeds.to(input_ids_device)

## Bug5: Decode 输出乱码

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

### Error again

修复未成功，可能是由于隐性错误导致很难修复，所以我们需要让程序给出更多信息，来帮助我们来定位错误
如果仍感觉有问题，可以获取一段简短的 logits 输出，以便我们再次检查解码后的专家权重。

若想查看驱动生成过程的 logits，单线程后端最直接的插入位置在接口循环中：

- ktransformers/server/backend/interfaces/transformers.py:302-324
  (decode_one_tokens) 已通过以下方式接收最后一步的 logits：

logits = self.model(self.current_ids, cache_position=...,
past_key_values=self.cache, return_dict=False, use_cache=True)[0]
logits = logits[0, -1, :]

- 你可以对这一块（以及预填充阶段中的类似调用，ktransformers/server/backend/interfaces/transformers.py:284-298）进行插桩，将张量暂存或记录日志。例如，在调用 self.logits_to_token 之前注册一个副作用：

torch.save(logits.cpu(), f"/tmp/logits_step{self.seq_length}.pt")

或把原始数值推入一个线程安全的队列/文件以供后续检查。

由于这些张量位于 GPU 上，请记得在序列化前将其移至 CPU。

其他实用方案：

1. 临时日志辅助工具
   在接口类中添加一个工具函数，当设置了环境变量（如 DUMP_LOGITS=1）时，将前几个 token 的 logits 写入磁盘。这能保持正常运行时的代码路径清晰，并按需提供样本。

2. 调试期间的内联打印
   如果仅需快速查看，可在 decode_one_tokens 中插入 print(logits[:10].tolist())。但请注意，每步都打印会显著拖慢解码循环，因此建议使用小计数器进行控制。

3. 结构化捕获用于回归测试
   对于更受控的运行场景（例如通过 python ktransformers/tests/test_speed.py …），可添加一个钩子，将 logits 添加到接口对象中存储的 Python 列表里，待 generate() 完成后统一导出。这样避免了逐 token 的文件 I/O，同时仍能完整保留分析所需的轨迹数据。

无论选择哪种方法，完成调试后请务必移除或保护相关插桩代码——持久化日志会影响吞吐量。
本地日志补集：

1. 在 ktransformers/server/backend/interfaces/transformers.py 的
   decode_one_tokens 中临时保存 logits（建议 torch.save(logits.cpu(), f"/tmp/
   logits_step{self.seq_length}.pt")，前 10 个 token 即可）。
2. 同时抓取每步被送入 KExpertsCPU.forward 的 expert_ids、weights（可在
   ktransformers/operators/experts.py:KQwen3MoeSparseMoeBlockV2.forward 中打印/
   保存）。
3. 重新运行测试命令并把首批 logits 张量、expert ids/weights 的样本，以及控制台关
   键信息贴上来。

这样我们就能核对 logits 是否已经正常、以及 CPU 专家权重是否仍然错位。
