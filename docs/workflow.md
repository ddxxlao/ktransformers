# Balance Serve 推理链路

## 1. 模型注册与初始化链路（以 Qwen3MoE 为例）

### 1.1 服务入口：加载全局配置并解析 CLI
- `main()` 先实例化全局单例配置 `Config()`，随后构造 `ArgumentParser` 以当前配置为默认值并解析命令行参数（`ktransformers/server/main.py:103`）。

- `ArgumentParser.parse_args()` 会将 CLI 与 `config.yaml` 中的默认值融合：校准模型路径、设备、最大 batch、KV cache 等配置，并根据实际模型重新计算 `args.architectures` / `args.gpu_memory_size`（`ktransformers/server/args.py:144`、`ktransformers/server/args.py:155`）。

- `parse_args()` 还会预留 Balance Serve 需要的内部端口（调度和监控端口），并把所有解析结果写回单例 `Config()`，确保后续模块（含子进程）可直接读取统一的运行参数（`ktransformers/server/args.py:161`–`ktransformers/server/args.py:171`）。

### 1.2 选择后端并构造推理接口
- 完成参数解析后，`create_interface()` 通过 `config.backend_type` 选择具体后端实现。当值为 `balance_serve` 时，实例化 `BalanceServeInterface` 并将其挂载到 `GlobalInterface`，同时初始化线程上下文管理器（`ktransformers/server/utils/create_interface.py:19`–`ktransformers/server/utils/create_interface.py:31`）。

- 此时传入的 `default_args` 本质上是刚刚更新过的全局 `Config()`，因此 `BalanceServeInterface` 能直接访问 CLI/配置文件里与 Qwen3MoE 相关的所有超参（模型目录、GGUF 路径、page size 等）。

### 1.3 BalanceServeInterface 初始化：父进程组件注册
- 初始化阶段首先建立推理输出的跨进程通信结构：为每个请求准备 `asyncio.Queue` 映射，并创建一个 `multiprocessing.Queue` 用于回收 token（`ktransformers/server/backend/interfaces/balance_serve.py:332`–`ktransformers/server/backend/interfaces/balance_serve.py:349`）。

- 立即加载与模型配套的 `AutoTokenizer`，以保证后续请求能够同步分词规则（`ktransformers/server/backend/interfaces/balance_serve.py:340`）。

- 启动 Balance Serve 的核心推理子进程：利用 `multiprocessing.get_context("spawn")` 拉起 `run_engine(...)`，并通过事件 `kvcache_event` 与父进程同步模型加载完成状态（`ktransformers/server/backend/interfaces/balance_serve.py:347`–`ktransformers/server/backend/interfaces/balance_serve.py:350`）。

- 为负载调度与 KV cache 管理启动独立的 RPC 进程：将当前 `args` 序列化后传给 `sched_rpc.py`，该进程会根据模型架构选择对应的调度配置生成函数（`ktransformers/server/backend/interfaces/balance_serve.py:353`–`ktransformers/server/backend/interfaces/balance_serve.py:366`、`ktransformers/server/balance_serve/sched_rpc.py:206`–`ktransformers/server/balance_serve/sched_rpc.py:224`）。

- `BalanceServeInterface` 还会在 FastAPI 生命周期里注册 `queue_proxy`，负责把子进程返回的 token 推送到每个请求的 `asyncio.Queue` 中（`ktransformers/server/backend/interfaces/balance_serve.py:417`–`ktransformers/server/backend/interfaces/balance_serve.py:438`）。

### 1.4 子进程 Engine：模型与缓存的实际装载
- `run_engine()` 内部实例化 `Engine`，并在需要时执行 CUDA Graph 预热（`ktransformers/server/backend/interfaces/balance_serve.py:307`–`ktransformers/server/backend/interfaces/balance_serve.py:313`）。

- `Engine.__init__()` 将父进程解析好的各项参数重新写入 `Config()` 单例，以便子进程也能访问统一配置（`ktransformers/server/backend/interfaces/balance_serve.py:123`–`ktransformers/server/backend/interfaces/balance_serve.py:129`）。

- 针对 Qwen3MoE，Engine 会专用地加载 `Qwen3MoeConfig`，并在 `torch.device("meta")` 环境下构造 `KGQACache` 与定制模型 `KQwen3MoeForCausalLM`，避免显存占用并与 Balance Serve 的批次调度兼容（`ktransformers/server/backend/interfaces/balance_serve.py:137`–`ktransformers/server/backend/interfaces/balance_serve.py:179`）。

- 如果模型目录中提供了 `generation_config.json`，会作为推理默认值加载；否则按照 CLI 参数临时生成一个采样配置（`ktransformers/server/backend/interfaces/balance_serve.py:191`–`ktransformers/server/backend/interfaces/balance_serve.py:199`）。

### 1.5 GGUF 注入与算子初始化
- Balance Serve 为常见模型预置了优化规则映射，Qwen3MoE 会自动定位到 `optimize/optimize_rules/Qwen3Moe-serve.yaml`（`ktransformers/server/backend/interfaces/balance_serve.py:201`–`ktransformers/server/backend/interfaces/balance_serve.py:204`）。
- 随后调用 `optimize_and_load_gguf()`：该函数读取优化规则并遍历模型 Module 树，把 GGUF 中的权重与替换算子注入到 `KQwen3MoeForCausalLM`，包括 FlashInfer 相关的自定义内核绑定（`ktransformers/optimize/optimize.py:117`–`ktransformers/optimize/optimize.py:133`）。
- 当模型缺省 `pad_token_id` 时，会在这里与 `eos_token_id` 对齐，确保推理时序列对齐行为正确（`ktransformers/server/backend/interfaces/balance_serve.py:213`–`ktransformers/server/backend/interfaces/balance_serve.py:215`）。

### 1.6 KV Cache 引导与调度器同步
- GGUF 权重加载完成后，子进程向调度器请求初始的 `InferenceContext`，用于恢复或初始化 KV Cache 的 GPU 页布局，并最终调用 `self.cache.load(...)` 完成缓存注册（`ktransformers/server/backend/interfaces/balance_serve.py:220`–`ktransformers/server/backend/interfaces/balance_serve.py:229`）。
- Engine 随后构造用于批量执行的 `ModelRunner`、`Sampler`、`QueryManager`，这些组件将被主循环用于预填充与 Decode 调度（`ktransformers/server/backend/interfaces/balance_serve.py:229`–`ktransformers/server/backend/interfaces/balance_serve.py:237`）。
- 与此同时，调度 RPC 进程会针对 Qwen3MoE 调整 head 维度、KV cache 策略等参数（`ktransformers/server/balance_serve/settings.py:126`–`ktransformers/server/balance_serve/settings.py:176`），确保推理子进程与调度器对模型结构的认知一致。

通过以上链路，选择 `balance_serve` 后端时，Qwen3MoE 模型会被注册到全局接口、完成 GGUF 注入、KV cache 初始化以及调度器同步，为后续的 Prefill/Decode 工作循环奠定基础。

## 2. GGUF 注入与算子替换流程详解（Qwen3MoE + balance_server）

### 2.1 入口：确定优化规则与 GGUF 目录
- Balance Serve 子进程在构造 `Engine` 时优先读取 CLI/配置中传入的 `--optimize-config-path`，缺省则回落到 `default_optimize_rules["Qwen3MoeForCausalLM"]` 指向的 `optimize/optimize_rules/Qwen3Moe-serve.yaml`（`ktransformers/server/backend/interfaces/balance_serve.py:201`–`ktransformers/server/backend/interfaces/balance_serve.py:205`）。
- 同一阶段会解析 `--gguf-path`，若未提供则阻塞等待人工输入，确保后续权重加载指向正确的 GGUF 文件夹（`ktransformers/server/backend/interfaces/balance_serve.py:206`–`ktransformers/server/backend/interfaces/balance_serve.py:211`）。

### 2.2 生成优化映射：`gen_optimize_config`
- `optimize_and_load_gguf()` 先读取 YAML 规则，调用 `gen_optimize_config()` 深度遍历 `KQwen3MoeForCausalLM` 的 Module 树，为每个匹配项生成 `{key, class, kwargs}` 记录并写入 `optimize_config`，未命中的模块会以 `"class": "default"` 挂载默认加载行为（`ktransformers/optimize/optimize.py:117`–`ktransformers/optimize/optimize.py:137`、`ktransformers/optimize/optimize.py:55`–`ktransformers/optimize/optimize.py:107`）。
- 匹配逻辑同时支持 `match.name` 的正则与 `match.class` 的精确类型判断，确保既能按照层号（`model.layers.<idx>`）定位，又能避免 `shared_expert_gate` 等特殊线性层被误替换（`ktransformers/optimize/optimize.py:61`–`ktransformers/optimize/optimize.py:88`）。
- 每个匹配到的模块都会把 YAML 中声明的 `kwargs` 预写入 `gguf_loader.tensor_device_map`，后续权重加载可依据 `prefill_device` / `generate_device` 决定具体搬运策略。

### 2.3 注入阶段：`inject`
- `inject()` 根据 `optimize_config` 递归替换子模块：当配置项的 `class` 不是 `"default"` 时，会通过反射实例化指定的算子类，并用 `set_module()` 把原模块替换为新的 `BaseInjectedModule` 子类；若 `class` 为 `"default"`，则仅保留原模块但仍记录设备信息（`ktransformers/optimize/optimize.py:20`–`ktransformers/optimize/optimize.py:43`、`ktransformers/util/utils.py:66`–`ktransformers/util/utils.py:78`）。
- 新模块持有 `orig_module` 引用，可在必要时复用原生实现；所有注入算子默认以 `torch.device("meta")` 实例化，避免替换阶段加载实际权重造成显存浪费（`ktransformers/optimize/optimize.py:127`–`ktransformers/optimize/optimize.py:129`）。
- 注入完成后执行 `del_meta()` 删除 meta 参数占位，保证之后的权重写入不会因残留的虚拟张量冲突（`ktransformers/optimize/optimize.py:44`–`ktransformers/optimize/optimize.py:54`、`ktransformers/optimize/optimize.py:133`）。

### 2.4 关键替换点（Qwen3Moe YAML 明细）
- **RotaryEmbedding**：`Qwen2MoeRotaryEmbedding` 被 `operators.RoPE.RotaryEmbedding` 替换，统一旋转位置编码的预填充/生成接口（`optimize/optimize_rules/Qwen3Moe-serve.yaml:1`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:7`）。
- **线性层内核**：`model.layers.*` 下所有非 `shared_expert_gate` 的 `torch.nn.Linear` 注入 `KTransformersLinear`，Prefill 阶段使用标准 Torch Kernel，生成阶段切换到 Marlin Kernel 以利用 FlashInfer 执行计划（`optimize/optimize_rules/Qwen3Moe-serve.yaml:31`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:39`，实现参见 `ktransformers/operators/linear.py:147`–`ktransformers/operators/linear.py:199`）。
- **自注意力**：整层 `self_attn` 替换成 `KQwen3MoeAttention`，内部依赖 `KGQACache` 的分页存储与 FlashInfer wrapper 做 Prefill/Decode 路径拆分（`optimize/optimize_rules/Qwen3Moe-serve.yaml:61`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:67`，实现见 `ktransformers/operators/balance_serve_attention.py:206`–`ktransformers/operators/balance_serve_attention.py:280`）。
- **MoE 框架**：`Qwen3MoeSparseMoeBlock` 与其 `experts` 子模块分别替换为 `KQwen3MoeSparseMoeBlockV2`、`KTransformersExpertsV2`，在 Prefill / Decode 间动态切换 GPU / CPU 执行器，同时支持 CUDA Graph 捕获（`optimize/optimize_rules/Qwen3Moe-serve.yaml:41`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:58`、`ktransformers/operators/experts.py:1243`–`ktransformers/operators/experts.py:1315`、`ktransformers/operators/experts.py:1581`–`ktransformers/operators/experts.py:1643`）。
- **归一化**：`Qwen3MoeRMSNorm` 被 `KQwen3MoeRMSNorm` 注入，为 Balance Serve 的分块 Batch 尺度提供 `fused_add_rmsnorm` 支持（`optimize/optimize_rules/Qwen3Moe-serve.yaml:81`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:87`、`ktransformers/operators/layernorm.py:128`–`ktransformers/operators/layernorm.py:160`）。
- **模型容器**：最上层 `model` 重定向到 `KQwen2MoeModel`，该容器负责跟踪跨设备流、执行计划和 CUDA Graph 状态，是调度器切换 Prefill/Decode 的对接点（`optimize/optimize_rules/Qwen3Moe-serve.yaml:68`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:73`、`ktransformers/operators/models.py:175`–`ktransformers/operators/models.py:200`）。
- **Embedding 与 LM Head**：`embed_tokens` 标记为默认替换并强制加载到 CPU，配合 MoE 规则在 Prefill 时避免重复搬运；`lm_head` 换成 `KTransformersLinear` 并提前单独加载，降低大矩阵权重在后续 `load_weights()` 阶段的峰值显存（`optimize/optimize_rules/Qwen3Moe-serve.yaml:9`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:19`、`optimize/optimize_rules/Qwen3Moe-serve.yaml:74`–`optimize/optimize_rules/Qwen3Moe-serve.yaml:79`）。

### 2.5 权重加载与设备调度
- `ModelLoaderFactory.create_loader()` 会根据目录中文件类型自动选择 `GGUFLoader`；加载时 `load_weights()` 遇到 `BaseInjectedModule` 会调用其 `load()` 方法，触发自定义的量化权重反量化或 CPU/GPU 双端缓存逻辑（`ktransformers/util/custom_loader.py:506`–`ktransformers/util/custom_loader.py:575`、`ktransformers/util/utils.py:169`–`ktransformers/util/utils.py:177`）。
- `GGUFLoader.tensor_device_map` 中的 `generate_device` / `prefill_device` 在 `load_cur_state_dict()` 阶段被读取，决定每块权重实际落在哪个 CUDA 设备或 CPU 上，避免整模一次性搬运到 GPU（`ktransformers/util/utils.py:120`–`ktransformers/util/utils.py:139`）。
- 某些算子会在 `load()` 内部根据推理态切换权重来源，例如 `KTransformersExpertsV2.load()` 会在 Prefill 时启用 GPU Experts，在 Decode 时卸载 GPU 数据并加载 CPU 版权重（`ktransformers/operators/experts.py:1243`–`ktransformers/operators/experts.py:1315`）。

### 2.6 与 Balance Serve 执行循环的衔接
- 注入完成后的模型由 `Engine` 交给 `ModelRunner` 管理，后者根据调度器下发的 Prefill/Decode 批次在不同 CUDA Graph 上调用已替换的算子，从而获取 FlashInfer 批量计划和 CPU/GPU 专家执行结果（`ktransformers/server/backend/interfaces/balance_serve.py:228`–`ktransformers/server/backend/interfaces/balance_serve.py:237`）。
- 由于所有关键子模块都继承自 `BaseInjectedModule` 并保留对 `orig_module` 的引用，必要时可以回退到原生实现（例如 Debug 或禁用 FlashInfer 时），保持与 Hugging Face 模型的接口兼容性。

## 3. KV Cache 与 cache（Balance Serve · Qwen3MoE）

### 3.1 调度器准备阶段
- `BalanceServeInterface.__init__()` 先创建推理子进程并阻塞等待 `kvcache_event`，随后序列化 CLI 配置启动调度 RPC 进程，保证在 KVCache 准备完成前不会接收请求（`ktransformers/server/backend/interfaces/balance_serve.py:332`–`ktransformers/server/backend/interfaces/balance_serve.py:366`）。
- 调度进程根据模型架构调用 `create_sched_settings_qwen3moe()`，为 Qwen3MoE 设定分页大小、最大 batch、GPU/CPU 内存池与 `k_cache_on`、`v_cache_on` 标志，确保调度器实际分配 K/V 缓存页（`ktransformers/server/balance_serve/settings.py:126`–`ktransformers/server/balance_serve/settings.py:176`）。
- `SchedulerServer.worker_routine()` 暴露 `get_inference_context` RPC，调用底层 `sched_ext` 拿到当前缓存视图，并通过 `mp.reductions.reduce_tensor` 将 GPU 张量句柄序列化，供父进程复现（`ktransformers/server/balance_serve/sched_rpc.py:92`–`ktransformers/server/balance_serve/sched_rpc.py:106`）。

### 3.2 Engine 加载远程 KVCache
- 子进程 `Engine.__init__()` 在 GGUF 注入完成后通过 `SchedulerClient.get_inference_context_raw()` 拉取缓存快照，并利用 `rebuild_inferece_context()` 把句柄转回 `sched_ext.InferenceContext`（`ktransformers/server/backend/interfaces/balance_serve.py:217`–`ktransformers/server/backend/interfaces/balance_serve.py:225`、`ktransformers/server/balance_serve/sched_rpc.py:192`–`ktransformers/server/balance_serve/sched_rpc.py:198`）。
- `KGQACache.load()` 逐层引用调度器返回的 K/V 块，并计算 `max_cache_len`（页数 × 每页 token），让模型可以直接在共享显存上读写（`ktransformers/models/custom_cache.py:296`–`ktransformers/models/custom_cache.py:307`）。
- Engine 初始化 `ModelRunner` 时以调度器返回的块数配置 FlashInfer wrapper，保证后续 Prefill/Decode 批次按同一分页布局复用缓存页（`ktransformers/server/backend/interfaces/balance_serve.py:228`–`ktransformers/server/backend/interfaces/balance_serve.py:234`）。

### 3.3 请求生命周期中的缓存寻址
- 调度器回传的批次通过 `QueryManager.add_query()` 写入本地状态，包括初始 `block_index`、prefill token 序列和终止条件；这些索引直接对应调度器持有的 KV 页号（`ktransformers/server/balance_serve/inference/query_manager.py:82`–`ktransformers/server/balance_serve/inference/query_manager.py:103`）。
- `ForwardBatchInput.ForwardMiniBatch` 将 QueryManager 里的 `block_index`、当前位置和 page size 重新拼装成 `kv_indptr` / `kv_indices`，把 Prefill 段与 Decode 段统一映射到分页缓存（`ktransformers/server/balance_serve/inference/forward_batch.py:31`–`ktransformers/server/balance_serve/inference/forward_batch.py:66`、`ktransformers/server/balance_serve/inference/forward_batch.py:129`–`ktransformers/server/balance_serve/inference/forward_batch.py:161`）。
- 执行阶段 `ModelRunner` 使用 `KGQACache.get_page_table()` 把 `position_ids` 转换为具体页索引和偏移量，再将结果传给 FlashInfer wrapper 以读取已填充的 K/V（`ktransformers/server/balance_serve/inference/model_runner.py:138`、`ktransformers/server/balance_serve/inference/model_runner.py:207`、`ktransformers/models/custom_cache.py:311`–`ktransformers/models/custom_cache.py:327`）。
- 定制模型 `KQwen3MoeForCausalLM.forward()` 把映射好的 `page_idx`、`page_offset` 传入自定义注意力实现，直接依赖共享缓存完成 Prefill/Decode 访存；输出 logits 后由 `fill_generated_tokens()` 写回 QueryManager，循环更新 active position 与生成 token（`ktransformers/models/custom_modeling_qwen3_moe.py:101`–`ktransformers/models/custom_modeling_qwen3_moe.py:114`、`ktransformers/server/backend/interfaces/balance_serve.py:93`–`ktransformers/server/backend/interfaces/balance_serve.py:105`、`ktransformers/server/balance_serve/inference/query_manager.py:105`–`ktransformers/server/balance_serve/inference/query_manager.py:154`）。


## 4. Token 流程（Prefill → Decode）

### 4.1 请求入队：BalanceServeInterface 父进程
- `BalanceServeInterface.inference()` 先将对话消息转换为 chat template，再编码成张量；若开启强制思维模式，会在 prompt 末尾补充 `<think>` token（`ktransformers/server/backend/interfaces/balance_serve.py:452`–`ktransformers/server/backend/interfaces/balance_serve.py:467`）。
- 该方法构造 `sched_ext.QueryAdd`，写入完整的 prompt token、长度、停止词与采样参数，并根据 `cache_lens` 限制估算最大序列长度（`ktransformers/server/backend/interfaces/balance_serve.py:473`–`ktransformers/server/backend/interfaces/balance_serve.py:488`）。
- 通过 `SchedulerClient.add_query()` 将请求提交给调度器，随后为返回的 `query_id` 注册每路 `asyncio.Queue`，并把流式生成逻辑绑定到 `chat_stream()`（`ktransformers/server/backend/interfaces/balance_serve.py:491`–`ktransformers/server/backend/interfaces/balance_serve.py:509`、`ktransformers/server/backend/interfaces/balance_serve.py:75`–`ktransformers/server/backend/interfaces/balance_serve.py:90`）。
- 首个 token 到达时视为 prefill 结束：暂停预填计时器、启动 decode 计时并动态统计吞吐；生成终止后依次发送 `streamer.end()`、终止原因与 `RawUsage` 统计（`ktransformers/server/backend/interfaces/balance_serve.py:497`–`ktransformers/server/backend/interfaces/balance_serve.py:520`）。

### 4.2 调度握手：Scheduler RPC 与 QueryManager
- Scheduler RPC 进程接收 `add_query` 请求后分配全局自增 ID，并在 `update_last_batch` 时返回下一批待执行的 `BatchQueryTodo`（`ktransformers/server/balance_serve/sched_rpc.py:66`–`ktransformers/server/balance_serve/sched_rpc.py:90`、`ktransformers/server/balance_serve/sched_rpc.py:187`–`ktransformers/server/balance_serve/sched_rpc.py:190`）。
- Engine 子进程通过同一个 `SchedulerClient` 拉取推理上下文、KV Cache 快照，与调度批次，从而和父进程共享一致的缓存状态（`ktransformers/server/backend/interfaces/balance_serve.py:220`–`ktransformers/server/backend/interfaces/balance_serve.py:237`）。
- `QueryManager.add_query()` 将调度下发的 prompt token、块索引和停止条件写入 GPU 常驻结构，并把 prefill mini-batch 的起始位置同步到 active cursor（`ktransformers/server/balance_serve/inference/query_manager.py:82`–`ktransformers/server/balance_serve/inference/query_manager.py:103`）。
- 每个 `update_last_batch` 返回前，`QueryManager.update()` 会根据 prefill/decoder 的进展更新光标、切换 `is_prefill` 标记，并依据停用词或长度约束设置 `decode_done`（`ktransformers/server/balance_serve/inference/query_manager.py:105`–`ktransformers/server/balance_serve/inference/query_manager.py:153`）。

### 4.3 Prefill 批次组装与执行
- `BatchQueryTodo.prefill_mini_batches` 记录形如 `(query_id, start, length)` 的片段，Engine 在进入主循环时先把上一轮的更新结果回传调度，再申请下一批任务（`ktransformers/server/backend/interfaces/balance_serve.py:262`–`ktransformers/server/backend/interfaces/balance_serve.py:283`）。
- `ForwardBatchInput.ForwardMiniBatch` 会将多个 prefill 片段拼接成一次性大张量，计算 `q_indptr`、`kv_indptr`、`position_ids` 等索引，以便模型能在 FlashInfer kernel 中一次处理所有 token（`ktransformers/server/balance_serve/inference/forward_batch.py:31`–`ktransformers/server/balance_serve/inference/forward_batch.py:66`）。
- 对于 decode 阶段已经激活的请求，同一个结构会附加一 token 的解码任务，并为每条请求保留采样温度、top_p，以保持与父进程参数一致（`ktransformers/server/balance_serve/inference/forward_batch.py:67`–`ktransformers/server/balance_serve/inference/forward_batch.py:145`）。
- `ModelRunner.run()` 依据 mini-batch 总 token 数选择合适的 CUDA Graph，准备批量 embedding、KV page table，并触发一次模型前向；prefill 子批和 decode 子批共享一次 GPU 计算（`ktransformers/server/balance_serve/inference/model_runner.py:164`–`ktransformers/server/balance_serve/inference/model_runner.py:227`）。

### 4.4 Decode 采样与流式回传
- 前向输出经由 `Sampler` 逐样本抽样得到新 token，并通过 `fill_generated_tokens()` 写回 `QueryUpdate.generated_token` 与缓存副本，确保下一轮 decode 能访问最新上下文（`ktransformers/server/backend/interfaces/balance_serve.py:240`–`ktransformers/server/backend/interfaces/balance_serve.py:295`、`ktransformers/server/backend/interfaces/balance_serve.py:93`–`ktransformers/server/backend/interfaces/balance_serve.py:101`）。
- Engine 调用 `self.gen_queue.put((query_id, token))` 将 decode 结果塞入父进程共享队列，并在检测到 `decode_done` 时插入 `None` 作为终止符号（`ktransformers/server/backend/interfaces/balance_serve.py:262`–`ktransformers/server/backend/interfaces/balance_serve.py:275`）。
- 父进程的 `queue_proxy()` 持续从多进程队列取出 `(query_id, token)`，再异步推入对应请求的 `asyncio.Queue`，防止 HTTP 线程阻塞（`ktransformers/server/backend/interfaces/balance_serve.py:417`–`ktransformers/server/backend/interfaces/balance_serve.py:434`）。
- `chat_stream()` 将 token 流交给 `TextStreamer` 解码为字符串；收到 `None` 时触发 `streamer.end()` 并退出循环，随后 `inference()` 通过 FastAPI StreamingResponse 将文本增量推送给客户端（`ktransformers/server/backend/interfaces/balance_serve.py:75`–`ktransformers/server/backend/interfaces/balance_serve.py:90`、`ktransformers/server/backend/interfaces/balance_serve.py:509`–`ktransformers/server/backend/interfaces/balance_serve.py:520`）。

## 5. 结束清退与日志工具（balance_serve + Qwen3MoE）

### 5.1 请求收尾：状态同步与使用统计
- `ThreadContext.work()` 在生成流结束后，根据运行状态触发 `RunObject` 与消息的完成/取消事件，并立即同步数据库，确保同一线程的后续请求看到最新状态（`ktransformers/server/backend/base.py:146`–`ktransformers/server/backend/base.py:167`）。
- Balance Serve 推理协程会在最后一次 token 推送后调用 `self.streamer.end()` 并输出结束符，再判断是否触发 `length`/`stop` 型 finish_reason，保持和 OpenAI Stream 协议一致（`ktransformers/server/backend/interfaces/balance_serve.py:496`–`ktransformers/server/backend/interfaces/balance_serve.py:516`）。
- 同一个协程随后打包 `RawUsage`，携带 tokenize/prefill/decode 的耗时与 token 数，供上层 API 汇总或落库（`ktransformers/server/backend/interfaces/balance_serve.py:518`–`ktransformers/server/backend/interfaces/balance_serve.py:524`、`ktransformers/server/schemas/endpoints/chat.py:95`–`ktransformers/server/schemas/endpoints/chat.py:100`）。

### 5.2 性能日志：Profiler + 统一 logger
- Balance Serve 为每次请求创建新的 `Profiler`，细分三个阶段的计时器和计数器，在首个 token 返回时切换 prefill→decode 计时，防止 Prefill 时间被重复计算（`ktransformers/server/backend/interfaces/balance_serve.py:454`–`ktransformers/server/backend/interfaces/balance_serve.py:507`、`ktransformers/server/utils/multi_timer.py:1`–`ktransformers/server/utils/multi_timer.py:75`）。
- `report_last_time_performance()` 将上述计数转换成 TPS 指标，并通过全局 `logger` 输出 info 日志；Qwen3MoE 线上定位性能回退时，可以用这一条日志快速对比 Prefill/Decode 的速度（`ktransformers/server/backend/interfaces/balance_serve.py:103`–`ktransformers/server/backend/interfaces/balance_serve.py:112`）。
- `logger` 本身由 `DailyRotatingFileHandler` 驱动，按天切分文件并保留配置数量的历史副本，写入 `~/.ktransformers/<log.dir>/<log.file>`；同时也输出带颜色的控制台日志便于实时观测（`ktransformers/server/config/log.py:145`–`ktransformers/server/config/log.py:173`、`ktransformers/server/config/config.py:46`–`ktransformers/server/config/config.py:84`）。

### 5.3 子进程与调度器的清退
- `BalanceServeInterface` 在启动推理子进程与调度器 RPC 进程后注册 SIGINT/SIGTERM 处理器，捕获退出信号时依次终止 `run_engine` 子进程和 `sched_rpc` 进程，避免残留 GPU 上下文占用（`ktransformers/server/backend/interfaces/balance_serve.py:347`–`ktransformers/server/backend/interfaces/balance_serve.py:385`）。
- 调度端本身支持优雅关闭：`SchedulerServer.stop_rpc_service()` 会先停掉调度循环，再关闭 ZeroMQ 前后端 socket 并释放上下文资源，防止端口悬挂（`ktransformers/server/balance_serve/sched_rpc.py:118`–`ktransformers/server/balance_serve/sched_rpc.py:142`）。
- 每个 `SchedulerClient` 在对象销毁时主动关闭 REQ socket 与 ZMQ context，确保 Balance Serve 父进程重启或热更新时不会泄漏文件描述符（`ktransformers/server/balance_serve/sched_rpc.py:150`–`ktransformers/server/balance_serve/sched_rpc.py:162`）。

### 5.4 日志落盘与外部进程监控
- 启动调度器时会把标准输出来向 `args.log_dir` 追加 `rpc.log`，用于排查 Qwen3MoE 的批调度异常或跨进程通信断链问题，文件路径与其他日志保持一致（`ktransformers/server/backend/interfaces/balance_serve.py:353`–`ktransformers/server/backend/interfaces/balance_serve.py:364`）。
- 主日志目录在首次运行时由 `Config()` 自动创建；Balance Serve 的性能日志与 RPC 侧日志共享这一目录，便于统一采集或备份（`ktransformers/server/config/config.py:58`–`ktransformers/server/config/config.py:84`）。
