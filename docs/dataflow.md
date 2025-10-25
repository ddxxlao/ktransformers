# Qwen3Moe Decode 阶段专家数据流

## 解码主链路

- `KQwen3MoeModel.forward` 会在解码阶段逐层推进解码器，并在循环中调用每一层的 `mlp`（`ktransformers/operators/models.py:603`）。
- 单层的 `Qwen3MoeDecoderLayer.forward` 在自注意力之后执行 `self.mlp(hidden_states)`（`ktransformers/models/modeling_qwen3_moe.py:477`），此处的 `mlp` 已被 YAML 注入为 `KQwen3MoeSparseMoeBlockV2`。
- `KQwen3MoeSparseMoeBlockV2.forward` 负责计算路由器 logits、选出 top‑k 专家，并在解码阶段通过 `self.experts(...)` 进入专家算子（`ktransformers/operators/experts.py:1643`）。
- `self.experts` 被注入为 `KTransformersExpertsV2`（`ktransformers/operators/experts.py:1305`），其内部根据 YAML 中的 `generate_device` 与 `generate_op` 选择具体的解码后端（`ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml:58-66`）。

## YAML 配置驱动的专家执行模式

| 配置组合 | 生成态实现 | 典型用途 |
| --- | --- | --- |
| `generate_device: "cpu"` + `generate_op: "KExpertsCPU"`（仓库默认） | 使用 CPU 后端并返回到 GPU（`KExpertsCPU`） | 单线程后端 + CPU 专家卸载 |
| `generate_device: "cuda"` + `generate_op: "KExpertsTorch"` | 全流程常驻 GPU（`KExpertsTorch`） | 纯 GPU 推理或 CUDA Graph 录制 |
| `generate_device: "cuda"` + `generate_op: "KExpertsMarlin"` | GPU Marlin 量化专家 | GPU 上的高效量化 MoE |

> 提示：`out_device` 默认为 `"cuda"`（`ktransformers/optimize/optimize_rules/Qwen3Moe-ktransformers.yaml:66`），确保专家输出总是返回至 GPU 参与后续层计算。

## CPU 生成路径（generate_device="cpu", generate_op="KExpertsCPU"）

### GPU→CPU→GPU 时序

| 步骤 | 代码位置 | 有效设备 | 说明 |
| --- | --- | --- | --- |
| 1 | `ktransformers/operators/models.py:603` | GPU | `KQwen3MoeModel` 将当前层的 `hidden_states` 送入 `decoder_layer.mlp`。 |
| 2 | `ktransformers/operators/experts.py:1651` | GPU | `KQwen3MoeSparseMoeBlockV2` 在 GPU 上执行路由器线性层与 softmax/top‑k。 |
| 3 | `ktransformers/operators/experts.py:1687` | GPU→接口 | `moe_on_cpuinfer` 触发 `self.experts(...)`，进入 `KTransformersExpertsV2.forward`。 |
| 4 | `ktransformers/operators/experts.py:1364` | GPU→CPU | `KTransformersExpertsV2` 在 `InferenceState.GENERATE` 下调用 `KExpertsCPU.forward`。 |
| 5 | `ktransformers/operators/experts.py:334-352` | CPU pinned | `KExpertsCPU.forward` 将 GPU 隐状态、出勤专家与权重拷贝到持久化的 CPU pin 内存缓存。缓存尺寸在 `KExpertsCPU.load` 中按 `Config().chunk_size` 预分配（`ktransformers/operators/experts.py:273-298`）。 |
| 6 | `ktransformers/operators/experts.py:338` + `ktransformers/operators/cpuinfer.py:742` | CPU | 通过 `CPUInfer.submit_with_cuda_stream` 把任务提交给 C++ `cpuinfer_ext` 后端执行专家矩阵乘。 |
| 7 | `ktransformers/operators/experts.py:340-353` | CPU→GPU | `CPUInfer.sync_with_cuda_stream` 回传完成信号，结果从 CPU 缓冲区拷贝至 `out_device`（默认为当前 GPU）。 |
| 8 | `ktransformers/operators/experts.py:1675-1704` | GPU | `KQwen3MoeSparseMoeBlockV2` 将专家输出 reshape 回 `(batch, seq, hidden)`，再交给后续残差与层归一化。 |

### CUDA Graph 单 token 快捷链路

- 当解码处于 CUDA Graph 捕获阶段时，`KQwen3MoeSparseMoeBlockV2.forward` 会调用专家的 `submit_for_one_decode`（`ktransformers/operators/experts.py:1670-1675`）。
- `KExpertsCPU.submit_for_one_decode` 将单 token 输入异步写入 pin 内存，并在当前 CUDA stream 上串联 `CPUInfer.submit_with_cuda_stream`（`ktransformers/operators/experts.py:300-314`）。
- 捕获完成后通过 `sync_for_one_decode` 回读结果（`ktransformers/operators/experts.py:317-325`），避免在 Graph 录制期间出现 CUDA->CPU->CUDA 的显式同步。

### 额外注意

- `bsz_tensor` 默认为 `None`，`KExpertsCPU.forward` 会即时构造长度为 1 的张量供 CPU 内核读取（`ktransformers/operators/experts.py:301-332`）。
- 调试可开启 `DUMP_QWEN3_EXPERTS`，`KExpertsCPU._maybe_dump_debug` 会在关键节点保留样本数据（`ktransformers/operators/experts.py:188-205`）。

## GPU 生成路径（generate_device="cuda", generate_op="KExpertsTorch"）

### 时序对比

| 步骤 | 代码位置 | 有效设备 | 说明 |
| --- | --- | --- | --- |
| 1-2 | 同上 | GPU | 路由器与 top‑k 仍在 GPU 执行。 |
| 3 | `ktransformers/operators/experts.py:1687` | GPU→接口 | 进入 `KTransformersExpertsV2.forward`。 |
| 4 | `ktransformers/operators/experts.py:1364` | GPU | `generate_device="cuda"` 时实例为 `KExpertsTorch`（`ktransformers/operators/experts.py:1326`）。 |
| 5 | `ktransformers/operators/experts.py:652-688` | GPU | `KExpertsTorch.forward` 在 GPU 上执行三段线性层与激活，不做跨设备拷贝。 |
| 6 | `ktransformers/operators/experts.py:1687-1704` | GPU | 输出直接返回上一层，链路不离开 GPU。 |

### 切换配置

在规则文件中改写 `generate_device` 与 `generate_op` 即可切换到纯 GPU 专家，例如：

```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExpertsV2
    kwargs:
      generate_device: "cuda"
      generate_op: "KExpertsTorch"
      prefill_device: "cuda"
      prefill_op: "KExpertsTorch"
      out_device: "cuda"
  recursive: False
```

切换至 `KExpertsMarlin` 仅需把 `generate_op` 改为 `"KExpertsMarlin"`，对应实现位于 `ktransformers/operators/experts.py:372-520`。

## 关键信息速查

- `KTransformersExpertsV2.load()` 默认以 `InferenceState.GENERATE` 预热生成态专家（`ktransformers/operators/experts.py:1337-1355`），解码开始即可直接调用。
- `KQwen3MoeModel.load_layer_to` 在需要时调度 `set_inference_mode`，允许后续扩展 CPU/GPU 分层卸载（`ktransformers/operators/models.py:693-727`）。
- CPU 专家输出通过 `out_device` 控制回流位置，若需把结果保留在 CPU，可在规则中调整该字段并同步更改后续层的 `generate_device`。
