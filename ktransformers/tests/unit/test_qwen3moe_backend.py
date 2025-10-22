import types

import os
import sys
import types
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HOME", str(WORKSPACE_ROOT))

_original_expanduser = os.path.expanduser


def _fake_expanduser(path):
    if path.startswith("~"):
        suffix = path[1:]
        if suffix.startswith(os.sep):
            suffix = suffix[len(os.sep):]
        return str(WORKSPACE_ROOT / suffix) if suffix else str(WORKSPACE_ROOT)
    return _original_expanduser(path)


os.path.expanduser = _fake_expanduser
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# 清理上一次导入的 site-packages 版本，确保使用当前工作区代码
for name in list(sys.modules):
    if name.startswith("ktransformers"):
        del sys.modules[name]

import pytest
import torch

# 提前注入 flashinfer 桩模块，避免真实包的 CUDA/日志依赖破坏单元测试
if "flashinfer" not in sys.modules:
    flashinfer_stub = types.ModuleType("flashinfer")

    class _DummyBatchMLAWrapper:
        def __init__(self, *_, **__):
            pass

        def plan(self, *_, **__):
            return None

        def decode(self, *_, **__):
            return None

        def capture(self, *_, **__):
            return None

        def replay(self, *_, **__):
            return None

    mla_module = types.ModuleType("flashinfer.mla")
    mla_module.BatchMLAPagedAttentionWrapper = _DummyBatchMLAWrapper

    sys.modules["flashinfer"] = flashinfer_stub
    sys.modules["flashinfer.mla"] = mla_module
    flashinfer_stub.mla = mla_module

prefill_module_name = "ktransformers.operators.flashinfer_batch_prefill_wrapper"
if prefill_module_name not in sys.modules:
    prefill_stub = types.ModuleType(prefill_module_name)

    class _DummyFlashInferAttn:
        def __init__(self, *_, **__):
            pass

        def plan(self, *_, **__):
            return None

        def calc_batch_indices(self, *_, **__):
            return None

        def forward(self, *_, **__):
            return None

    prefill_stub.flashInferAttn = _DummyFlashInferAttn
    sys.modules[prefill_module_name] = prefill_stub

from ktransformers.local_chat import KQwen3MoeForCausalLMStatic
from ktransformers.local_chat import custom_models as runtime_custom_models
from ktransformers.local_chat import default_optimize_rules as runtime_default_rules
from ktransformers.local_chat_test import custom_models as test_custom_models
from ktransformers.local_chat_test import default_optimize_rules as test_default_rules
from ktransformers.server.backend.interfaces import ktransformers as ktransformers_if
from ktransformers.server.backend.interfaces.ktransformers import KTransformersInterface
from ktransformers.util.utils import InferenceState
import ktransformers.operators.models as operators_models


@pytest.fixture(autouse=True)
def _no_cuda_empty_cache(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    return


def test_local_chat_registers_qwen3moe_static(tmp_path, monkeypatch):
    fake_path = str(tmp_path / "qwen3.yaml")
    monkeypatch.setitem(runtime_default_rules, "Qwen3MoeForCausalLM", fake_path)
    assert runtime_custom_models["Qwen3MoeForCausalLM"] is KQwen3MoeForCausalLMStatic
    assert runtime_default_rules["Qwen3MoeForCausalLM"] == fake_path


def test_local_chat_test_mirror_registration(tmp_path, monkeypatch):
    fake_path = str(tmp_path / "qwen3.yaml")
    monkeypatch.setitem(test_default_rules, "Qwen3MoeForCausalLM", fake_path)
    assert test_custom_models["Qwen3MoeForCausalLM"] is KQwen3MoeForCausalLMStatic
    assert test_default_rules["Qwen3MoeForCausalLM"] == fake_path


def test_interface_sets_flash_attn(monkeypatch, tmp_path):
    class DummyTokenizer:
        def decode(self, tokens, **_):
            return ""

    class DummyGenerationConfig:
        pad_token_id = None
        eos_token_id = 0

    class DummyGGUFLoader:
        tensor_device_map = {}

    class DummyModel:
        def __init__(self, config):
            self.config = config
            self.dtype = torch.float16
            self.gguf_loader = DummyGGUFLoader()

    class DummyConfig:
        architectures = ["Qwen3MoeForCausalLM"]
        torch_dtype = torch.float16
        max_position_embeddings = 16
        num_attention_heads = 2
        num_key_value_heads = 2
        num_hidden_layers = 1
        hidden_size = 8
        head_dim = 4
        output_attentions = False
        output_hidden_states = False
        output_router_logits = False
        use_cache = False
        use_return_dict = True

    def fake_auto_config(*_, **__):
        return DummyConfig()

    def fake_auto_tokenizer(*_, **__):
        return DummyTokenizer()

    def fake_generation_config(*_, **__):
        return DummyGenerationConfig()

    dummy_yaml = tmp_path / "rule.yaml"
    dummy_yaml.write_text("rules: []")

    dummy_args = types.SimpleNamespace(
        model_dir="dummy",
        device="cuda",
        trust_remote_code=True,
        model_name="Qwen3MoeForCausalLM",
        optimize_config_path=None,
        gguf_path=str(tmp_path / "weights.gguf"),
        batch_size=1,
        cache_lens=8,
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.9,
        use_cuda_graph=False,
    )

    monkeypatch.setattr(
        ktransformers_if, "custom_models", {"Qwen3MoeForCausalLM": DummyModel}
    )
    monkeypatch.setattr(
        ktransformers_if,
        "default_optimize_rules",
        {"Qwen3MoeForCausalLM": str(dummy_yaml)},
    )
    monkeypatch.setattr(
        ktransformers_if.AutoConfig, "from_pretrained", staticmethod(fake_auto_config)
    )
    monkeypatch.setattr(
        ktransformers_if.AutoTokenizer, "from_pretrained", staticmethod(fake_auto_tokenizer)
    )
    monkeypatch.setattr(
        ktransformers_if.GenerationConfig,
        "from_pretrained",
        staticmethod(fake_generation_config),
    )
    monkeypatch.setattr(
        ktransformers_if, "optimize_and_load_gguf", lambda *_, **__: None
    )
    class DummyStaticCache:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ktransformers_if, "StaticCache", DummyStaticCache)

    interface = KTransformersInterface(dummy_args)
    assert interface.model.config._attn_implementation == "flash_attention_2"
    assert isinstance(interface.model, DummyModel)


def _make_dummy_config(use_cache=False):
    return types.SimpleNamespace(
        output_attentions=False,
        output_router_logits=False,
        output_hidden_states=False,
        use_cache=use_cache,
        use_return_dict=True,
    )


class DummyLayer(torch.nn.Module):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        self.recorder.append(cache_position.clone() if cache_position is not None else None)
        return (hidden_states + 1,)


class DummyOrigModule(torch.nn.Module):
    def __init__(self, config, recorder):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([DummyLayer(recorder) for _ in range(2)])
        self.norm = torch.nn.Identity()
        self.gradient_checkpointing = False
        self.training = False

    def _update_causal_mask(self, *_, **__):
        return None

    def rotary_emb(self, *_):
        return None


class DummyCache:
    def __init__(self, seq_len):
        self._seq_len = seq_len

    def get_seq_length(self):
        return self._seq_len


def _make_kqwen3(recorder, use_cache=False, threshold=1):
    config = _make_dummy_config(use_cache=use_cache)
    gguf_loader = types.SimpleNamespace()
    orig_module = DummyOrigModule(config, recorder)
    return operators_models.KQwen3MoeModel(
        key="layers",
        gguf_loader=gguf_loader,
        config=config,
        orig_module=orig_module,
        device="cpu",
        per_layer_prefill_intput_threshold=threshold,
    )


def test_kqwen3moe_prefill_threshold_invokes_unload():
    recorder: list = []
    model = _make_kqwen3(recorder, use_cache=False, threshold=1)

    targets = []

    def fake_load(self, layer, target):
        targets.append(target)

    model.load_layer_to = types.MethodType(fake_load, model)
    inputs = torch.ones(1, 3, 4)

    model.forward(inputs_embeds=inputs, return_dict=True)

    assert targets[: len(model.layers)] == [InferenceState.UNLOAD] * len(model.layers)
    assert InferenceState.GENERATE in targets


def test_kqwen3moe_cache_position_defaults():
    recorder: list = []
    model = _make_kqwen3(recorder, use_cache=False)

    model.load_layer_to = types.MethodType(lambda *_, **__: None, model)
    past = DummyCache(seq_len=2)
    inputs = torch.zeros(1, 2, 4)

    model.forward(inputs_embeds=inputs, past_key_values=past, return_dict=True)

    recorded = recorder[0]
    assert torch.equal(recorded, torch.tensor([2, 3]))


def test_kqwen3moe_returns_existing_cache_when_no_new_cache():
    recorder = []
    model = _make_kqwen3(recorder, use_cache=True)
    model.config.use_cache = True

    model.load_layer_to = types.MethodType(lambda *_, **__: None, model)
    inputs = torch.zeros(1, 1, 4)
    past = DummyCache(seq_len=0)

    outputs = model.forward(inputs_embeds=inputs, past_key_values=past, return_dict=True)

    assert outputs.past_key_values is past


def test_load_layer_to_toggles_components(monkeypatch):
    class ModeTracker:
        def __init__(self):
            self.calls = []

        def set_inference_mode(self, target):
            self.calls.append(target)

    class DeviceTracker:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    class FakeDecoderLayer:
        def __init__(self):
            self.self_attn = types.SimpleNamespace(
                q_proj=ModeTracker(),
                k_proj=ModeTracker(),
                v_proj=ModeTracker(),
                o_proj=ModeTracker(),
                rotary_emb=DeviceTracker(),
            )
            self.mlp = types.SimpleNamespace(
                gate_proj=ModeTracker(),
                up_proj=ModeTracker(),
                down_proj=ModeTracker(),
            )
            self.input_layernorm = DeviceTracker()
            self.post_attention_layernorm = DeviceTracker()

    monkeypatch.setattr(operators_models, "Qwen3MoeDecoderLayer", FakeDecoderLayer)

    model = operators_models.KQwen3MoeModel(
        key="layers",
        gguf_loader=types.SimpleNamespace(),
        config=_make_dummy_config(),
        orig_module=torch.nn.Module(),
        device="cpu",
    )

    layer = FakeDecoderLayer()

    model.load_layer_to(layer, InferenceState.UNLOAD)
    attn_modules = [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
    ]
    for module in attn_modules + [
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]:
        assert module.calls[-1] == InferenceState.UNLOAD
    assert layer.input_layernorm.devices[-1] == "cpu"
    assert layer.post_attention_layernorm.devices[-1] == "cpu"

    model.load_layer_to(layer, InferenceState.GENERATE)
    for module in attn_modules + [
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]:
        assert module.calls[-1] == InferenceState.GENERATE
    assert layer.input_layernorm.devices[-1] == "cuda"
    assert layer.post_attention_layernorm.devices[-1] == "cuda"
