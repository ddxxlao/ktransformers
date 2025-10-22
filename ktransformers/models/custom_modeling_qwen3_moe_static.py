"""
Static-cache enabled wrapper for Qwen3 MoE.

This class simply toggles the `_supports_static_cache` flag so the single-process
ktransformers backend can instantiate a `StaticCache` while reusing the original
HuggingFace forward implementation.
"""

from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM


class KQwen3MoeForCausalLMStatic(Qwen3MoeForCausalLM):
    _supports_static_cache = True
