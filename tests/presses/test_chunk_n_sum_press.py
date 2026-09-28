# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from kvpress import ChunkNSumPress, SnapKVPress

CONFIG_KWARGS = dict(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
)
SUMMARY_IDS = [5, 6, 7]


@pytest.mark.parametrize("model_cls,config_cls", [(LlamaForCausalLM, LlamaConfig), (Qwen3ForCausalLM, Qwen3Config)])
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_chunk_n_sum_press(model_cls, config_cls, attn_implementation):
    torch.manual_seed(0)
    model = model_cls(config_cls(**CONFIG_KWARGS, attn_implementation=attn_implementation)).eval()
    press = ChunkNSumPress(
        SnapKVPress(compression_ratio=0.5, window_size=4),
        chunk_length=4,
        summary_span=4,
        summary_token_ids=SUMMARY_IDS,
    )
    ids = torch.randint(10, 1000, (1, 50))
    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model.model(input_ids=ids, past_key_values=cache)

    # Summary tokens of the first span must match a plain forward on [first span + summary tokens]
    span_len, n_spans, n_sum = 16, 3, len(SUMMARY_IDS)
    ref = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=torch.cat([ids[:, :span_len], torch.tensor([SUMMARY_IDS])], dim=1), past_key_values=ref)

    for layer_idx in range(len(cache)):
        keys, values = cache.layers[layer_idx].keys, cache.layers[layer_idx].values
        assert keys.shape[2] < ids.shape[1]
        start = keys.shape[2] - n_spans * n_sum
        assert torch.allclose(keys[:, :, start : start + n_sum], ref.layers[layer_idx].keys[:, :, span_len:], atol=1e-5)
        assert torch.allclose(
            values[:, :, start : start + n_sum], ref.layers[layer_idx].values[:, :, span_len:], atol=1e-5
        )
