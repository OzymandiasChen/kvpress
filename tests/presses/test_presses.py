# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from transformers import DynamicCache

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    ChunkKVSummaryPress,
    ChunkPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DMSPress,
    EntropyGatedChunkKVPress,
    FastKVzipPress,
    KeyRerotationPress,
    KnormPress,
    KVComposePress,
    KVzipPress,
    MergingPress,
    ObservedAttentionPress,
    ScorerPress,
    SnapKVPress,
    ThinKPress,
)
from tests.default_presses import default_presses
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def test_composed_press(unit_test_model):  # noqa: F811
    press1 = KnormPress(compression_ratio=0.5)
    press2 = ThinKPress(key_channel_compression_ratio=0.5, window_size=2)
    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"].to(unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_chunk_press(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


def test_chunkkv_press(unit_test_model):  # noqa: F811
    press = SnapKVPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkKVPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


class _StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [11, 12, 13]


def test_chunkkv_summary_press(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 280), device=unit_test_model.device)

    def run(press):
        cache = DynamicCache()
        with press(unit_test_model), torch.no_grad():
            unit_test_model.model(input_ids=input_ids, past_key_values=cache)
        return cache

    # summary_length=0 is exactly ChunkKV
    reference = run(ChunkKVPress(press=SnapKVPress(compression_ratio=0.5), chunk_length=16))
    disabled = run(ChunkKVSummaryPress(press=SnapKVPress(compression_ratio=0.5), chunk_length=16, summary_length=0))
    for ref_layer, layer in zip(reference.layers, disabled.layers):
        assert torch.equal(ref_layer.keys, layer.keys) and torch.equal(ref_layer.values, layer.values)

    # M=16, P=4, N=8: 280 tokens contain 4 groups of 64 tokens -> ChunkKV tokens + 4 * 8 summary tokens
    press = ChunkKVSummaryPress(
        press=SnapKVPress(compression_ratio=0.5),
        chunk_length=16,
        chunks_per_summary=4,
        summary_length=8,
        tokenizer=_StubTokenizer(),
    )
    cache = run(press)
    assert len(press.summary_token_ids) == 4 and all(len(ids) == 8 for ids in press.summary_token_ids)
    for layer_idx in range(len(cache)):
        assert cache.get_seq_length(layer_idx) == reference.get_seq_length(layer_idx) + 4 * 8


@pytest.mark.parametrize("press_dict", default_presses)
@pytest.mark.parametrize(
    "wrapper_press",
    [
        None,
        ComposedPress,
        KeyRerotationPress,
        AdaKVPress,
        ChunkPress,
        CriticalKVPress,
        CriticalAdaKVPress,
        DMSPress,
        MergingPress,
        EntropyGatedChunkKVPress,
    ],
)
def test_presses_run(unit_test_model, press_dict, wrapper_press):  # noqa: F811
    cls = press_dict["cls"]
    for kwargs in press_dict["kwargs"]:
        press = cls(**kwargs)
        if wrapper_press is not None:
            if hasattr(press, "post_init_from_model"):
                press.post_init_from_model(unit_test_model)
            if issubclass(wrapper_press, ComposedPress):
                if isinstance(press, (KVzipPress, FastKVzipPress, KVComposePress)):
                    # KVzipPress, FastKVzipPress and KVComposePress are currently not compatible with ComposedPress
                    return
                press = ComposedPress(presses=[press])
            elif not isinstance(press, ScorerPress):  # remaining wrapper presses only support ScorerPress
                return
            elif issubclass(
                wrapper_press,
                (
                    KeyRerotationPress,
                    AdaKVPress,
                    CriticalKVPress,
                    CriticalAdaKVPress,
                    MergingPress,
                    EntropyGatedChunkKVPress,
                ),
            ):
                press = wrapper_press(press=press)
            elif issubclass(wrapper_press, ChunkPress):
                press = ChunkPress(press=press, chunk_length=24)
            elif issubclass(wrapper_press, DMSPress):
                press = DMSPress(press=press, threshold=-0.5, sliding_window_size=32)

        # TODO: Handle post_init_from_model differently
        if hasattr(press, "post_init_from_model"):
            press.post_init_from_model(unit_test_model)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
        # Check that the press has a compression_ratio attribute
        assert hasattr(press, "compression_ratio")


def test_presses_run_observed_attention(unit_test_model_output_attention):  # noqa: F811
    for cls in [ObservedAttentionPress]:
        for compresion_ratio in [0.2, 0.8]:
            press = cls(compression_ratio=compresion_ratio)
            with press(unit_test_model_output_attention):
                input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"].to(
                    unit_test_model_output_attention.device
                )
                unit_test_model_output_attention(input_ids, past_key_values=DynamicCache()).past_key_values


@dataclass
class StoreKnormPress(ScorerPress):
    def __post_init__(self):
        self.scores = []

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        scores = -keys.norm(dim=-1)
        self.scores.append(scores)
        return scores


@torch.no_grad()
def test_presses_keep_highest_score(unit_test_model):  # noqa: F811
    """
    Test that kept keys are those with the highest score
    """
    for compresion_ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
        press = StoreKnormPress(compression_ratio=compresion_ratio)
        with press(unit_test_model):
            input_ids = torch.randint(0, 3_000, (5, 256), device=unit_test_model.device)
            past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

        keys = [layer.keys for layer in past_key_values.layers]
        for scores, key in zip(press.scores, keys):
            max_scores = -key.norm(dim=-1)
            for batch_idx in range(scores.shape[0]):
                for head_idx in range(scores.shape[1]):
                    assert torch.allclose(
                        scores[batch_idx, head_idx].sort().values[-max_scores.shape[-1] :],
                        max_scores[batch_idx, head_idx].sort().values,
                    )
