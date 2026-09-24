# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, QuantizedCache

from kvpress.presses.chunkkv_press import ChunkKVPress


@dataclass
class ChunkKVSummaryPress(ChunkKVPress):
    """
    ChunkKV with generated summary tokens.

    Runs ChunkKV profiling and chunk selection as usual, and additionally, for every group of
    ``chunks_per_summary`` chunks (P chunks of ``chunk_length`` = M tokens), inserts ``summary_prompt``
    right after the group and greedily generates ``summary_length`` (N) tokens. The KV pairs of these
    generated tokens are kept in the cache next to the ChunkKV-retained tokens of their group:

        KV cache = ChunkKV-retained original tokens + summary tokens

    Original tokens discarded by ChunkKV are removed even if they were summarized, and the KV pairs
    of the prompt itself are dropped. With ``summary_length=0`` this press is exactly ChunkKVPress.

    How it works:
    1. During pre-filling, the forward hook of every attention layer calls ``compress``, which runs
       the ChunkKV profiling / selection (``select_indices``) and stores the kept positions per layer,
       but leaves the cache untouched so the summaries can attend to the full group.
    2. When the context manager exits (after pre-filling), for each group g ending at position
       ``end_g``, every layer of the cache is truncated to ``[0, end_g)`` and the model processes
       ``summary_prompt`` followed by N greedy steps. The summary therefore sees exactly what it would
       see if ``summary_prompt`` had been inserted inline after the group, and receives the positions
       ``end_g, end_g + 1, ...`` (the RoPE positions of an inline insertion).
    3. Every layer of the cache is rebuilt as
       ``[kept tokens of group 0, summary 0, kept tokens of group 1, summary 1, ..., kept tail tokens]``.

    Trailing tokens that do not fill a complete group of P chunks are compressed by ChunkKV only.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used to compute global importance scores.
    chunk_length : int, default=16
        M, the length of each chunk for token selection.
    chunks_per_summary : int, default=4
        P, the number of consecutive chunks summarized together.
    summary_length : int, default=8
        N, the number of generated summary tokens kept per group. 0 disables summarization.
    summary_prompt : str, default="summary:"
        Prompt inserted after each group of P chunks. Its KV pairs are not kept.
    tokenizer : optional
        Tokenizer used to encode ``summary_prompt``. Loaded from the model name if None.

    Notes
    -----
    Summary generation requires batch size 1 (as in the kvpress pipeline), a non-quantized cache and
    full attention in every layer (no sliding-window layers). The generated token ids of the last run
    are stored in ``summary_token_ids`` for inspection.
    """

    chunk_length: int = 16
    chunks_per_summary: int = 4
    summary_length: int = 8
    summary_prompt: str = "summary:"
    tokenizer: Optional[Any] = field(default=None, repr=False)

    summary_token_ids: list[list[int]] = field(default_factory=list, init=False, repr=False)
    _keep_indices: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _cache: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.chunk_length > 0 and self.chunks_per_summary > 0 and self.summary_length >= 0

    def post_init_from_model(self, model):
        super().post_init_from_model(model)
        if self.summary_length > 0 and self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    @property
    def group_length(self) -> int:
        return self.chunk_length * self.chunks_per_summary

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_len = keys.shape[2]
        if self.summary_length == 0 or kv_len // self.group_length == 0:
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)

        cache = kwargs["past_key_values"]
        if isinstance(cache, QuantizedCache):
            raise NotImplementedError("ChunkKVSummaryPress does not support QuantizedCache")
        assert keys.shape[0] == 1, "ChunkKVSummaryPress only supports batch size 1"

        # ChunkKV profiling and token selection, applied later in _summarize_and_compress
        if self.press.compression_ratio == 0:
            indices = torch.arange(kv_len, device=keys.device)
        else:
            assert attentions is None, "ChunkPress does not support attentions."
            indices = self.select_indices(module, hidden_states, keys, values, attentions, kwargs)
        self._keep_indices[int(module.layer_idx)] = indices
        self._cache = cache
        return keys, values

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self._keep_indices, self._cache = {}, None
        with super().__call__(model):
            yield
        # Hooks are removed at this point, so the summary forward passes below are not compressed
        if self._cache is not None:
            self._summarize_and_compress(model)
        self._keep_indices, self._cache = {}, None

    @torch.no_grad()
    def _summarize_and_compress(self, model: PreTrainedModel):
        cache = self._cache
        n_layers = len(cache.layers)
        if len(self._keep_indices) != n_layers:
            raise NotImplementedError("ChunkKVSummaryPress requires every layer to be compressed")

        original = [(layer.keys, layer.values) for layer in cache.layers]
        kv_len = original[0][0].shape[2]
        n_groups = kv_len // self.group_length
        prompt_ids = self.tokenizer.encode(self.summary_prompt, add_special_tokens=False)
        prompt_ids = torch.tensor([prompt_ids], device=original[0][0].device)

        # 1. Generate N summary tokens for each group of P chunks
        self.summary_token_ids = []
        summaries = []  # summaries[g][layer_idx] = (keys, values) of the N summary tokens of group g
        for g in range(n_groups):
            end = (g + 1) * self.group_length
            # Keep only the context up to the end of the group, as if the prompt was inserted inline.
            # Cache positions then equal the inline positions, so default position ids are correct.
            for layer, (keys, values) in zip(cache.layers, original):
                layer.keys, layer.values = keys[:, :, :end], values[:, :, :end]

            input_ids, generated = prompt_ids, []
            for _ in range(self.summary_length):
                logits = model(input_ids=input_ids, past_key_values=cache, logits_to_keep=1).logits
                input_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(input_ids.item())
            # The last generated token has not been through the model yet: run it to get its KV pairs
            model.model(input_ids=input_ids, past_key_values=cache)

            start = end + prompt_ids.shape[1]  # drop the KV pairs of the prompt
            summaries.append([(layer.keys[:, :, start:], layer.values[:, :, start:]) for layer in cache.layers])
            self.summary_token_ids.append(generated)

        # 2. Apply ChunkKV to the original tokens and insert each summary after its group
        for layer_idx, (layer, (keys, values)) in enumerate(zip(cache.layers, original)):
            indices = self._keep_indices[layer_idx]
            new_keys, new_values = [], []
            for g in range(n_groups + 1):
                start, end = g * self.group_length, (g + 1) * self.group_length
                kept = indices[(indices >= start) & (indices < end)]
                new_keys.append(keys[:, :, kept])
                new_values.append(values[:, :, kept])
                if g < n_groups:
                    new_keys.append(summaries[g][layer_idx][0])
                    new_values.append(summaries[g][layer_idx][1])
            layer.keys = torch.cat(new_keys, dim=2).contiguous()
            layer.values = torch.cat(new_values, dim=2).contiguous()


if __name__ == "__main__":
    # Example: M=16, P=4, N=8 on a 256-token context with 50% ChunkKV compression.
    # 256 tokens = 16 chunks = 4 groups -> 128 retained original tokens + 4 * 8 summary tokens = 160.
    from transformers import AutoModelForCausalLM, DynamicCache

    from kvpress import SnapKVPress

    ckpt = "MaxJeblick/llama2-0b-unit-test"
    model = AutoModelForCausalLM.from_pretrained(ckpt).eval()
    input_ids = torch.randint(0, 1024, (1, 256))

    for summary_length in [0, 8]:
        press = ChunkKVSummaryPress(
            press=SnapKVPress(compression_ratio=0.5),
            chunk_length=16,
            chunks_per_summary=4,
            summary_length=summary_length,
        )
        cache = DynamicCache()
        with press(model), torch.no_grad():
            model.model(input_ids=input_ids, past_key_values=cache)
        print(f"N={summary_length}: cache length per layer = {[cache.get_seq_length(i) for i in range(len(cache))]}")
        if summary_length:
            print(f"summary token ids per group: {press.summary_token_ids}")
