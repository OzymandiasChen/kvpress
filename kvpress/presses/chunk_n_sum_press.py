# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, DynamicCache, Gemma3ForConditionalGeneration, PreTrainedModel, QuantizedCache

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress, is_prefilling
from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class ChunkNSumPress(BasePress):
    """
    Chunk-n-Sum: ChunkKV compression plus learned-free "summary" KV pairs for every span of the context.

    The context is split into spans of ``chunk_length * summary_span`` tokens. For every complete span, the tokens
    of ``summary_prompt`` (e.g. "Summary :") are virtually appended right after the span (positions
    ``span_end, span_end + 1, ...``) and forwarded through the model, layer by layer, attending only to the
    (uncompressed) span and to themselves. Their KV pairs are then added to the compressed cache so that the
    information of dropped chunks can still be accessed through these summary tokens.

    For each decoder layer, during pre-filling:
        1. summary: run the decoder layer on the summary hidden states of every span, using the span keys/values
           as the past. Keep the resulting summary KV pairs, and the output hidden states as the input for the
           next layer (layer 0 starts from the embeddings of ``summary_prompt``).
        2. compress: apply ChunkKV (chunk-wise selection with ``press`` scores) to the original context KV pairs.
        3. merge: concatenate the compressed context KV pairs and the summary KV pairs.

    The hook is registered on decoder layers (not attention layers) because the summary tokens need the full
    layer (attention + MLP) to produce the hidden states of the next layer.

    Note that the summary KV pairs are added on top of the ChunkKV budget: the final cache length is
    ``n_kept_by_chunkkv + n_spans * len(summary_prompt_tokens)``.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used by ChunkKV (SnapKVPress is a good choice).
    chunk_length : int, default=16
        Length of each chunk for ChunkKV selection.
    summary_span : int, default=8
        Number of chunks summarized by each group of summary tokens.
    summary_prompt : str, default="Summary :"
        Text whose tokens are used as summary tokens.
    summary_token_ids : list[int], optional
        Token ids of the summary tokens. If None, ``summary_prompt`` is tokenized with the tokenizer
        loaded from ``model.config.name_or_path``.
    """

    press: ScorerPress
    chunk_length: int = 16
    summary_span: int = 8
    summary_prompt: str = "Summary :"
    summary_token_ids: Optional[list[int]] = None

    _chunkkv: ChunkKVPress = field(init=False, repr=False)
    _summary_hidden_states: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkNSumPress requires a ScorerPress as input"
        assert self.chunk_length > 0 and self.summary_span > 0, "chunk_length and summary_span must be positive"
        self._chunkkv = ChunkKVPress(press=self.press, chunk_length=self.chunk_length)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @property
    def span_length(self) -> int:
        return self.chunk_length * self.summary_span

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)
        if self.summary_token_ids is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            self.summary_token_ids = tokenizer.encode(self.summary_prompt, add_special_tokens=False)
        assert len(self.summary_token_ids) > 0, "summary_prompt must contain at least one token"

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")
        if isinstance(model, Gemma3ForConditionalGeneration):
            raise NotImplementedError("ChunkNSumPress does not support Gemma3 (sliding window layers)")
        if model.config._attn_implementation not in ("eager", "sdpa"):
            raise NotImplementedError("ChunkNSumPress requires attn_implementation='eager' or 'sdpa'")

        self.post_init_from_model(model)
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self._embed_tokens = language_model.embed_tokens
        self._rotary_emb = language_model.rotary_emb
        self._summary_hidden_states = None

        hooks = []
        try:
            for layer in language_model.layers:
                layer.self_attn.rotary_emb = language_model.rotary_emb
                # Hook on the decoder layer (not self_attn): summary tokens need the full layer forward
                hooks.append(layer.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
            self._summary_hidden_states = None

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output):
        """
        Forward hook on a decoder layer. `module` is the decoder layer, `module.self_attn` its attention layer.
        """
        # Decoder layers receive hidden_states positionally
        layer_input = input[0] if len(input) > 0 else kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        layer_idx = module.self_attn.layer_idx
        cache_layer = cache.layers[layer_idx]
        q_len = layer_input.shape[1]

        # Don't compress after pre-filling
        if not is_prefilling(kwargs["cache_position"], q_len):
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)

        # 1. Summary KV pairs, computed on the uncompressed cache
        summary_keys, summary_values = self.summary(module, keys, values)

        # 2. ChunkKV compression. Scorer presses expect the input of the attention layer (i.e. after layernorm)
        hidden_states = module.input_layernorm(layer_input)
        keys, values = self._chunkkv.compress(module.self_attn, hidden_states, keys, values, None, kwargs)

        # 3. Merge
        keys, values = self.merge(keys, values, summary_keys, summary_values)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    def summary(
        self, module: nn.Module, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward the summary tokens of every complete span through the decoder layer `module`.

        Each span is treated as an independent batch element: the summary tokens of span i attend to the
        keys/values of span i and (causally) to themselves. The output hidden states are stored to be used
        as input of the next layer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] or (None, None)
            Summary keys and values with shape (batch_size, num_kv_heads, n_spans * n_summary_tokens, head_dim),
            or (None, None) if the context is shorter than one span.
        """
        bsz, num_kv_heads, kv_len, head_dim = keys.shape
        n_spans = kv_len // self.span_length
        if n_spans == 0:
            return None, None
        layer_idx = int(module.self_attn.layer_idx)
        span_len = self.span_length
        n_sum = len(self.summary_token_ids)  # type: ignore[arg-type]
        device = keys.device

        # Input hidden states of the summary tokens: embeddings at layer 0, previous layer output otherwise
        if layer_idx == 0 or self._summary_hidden_states is None:
            ids = torch.tensor(self.summary_token_ids, device=self._embed_tokens.weight.device)
            hidden = self._embed_tokens(ids)[None, None].expand(bsz, n_spans, -1, -1)
            hidden = hidden.reshape(bsz * n_spans, n_sum, -1)
        else:
            hidden = self._summary_hidden_states
        hidden = hidden.to(device)

        # (bsz, H, n_spans * span_len, D) -> (bsz * n_spans, H, span_len, D)
        def to_spans(x):
            x = x[:, :, : n_spans * span_len].reshape(bsz, num_kv_heads, n_spans, span_len, head_dim)
            return x.transpose(1, 2).reshape(bsz * n_spans, num_kv_heads, span_len, head_dim)

        # Past of the summary tokens: the span keys/values only
        span_cache = DynamicCache()
        span_cache.update(to_spans(keys), to_spans(values), layer_idx)

        # Summary tokens are placed right after their span
        span_ends = torch.arange(1, n_spans + 1, device=device) * span_len
        position_ids = (span_ends[:, None] + torch.arange(n_sum, device=device)[None]).repeat(bsz, 1)
        position_embeddings = self._rotary_emb(hidden, position_ids)

        # Additive mask: attend to the full span and causally to the summary tokens
        dtype = hidden.dtype
        mask = torch.zeros(n_sum, span_len + n_sum, dtype=dtype, device=device)
        mask[:, span_len:] = torch.triu(torch.full((n_sum, n_sum), torch.finfo(dtype).min, device=device), 1)
        mask = mask[None, None].expand(bsz * n_spans, 1, -1, -1)

        # Call forward directly (not __call__) to avoid re-triggering this hook
        out = module.forward(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=span_cache,
            use_cache=True,
            cache_position=torch.arange(span_len, span_len + n_sum, device=device),
            position_embeddings=position_embeddings,
        )
        self._summary_hidden_states = out[0] if isinstance(out, tuple) else out

        # (bsz * n_spans, H, n_sum, D) -> (bsz, H, n_spans * n_sum, D)
        def from_spans(x):
            x = x[:, :, span_len:].reshape(bsz, n_spans, num_kv_heads, n_sum, head_dim)
            return x.transpose(1, 2).reshape(bsz, num_kv_heads, n_spans * n_sum, head_dim)

        summary_keys = from_spans(span_cache.layers[layer_idx].keys)
        summary_values = from_spans(span_cache.layers[layer_idx].values)
        return summary_keys, summary_values

    def merge(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        summary_keys: Optional[torch.Tensor],
        summary_values: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate compressed context KV pairs and summary KV pairs. The order in the cache does not matter
        for attention, positional information is already encoded in the keys (RoPE).
        """
        if summary_keys is None:
            return keys, values
        keys = torch.cat([keys, summary_keys.to(keys.dtype)], dim=2).contiguous()
        values = torch.cat([values, summary_values.to(values.dtype)], dim=2).contiguous()  # type: ignore[union-attr]
        return keys, values

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        raise NotImplementedError("ChunkNSumPress applies compression in forward_hook on decoder layers")
