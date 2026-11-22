"""
Tree attention as a Hugging Face attention implementation, a drop-in for a model's existing attention.

``register_liger_tree_attention()`` registers it under ``"liger_tree"``; then
``model.set_attn_implementation("liger_tree")`` (or ``from_pretrained(..., attn_implementation="liger_tree")``) swaps
it in for every attention layer of the model. It runs, from what the forward is given:

- a prefix-tree-packed row (TRL's ``TreeRow.pack`` keyword arguments ``tree_cu_seqlens_q`` and ``tree_segment_parents``):
  each segment attends to its own earlier tokens and to all of its ancestors, without gathering any K/V;
- a padding-free packed batch (transformers' ``cu_seq_lens_q`` / ``cu_seq_lens_k``, as ``DataCollatorWithFlattening``
  produces them): causal attention within each sequence;
- an unpadded batch: causal attention within each row;
- a decoding step with a KV cache (one query token per row): plain attention over the cache, through SDPA.

Padded batches are refused: pack them instead, as above.

To split a long row over GPUs (context parallel, or Ulysses sequence parallel), build the row's metadata once with
``build_tree_attention_metadata(cu_seqlens, parents, n_tokens, cp_group=...)`` (or ``sp_group=...``) from the whole
row's tree, feed each rank its contiguous ``n_tokens / P`` tokens (``input_ids`` and tree ``position_ids``), and pass
the metadata to the model as ``tree_attention_metadata``.
"""

from typing import Optional

import torch

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface

from liger_kernel.ops import LigerTreeAttentionFunction
from liger_kernel.ops.tree_attention import TreeAttentionMetadata
from liger_kernel.ops.tree_attention import build_tree_attention_metadata

LIGER_TREE_ATTENTION = "liger_tree"


def _row_metadata(query, kwargs, tree_attention_metadata):
    """The packed row's tree, or `None` for a batch that is not one packed row."""
    if tree_attention_metadata is not None:
        return tree_attention_metadata
    n_tokens = query.shape[2]
    if kwargs.get("tree_cu_seqlens_q") is not None:
        return build_tree_attention_metadata(kwargs["tree_cu_seqlens_q"], kwargs["tree_segment_parents"], n_tokens)
    cu_seqlens = kwargs.get("cu_seq_lens_q")
    if cu_seqlens is not None:
        cu_seqlens_k = kwargs.get("cu_seq_lens_k")
        if cu_seqlens_k is not None and cu_seqlens_k is not cu_seqlens and not torch.equal(cu_seqlens, cu_seqlens_k):
            raise NotImplementedError("liger tree attention needs the same query and key sequences")
        roots = torch.full((cu_seqlens.numel() - 1,), -1, dtype=torch.int32, device=cu_seqlens.device)
        return build_tree_attention_metadata(cu_seqlens, roots, n_tokens)
    return None


def liger_tree_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    tree_attention_metadata: Optional[TreeAttentionMetadata] = None,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    s_aux: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Attention in the `AttentionInterface` signature (see the module docstring for the inputs it takes).

    Args:
        query, key, value: `(B, H, L, D)`; a packed row is one batch element.
        tree_attention_metadata: a packed row's [`TreeAttentionMetadata`], to build once and share across layers, and
            needed to split the row over a `cp_group` / `sp_group`. Without it, it is built per layer from
            `tree_cu_seqlens_q` and `tree_segment_parents` (or `cu_seq_lens_q`), on the GPU and without a host sync.
        scaling: softmax scale, `1 / sqrt(D)` when `None`.
        sliding_window: window in positions, as transformers passes it to FlashAttention.

    Returns:
        `(B, L, H, D)` attention output, and `None` attention weights.
    """
    if softcap:
        raise NotImplementedError("liger tree attention does not support attention logit softcapping")
    if s_aux is not None:
        raise NotImplementedError("liger tree attention does not support attention sinks")
    batch, _, q_len, _ = query.shape
    if q_len != key.shape[2]:
        if q_len == 1:  # a decoding step over a KV cache: the query sees every cached key
            return sdpa_attention_forward(module, query, key, value, None, scaling=scaling, is_causal=False)
        raise NotImplementedError("liger tree attention does not support chunked prefill over a KV cache")

    meta = _row_metadata(query, kwargs, tree_attention_metadata)
    if meta is not None:
        if batch != 1:
            raise ValueError(f"a packed row is one batch element, got a batch of {batch}")
        # (1, H, N, D) -> (N, H, D) views; the kernels take the strides as they are
        q, k, v = (x[0].transpose(0, 1) for x in (query, key, value))
        return LigerTreeAttentionFunction.apply(q, k, v, meta, scaling, sliding_window).unsqueeze(0), None

    # an unpadded batch: every row is its own root
    q, k, v = (x.transpose(1, 2).reshape(batch * q_len, x.shape[1], x.shape[3]) for x in (query, key, value))
    cu_seqlens = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=query.device)
    roots = torch.full((batch,), -1, dtype=torch.int32, device=query.device)
    meta = build_tree_attention_metadata(cu_seqlens, roots, batch * q_len)
    out = LigerTreeAttentionFunction.apply(q, k, v, meta, scaling, sliding_window)
    return out.view(batch, q_len, *out.shape[1:]), None


def _no_mask(attention_mask: Optional[torch.Tensor] = None, **kwargs) -> None:
    """Visibility comes from the tree or the packing, so no mask is built; a padding mask cannot be honored."""
    if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
        raise NotImplementedError(
            "liger tree attention does not support padded batches; pack them padding-free instead "
            "(e.g. transformers' DataCollatorWithFlattening, or TRL's tree packing)"
        )
    return None


def register_liger_tree_attention(name: str = LIGER_TREE_ATTENTION) -> str:
    """Register [`liger_tree_attention_forward`] with transformers under `name`; returns `name`."""
    AttentionInterface.register(name, liger_tree_attention_forward)
    AttentionMaskInterface.register(name, _no_mask)
    return name
