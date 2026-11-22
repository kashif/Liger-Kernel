# The forward and backward kernels below follow flash-linear-attention's fla/ops/attn/parallel.py
# (https://github.com/fla-org/flash-linear-attention), distributed under the MIT license:
#
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Tree attention over a prefix-tree-packed row, without materializing per-segment K/V.

A tree-packed row (as built by TRL's ``TreeRow.pack``) lays a prefix trie out depth-first: every unary chain of the
trie is a *segment*, a contiguous run of packed tokens, and segments are numbered in DFS pre-order. A query in segment
``s`` attends causally to the earlier tokens of ``s`` and to every token of every ancestor segment of ``s``.

Gathering each segment's root-to-segment path and calling a varlen attention kernel duplicates a shared prefix once per
descendant segment, and that gathered K/V is what the attention kernel saves for backward. Here nothing is gathered:

- Forward / dQ: a query tile walks the parent pointers of its segment and streams each ancestor's contiguous K/V range.
- dK / dV: in pre-order, the descendants of a segment occupy one contiguous token range, so a K/V tile streams its own
  segment's later queries (causally) and then that range (unmasked). Each K/V head reduces its query-head group in
  registers, so the backward is deterministic and needs no atomics and no per-query-head dK/dV buffers.

Activations saved for backward are Q, K, V, O and the log-sum-exp, all sized by the *unique* tokens of the row.

Numerics, following "Broken symmetry in BF16 attention" (Chen et al., arXiv:2609.34272):

- Subtract-before-scale: the running max and the saved log-sum-exp are kept in raw score units and probabilities are
  ``exp2((qk - m) * scale)`` in forward and backward alike, so backward recomputes the probabilities that made ``O``.
- GProj: for half-precision inputs the score gradient ``dS = P * (dP - delta)`` sums to zero per query, but its cast to
  the matmul dtype does not. The leaked row mass ``rho`` couples into dQ/dK through the key/query mean. It is projected
  out against ``r = cast(P)`` as in the paper's Algorithm 1: with ``lambda = rho / sum(r)`` (0 where ``sum(r) = 0``),
  ``dQ = alpha * (dQ_acc - lambda * (r @ K))`` and ``dK = alpha * (dK_acc - r^T @ cast(lambda * Q))``, both corrected in
  fp32 before the one cast to the output dtype; dV is not projected. Unlike the paper's fused kernel, whose key pass
  recomputes ``lambda``, the dQ kernel here stores it and the dK/dV kernel reads it, so both use the same ``lambda``.

Hardware: on GPUs with TMA (sm90+) the streamed K/V (forward, dQ) and Q/dO (dK/dV) tiles are read through host tensor
descriptors, which Triton lowers to TMA loads; elsewhere (and on ROCm, Triton builds without tensor descriptors, or
tensors a descriptor cannot address) the kernels read them through masked pointer loads. Tile sizes, warps and pipeline stages
are chosen by ``triton.autotune`` (``LIGER_TREE_ATTENTION_AUTOTUNE=0`` pins one portable config). The streamed loops run
inside the walk over a segment's ancestors, with data-dependent bounds, so they are not warp-specialized.

Sequence parallelism: a row split evenly over a process group, every rank holding its contiguous shard of tokens (as
transformers / accelerate shard ``input_ids``), with the group passed to [`build_tree_attention_metadata`]:

- ``cp_group`` (context parallel, Llama 3 style): each rank all-gathers K/V (small under GQA) and runs only its own
  query tiles against the whole row; the kernels take the rank's query range, so the tree metadata stays global. In
  backward each rank re-gathers K/V, computes the dK/dV of its queries for the whole row in fp32, and the group
  reduce-scatters them. Unlike ring attention, no partial outputs are merged, so the numerics are those of one GPU.
- ``sp_group`` (Ulysses / DeepSpeed ALST style): an all-to-all trades the token shard for a head shard, each rank runs
  the whole row for ``H / P`` heads (key/value heads replicated when there are fewer than ``P``), and a second all-to-all
  trades back.
"""

import functools
import math
import os

from typing import NamedTuple
from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import is_hip

# Optional: not all Triton builds ship triton.tools.tensor_descriptor.
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ModuleNotFoundError:
    TensorDescriptor = None

LOG2E = 1.4426950408889634
BLOCK_T_SIZES = (32, 64, 128)  # outer tile sizes the autotuner picks from; the metadata holds a tile list for each


class TreeAttentionMetadata(NamedTuple):
    """Per-row launch metadata, shared by every attention layer of one forward. Built without a host sync."""

    cu_seqlens: torch.Tensor  # (S + 1,) int32, segment boundaries in the packed row
    parents: torch.Tensor  # (S,) int32, parent segment, -1 for a root; parents[s] < s (pre-order)
    subtree_end: torch.Tensor  # (S,) int32, end token (exclusive) of the pre-order range of s and its descendants
    positions: torch.Tensor  # (N,) int32, depth of each token in the tree (its position in its samples)
    tiles: tuple  # per size in BLOCK_T_SIZES: (T, 2) int32, (segment, first token) of each tile, segment -1 for padding
    # context / sequence parallelism (see the module docstring); the fields above always describe the whole row
    query_start: int = 0  # first token of this rank's query shard
    query_tiles: Optional[tuple] = None  # `tiles` of this rank's query shard (None: `tiles`)
    cp_group: Optional["torch.distributed.ProcessGroup"] = None
    sp_group: Optional["torch.distributed.ProcessGroup"] = None


# ---------------------------------------------------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit(do_not_specialize=["S"])
def _tree_segment_info_kernel(
    CU,
    PARENTS,
    SUBTREE_END,
    POS_START,
    S,
    BLOCK: tl.constexpr,
):
    """Per segment: the end of its pre-order subtree, and its depth offset (summed length of its ancestors)."""
    s = tl.program_id(0)

    # The subtree of s is the segments s+1 .. e-1, where e is the first later segment whose parent precedes s.
    end = S
    found = 0
    start = s + 1
    while (found == 0) & (start < S):
        t = start + tl.arange(0, BLOCK)
        p = tl.load(PARENTS + t, mask=t < S, other=0)
        hit = (t < S) & (p < s)
        first = tl.min(tl.where(hit, t, S))
        if first < S:
            end = first
            found = 1
        start += BLOCK
    tl.store(SUBTREE_END + s, tl.load(CU + end))

    pos = 0
    a = tl.load(PARENTS + s)
    while a >= 0:
        pos += tl.load(CU + a + 1) - tl.load(CU + a)
        a = tl.load(PARENTS + a)
    tl.store(POS_START + s, pos)


def _segment_tiles(cu_seqlens: torch.Tensor, block: int, lo: int, hi: int) -> torch.Tensor:
    """
    (segment, first token) of every `block`-token tile of every segment's part in tokens `[lo, hi)`, padded with
    segment -1 to a static count.
    """
    n_segments = cu_seqlens.numel() - 1
    starts = cu_seqlens[:-1].long().clamp(lo, hi)
    lengths = cu_seqlens[1:].long().clamp(lo, hi) - starts
    chunks = (lengths + block - 1) // block
    ends = chunks.cumsum(0)
    n_tiles = triton.cdiv(hi - lo, block) + n_segments  # upper bound on sum(ceil(len / block)), no sync
    tile = torch.arange(n_tiles, device=cu_seqlens.device)
    segment = torch.searchsorted(ends, tile, right=True)
    valid = segment < n_segments
    segment = segment.clamp(max=n_segments - 1)
    start = starts[segment] + (tile - (ends[segment] - chunks[segment])) * block
    return torch.stack([torch.where(valid, segment, -1), torch.where(valid, start, 0)], dim=1).to(torch.int32)


def build_tree_attention_metadata(
    cu_seqlens: torch.Tensor,
    parents: torch.Tensor,
    n_tokens: Optional[int] = None,
    *,
    cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    sp_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> TreeAttentionMetadata:
    """
    Launch metadata for [`LigerTreeAttentionFunction`], to build once per row and share across layers.

    Args:
        cu_seqlens: `(S + 1,)` segment boundaries of the packed row (TRL's `tree_cu_seqlens_q`).
        parents: `(S,)` parent segment of each segment, `-1` for a root (TRL's `tree_segment_parents`). Segments must be
            numbered in DFS pre-order and laid out in that order, so `parents[s] < s`.
        n_tokens: tokens in the row. Pass it to avoid a host sync on `cu_seqlens[-1]`.
        cp_group: split the row over this group by context parallelism (see the module docstring). Every rank passes the
            whole row's tree and then its own contiguous `n_tokens / P` tokens of Q/K/V to the attention.
        sp_group: split the row over this group by Ulysses sequence parallelism, with the same inputs as `cp_group`.
    """
    if cp_group is not None and sp_group is not None:
        raise ValueError("pass one of cp_group and sp_group")
    cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
    parents = parents.to(torch.int32).contiguous()
    n_segments = parents.numel()
    if n_tokens is None:
        n_tokens = int(cu_seqlens[-1])
    device = cu_seqlens.device

    subtree_end = torch.empty(n_segments, dtype=torch.int32, device=device)
    pos_start = torch.empty(n_segments, dtype=torch.int32, device=device)
    with device_context(device):
        _tree_segment_info_kernel[(n_segments,)](cu_seqlens, parents, subtree_end, pos_start, n_segments, BLOCK=256)

    tokens = torch.arange(n_tokens, device=device, dtype=torch.int32)
    segment = torch.searchsorted(cu_seqlens, tokens, right=True) - 1
    positions = (pos_start[segment] + tokens - cu_seqlens[segment]).to(torch.int32)

    tiles = tuple(_segment_tiles(cu_seqlens, block, 0, n_tokens) for block in BLOCK_T_SIZES)
    query_start, query_tiles = 0, None
    group = cp_group if cp_group is not None else sp_group
    if group is not None:
        world, rank = torch.distributed.get_world_size(group), torch.distributed.get_rank(group)
        if n_tokens % world != 0:
            raise ValueError(f"a row of {n_tokens} tokens does not split evenly over {world} ranks; pad it")
        if cp_group is not None:
            shard = n_tokens // world
            query_start = rank * shard
            query_tiles = tuple(
                _segment_tiles(cu_seqlens, block, query_start, query_start + shard) for block in BLOCK_T_SIZES
            )
    return TreeAttentionMetadata(
        cu_seqlens=cu_seqlens,
        parents=parents,
        subtree_end=subtree_end,
        positions=positions,
        tiles=tiles,
        query_start=query_start,
        query_tiles=query_tiles,
        cp_group=cp_group,
        sp_group=sp_group,
    )


# ---------------------------------------------------------------------------------------------------------------------
# launch configs
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _pick_tiles(TILES32, TILES64, TILES128, BLOCK_T: tl.constexpr):
    if BLOCK_T == 32:
        return TILES32
    elif BLOCK_T == 64:
        return TILES64
    else:
        return TILES128


def _set_descriptor_blocks(nargs):
    """Autotune pre-hook: the streamed tiles of a config are `BLOCK_S` rows of one head (as `mlp.py` does)."""
    for name in ("K_desc", "V_desc", "Q_desc", "DO_desc"):
        desc = nargs.get(name)
        if TensorDescriptor is not None and isinstance(desc, TensorDescriptor):
            desc.block_shape = [nargs["BLOCK_S"], 1, nargs["BLOCK_D"]]


# LIGER_TREE_ATTENTION_AUTOTUNE=0 pins every kernel to `_DEFAULT_CONFIG`, skipping Triton's benchmarking of each
# config (as LIGER_FUSED_MOE_AUTOTUNE does for the fused MoE). Read at launch, so it can be set at any time.
_AUTOTUNE_ENV = "LIGER_TREE_ATTENTION_AUTOTUNE"
# (BLOCK_T, BLOCK_S, num_warps, num_stages) small enough for the shared memory of any backend, fp32 included
_DEFAULT_CONFIG = (32, 32, 4, 1)


def _get_autotune_configs(candidates):
    return [
        triton.Config(
            {"BLOCK_T": block_t, "BLOCK_S": block_s},
            num_warps=num_warps,
            num_stages=num_stages,
            pre_hook=_set_descriptor_blocks,
        )
        for block_t, block_s, num_warps, num_stages in candidates
    ]


def _autotune_space():
    """(BLOCK_T, BLOCK_S, num_warps, num_stages) the autotuner searches."""
    # The backward holds up to three fp32 `BLOCK_T x D` accumulators, so small tiles over many warps stay in the space.
    return tuple(
        (block_t, block_s, num_warps, num_stages)
        for block_t in BLOCK_T_SIZES
        for block_s in [16, 32, 64]
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3]
    )


def _candidates():
    if os.environ.get(_AUTOTUNE_ENV, "1").lower() in ("0", "false", "no"):
        return (_DEFAULT_CONFIG,)
    return _autotune_space()


@functools.cache
def _autotuned(kernel, candidates, key):
    return triton.autotune(configs=_get_autotune_configs(candidates), key=list(key))(kernel)


def _launcher(kernel, *key):
    """The kernel autotuned over the current candidates (one autotuner, and one cache, per candidate set)."""
    return _autotuned(kernel, _candidates(), ("HEAD_DIM", "G", "USE_WINDOW", "N_BUCKET", *key))


# ---------------------------------------------------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _load_rows2(
    A, B, A_desc, B_desc, start, offs, row_mask, head, offs_d, d_mask, stride_a, stride_b,
    USE_DESC: tl.constexpr, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
):  # fmt: skip
    """
    `BLOCK` rows of one head of two `(N, H, D)` tensors from row `start`. Through descriptors, rows past the range are
    other segments' (finite) rows or zero past `N`; callers mask them out of the scores, so they never contribute.
    """
    if USE_DESC:
        a = A_desc.load([start, head, 0]).reshape(BLOCK, BLOCK_D)
        b = B_desc.load([start, head, 0]).reshape(BLOCK, BLOCK_D)
    else:
        mask = row_mask[:, None] & d_mask[None, :]
        a = tl.load(A + offs[:, None].to(tl.int64) * stride_a + offs_d[None, :], mask=mask, other=0.0)
        b = tl.load(B + offs[:, None].to(tl.int64) * stride_b + offs_d[None, :], mask=mask, other=0.0)
    return a, b


@triton.jit
def _fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    pos_q,
    offs_m,
    K,
    V,
    K_desc,
    V_desc,
    h,
    POS,
    stride_kn,
    stride_vn,
    lo,
    hi,
    qk_scale,
    window,
    offs_d,
    d_mask,
    CAUSAL: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    USE_DESC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        k, v = _load_rows2(K, V, K_desc, V_desc, start_n, offs_n, n_mask, h, offs_d, d_mask, stride_kn, stride_vn,
                           USE_DESC, BLOCK_N, BLOCK_D)  # fmt: skip
        # raw scores; the scale is applied after subtracting the max (subtract-before-scale)
        s = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        valid = n_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        if USE_WINDOW:
            pos_k = tl.load(POS + offs_n, mask=n_mask, other=0)
            valid = valid & ((pos_q[:, None] - pos_k[None, :]) < window)
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        # keep the pivot finite for rows with no visible key yet (sliding window), so -inf - -inf never appears
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_safe) * qk_scale)
        p = tl.math.exp2((s - m_safe[:, None]) * qk_scale)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=DOT_PRECISION)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit(do_not_specialize=["window", "q_off", "q_end", "N_BUCKET"])
def _tree_attn_fwd_kernel(
    Q,
    K,
    V,
    K_desc,
    V_desc,
    Out,
    LSE,
    CU,
    PARENTS,
    POS,
    TILES32,
    TILES64,
    TILES128,
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_on,
    stride_oh,
    stride_lse_h,
    qk_scale,
    window,
    q_off,
    q_end,
    N_BUCKET,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    USE_DESC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Q, Out and LSE hold the queries [q_off, q_end) of the row (all of it without context parallelism); tokens are
    # numbered in the row, so a query's row in them is its token - q_off
    BLOCK_M: tl.constexpr = BLOCK_T
    TILES = _pick_tiles(TILES32, TILES64, TILES128, BLOCK_T)
    tile = tl.program_id(0)
    hq = tl.program_id(1)
    h = hq // G
    seg = tl.load(TILES + 2 * tile)
    if seg < 0:
        return
    q_start = tl.load(TILES + 2 * tile + 1)
    seg_start = tl.load(CU + seg)
    seg_end = tl.load(CU + seg + 1)

    q_hi = tl.minimum(seg_end, q_end)
    offs_m = q_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_hi
    rows = (offs_m - q_off).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    q = tl.load(
        Q + rows[:, None] * stride_qn + hq * stride_qh + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    if USE_WINDOW:
        pos_q = tl.load(POS + offs_m, mask=m_mask, other=0)
    else:
        pos_q = offs_m

    K += h * stride_kh
    V += h * stride_vh
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # own segment, causal
    lo = seg_start
    if USE_WINDOW:
        # positions are consecutive within a segment, so keys before q_start - window + 1 are never visible
        lo = tl.maximum(seg_start, q_start - window + 1)
    hi = tl.minimum(q_start + BLOCK_M, q_hi)
    acc, l_i, m_i = _fwd_inner(
        acc, l_i, m_i, q, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn, stride_vn, lo, hi, qk_scale,
        window, offs_d, d_mask, True, USE_WINDOW, USE_DESC, BLOCK_S, BLOCK_D, DOT_PRECISION,
    )  # fmt: skip

    # ancestors, fully visible
    a = tl.load(PARENTS + seg)
    while a >= 0:
        a_start = tl.load(CU + a)
        a_end = tl.load(CU + a + 1)
        if USE_WINDOW:
            # the nearest query of the tile is q_start; once an ancestor ends out of its window, all further ones do
            if tl.load(POS + q_start) - tl.load(POS + a_end - 1) >= window:
                a = -1
            else:
                acc, l_i, m_i = _fwd_inner(
                    acc, l_i, m_i, q, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn, stride_vn, a_start, a_end,
                    qk_scale, window, offs_d, d_mask, False, USE_WINDOW, USE_DESC, BLOCK_S, BLOCK_D,
                    DOT_PRECISION,
                )  # fmt: skip
                a = tl.load(PARENTS + a)
        else:
            acc, l_i, m_i = _fwd_inner(
                acc, l_i, m_i, q, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn, stride_vn, a_start, a_end,
                qk_scale, window, offs_d, d_mask, False, USE_WINDOW, USE_DESC, BLOCK_S, BLOCK_D,
                DOT_PRECISION,
            )  # fmt: skip
            a = tl.load(PARENTS + a)

    acc = acc / l_i[:, None]
    tl.store(
        Out + rows[:, None] * stride_on + hq * stride_oh + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )
    # log-sum-exp in raw score units: p = exp2((qk - lse) * qk_scale)
    tl.store(LSE + hq * stride_lse_h + rows, m_i + tl.math.log2(l_i) / qk_scale, mask=m_mask)


# ---------------------------------------------------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N"])
def _bwd_preprocess_kernel(
    Out,
    DO,
    DELTA,
    stride_on,
    stride_oh,
    stride_don,
    stride_doh,
    stride_delta_h,
    N,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hq = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < N) & (offs_d[None, :] < HEAD_DIM)
    o = tl.load(Out + offs_m[:, None].to(tl.int64) * stride_on + hq * stride_oh + offs_d[None, :], mask=mask, other=0.0)
    do = tl.load(
        DO + offs_m[:, None].to(tl.int64) * stride_don + hq * stride_doh + offs_d[None, :], mask=mask, other=0.0
    )
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(DELTA + hq * stride_delta_h + offs_m, delta, mask=offs_m < N)


@triton.jit
def _bwd_dq_inner(
    dq,
    b,
    rho,
    r_sum,
    q,
    do,
    lse,
    delta,
    pos_q,
    offs_m,
    K,
    V,
    K_desc,
    V_desc,
    h,
    POS,
    stride_kn,
    stride_vn,
    lo,
    hi,
    qk_scale,
    window,
    offs_d,
    d_mask,
    CAUSAL: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    ZERO_SUM: tl.constexpr,
    USE_DESC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        k, v = _load_rows2(K, V, K_desc, V_desc, start_n, offs_n, n_mask, h, offs_d, d_mask, stride_kn, stride_vn,
                           USE_DESC, BLOCK_N, BLOCK_D)  # fmt: skip
        s = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        valid = n_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        if USE_WINDOW:
            pos_k = tl.load(POS + offs_n, mask=n_mask, other=0)
            valid = valid & ((pos_q[:, None] - pos_k[None, :]) < window)
        p = tl.where(valid, tl.math.exp2((s - lse[:, None]) * qk_scale), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        t = (p * (dp - delta[:, None])).to(k.dtype)
        dq = tl.dot(t, k, dq, input_precision=DOT_PRECISION)
        if ZERO_SUM:
            r = p.to(k.dtype)
            rho += tl.sum(t.to(tl.float32), 1)
            r_sum += tl.sum(r.to(tl.float32), 1)
            b = tl.dot(r, k, b, input_precision=DOT_PRECISION)
    return dq, b, rho, r_sum


@triton.jit(do_not_specialize=["window", "q_off", "q_end", "N_BUCKET"])
def _tree_attn_bwd_dq_kernel(
    Q,
    K,
    V,
    K_desc,
    V_desc,
    DO,
    DQ,
    LSE,
    DELTA,
    LAMBDA,
    CU,
    PARENTS,
    POS,
    TILES32,
    TILES64,
    TILES128,
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_don,
    stride_doh,
    stride_dqn,
    stride_dqh,
    stride_lse_h,
    sm_scale,
    qk_scale,
    window,
    q_off,
    q_end,
    N_BUCKET,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    ZERO_SUM: tl.constexpr,
    USE_DESC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # query-side tensors (Q, dO, dQ, LSE, delta, lambda) hold the queries [q_off, q_end), as in the forward
    BLOCK_M: tl.constexpr = BLOCK_T
    TILES = _pick_tiles(TILES32, TILES64, TILES128, BLOCK_T)
    tile = tl.program_id(0)
    hq = tl.program_id(1)
    h = hq // G
    seg = tl.load(TILES + 2 * tile)
    if seg < 0:
        return
    q_start = tl.load(TILES + 2 * tile + 1)
    seg_start = tl.load(CU + seg)
    seg_end = tl.load(CU + seg + 1)

    q_hi = tl.minimum(seg_end, q_end)
    offs_m = q_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_hi
    rows = (offs_m - q_off).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    qd_mask = m_mask[:, None] & d_mask[None, :]

    q = tl.load(Q + rows[:, None] * stride_qn + hq * stride_qh + offs_d[None, :], mask=qd_mask, other=0.0)
    do = tl.load(DO + rows[:, None] * stride_don + hq * stride_doh + offs_d[None, :], mask=qd_mask, other=0.0).to(
        Q.dtype.element_ty
    )
    lse = tl.load(LSE + hq * stride_lse_h + rows, mask=m_mask, other=0.0)
    delta = tl.load(DELTA + hq * stride_lse_h + rows, mask=m_mask, other=0.0)
    if USE_WINDOW:
        pos_q = tl.load(POS + offs_m, mask=m_mask, other=0)
    else:
        pos_q = offs_m

    K += h * stride_kh
    V += h * stride_vh
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    b = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # r @ K, for the zero-sum projection
    rho = tl.zeros([BLOCK_M], dtype=tl.float32)  # row mass of the cast score gradient
    r_sum = tl.zeros([BLOCK_M], dtype=tl.float32)  # row mass of the cast probabilities

    lo = seg_start
    if USE_WINDOW:
        lo = tl.maximum(seg_start, q_start - window + 1)
    hi = tl.minimum(q_start + BLOCK_M, q_hi)
    dq, b, rho, r_sum = _bwd_dq_inner(
        dq, b, rho, r_sum, q, do, lse, delta, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn, stride_vn, lo, hi,
        qk_scale, window, offs_d, d_mask, True, USE_WINDOW, ZERO_SUM, USE_DESC, BLOCK_S, BLOCK_D,
        DOT_PRECISION,
    )  # fmt: skip

    a = tl.load(PARENTS + seg)
    while a >= 0:
        a_start = tl.load(CU + a)
        a_end = tl.load(CU + a + 1)
        if USE_WINDOW:
            if tl.load(POS + q_start) - tl.load(POS + a_end - 1) >= window:
                a = -1
            else:
                dq, b, rho, r_sum = _bwd_dq_inner(
                    dq, b, rho, r_sum, q, do, lse, delta, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn,
                    stride_vn, a_start, a_end, qk_scale, window, offs_d, d_mask, False, USE_WINDOW, ZERO_SUM, USE_DESC,
                    BLOCK_S, BLOCK_D, DOT_PRECISION,
                )  # fmt: skip
                a = tl.load(PARENTS + a)
        else:
            dq, b, rho, r_sum = _bwd_dq_inner(
                dq, b, rho, r_sum, q, do, lse, delta, pos_q, offs_m, K, V, K_desc, V_desc, h, POS, stride_kn,
                stride_vn, a_start, a_end, qk_scale, window, offs_d, d_mask, False, USE_WINDOW, ZERO_SUM, USE_DESC,
                BLOCK_S, BLOCK_D, DOT_PRECISION,
            )  # fmt: skip
            a = tl.load(PARENTS + a)

    if ZERO_SUM:
        lam = tl.where(r_sum > 0, rho / r_sum, 0.0)
        dq = dq - lam[:, None] * b
        tl.store(LAMBDA + hq * stride_lse_h + rows, lam, mask=m_mask)
    tl.store(
        DQ + rows[:, None] * stride_dqn + hq * stride_dqh + offs_d[None, :],
        (dq * sm_scale).to(DQ.dtype.element_ty),
        mask=qd_mask,
    )


@triton.jit
def _bwd_dkv_inner(
    dk,
    dv,
    c,
    k,
    v,
    pos_k,
    offs_n,
    n_mask,
    Q,
    DO,
    Q_desc,
    DO_desc,
    hq,
    LSE,
    DELTA,
    LAMBDA,
    POS,
    stride_qn,
    stride_don,
    lo,
    hi,
    q_off,
    qk_scale,
    window,
    offs_d,
    d_mask,
    CAUSAL: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    ZERO_SUM: tl.constexpr,
    USE_DESC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < hi
        rows = offs_m - q_off
        q, do = _load_rows2(Q, DO, Q_desc, DO_desc, start_m - q_off, rows, m_mask, hq, offs_d, d_mask, stride_qn,
                            stride_don, USE_DESC, BLOCK_M, BLOCK_D)  # fmt: skip
        do = do.to(q.dtype)
        lse = tl.load(LSE + rows, mask=m_mask, other=0.0)
        delta = tl.load(DELTA + rows, mask=m_mask, other=0.0)

        # transposed scores, [BLOCK_N, BLOCK_M]
        s = tl.dot(k, tl.trans(q), input_precision=DOT_PRECISION)
        valid = n_mask[:, None] & m_mask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])
        if USE_WINDOW:
            pos_q = tl.load(POS + offs_m, mask=m_mask, other=0)
            valid = valid & ((pos_q[None, :] - pos_k[:, None]) < window)
        p = tl.where(valid, tl.math.exp2((s - lse[None, :]) * qk_scale), 0.0)
        r = p.to(do.dtype)
        dv = tl.dot(r, do, dv, input_precision=DOT_PRECISION)
        dp = tl.dot(v, tl.trans(do), input_precision=DOT_PRECISION)
        t = (p * (dp - delta[None, :])).to(q.dtype)
        dk = tl.dot(t, q, dk, input_precision=DOT_PRECISION)
        if ZERO_SUM:
            # C += r^T z with z = cast(lambda * Q), lambda as computed by the dQ pass
            lam = tl.load(LAMBDA + rows, mask=m_mask, other=0.0)
            z = (lam[:, None] * q.to(tl.float32)).to(q.dtype)
            c = tl.dot(r, z, c, input_precision=DOT_PRECISION)
    return dk, dv, c


@triton.jit(do_not_specialize=["window", "q_off", "q_end", "N_BUCKET"])
def _tree_attn_bwd_dkv_kernel(
    Q,
    K,
    V,
    DO,
    Q_desc,
    DO_desc,
    DK,
    DV,
    LSE,
    DELTA,
    LAMBDA,
    CU,
    SUBTREE_END,
    POS,
    TILES32,
    TILES64,
    TILES128,
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_don,
    stride_doh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    stride_lse_h,
    sm_scale,
    qk_scale,
    window,
    q_off,
    q_end,
    N_BUCKET,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    ZERO_SUM: tl.constexpr,
    USE_DESC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # every K/V tile of the row, against only the queries [q_off, q_end) that the query-side tensors hold
    BLOCK_N: tl.constexpr = BLOCK_T
    TILES = _pick_tiles(TILES32, TILES64, TILES128, BLOCK_T)
    tile = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.load(TILES + 2 * tile)
    if seg < 0:
        return
    k_start = tl.load(TILES + 2 * tile + 1)
    seg_end = tl.load(CU + seg + 1)
    sub_end = tl.load(SUBTREE_END + seg)

    offs_n = k_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < seg_end
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    kd_mask = n_mask[:, None] & d_mask[None, :]

    k = tl.load(K + offs_n[:, None].to(tl.int64) * stride_kn + h * stride_kh + offs_d[None, :], mask=kd_mask, other=0.0)
    v = tl.load(V + offs_n[:, None].to(tl.int64) * stride_vn + h * stride_vh + offs_d[None, :], mask=kd_mask, other=0.0)
    if USE_WINDOW:
        pos_k = tl.load(POS + offs_n, mask=n_mask, other=0)
        # within the segment positions are consecutive: queries past the last key + window see none of the tile
        own_hi = tl.minimum(seg_end, k_start + BLOCK_N - 1 + window)
    else:
        pos_k = offs_n
        own_hi = seg_end
    own_lo = tl.maximum(k_start, q_off)
    own_hi = tl.minimum(own_hi, q_end)
    sub_lo = tl.maximum(seg_end, q_off)
    sub_end = tl.minimum(sub_end, q_end)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    c = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)  # zero-sum correction of dK
    # the query heads sharing this K/V head are reduced here, in registers
    for g in range(G):
        hq = h * G + g
        Qh = Q + hq * stride_qh  # pointer path; the descriptors take the head as a coordinate
        DOh = DO + hq * stride_doh
        LSEh = LSE + hq * stride_lse_h
        DELTAh = DELTA + hq * stride_lse_h
        LAMBDAh = LAMBDA + hq * stride_lse_h
        # own segment, causal: queries from the tile start on
        dk, dv, c = _bwd_dkv_inner(
            dk, dv, c, k, v, pos_k, offs_n, n_mask, Qh, DOh, Q_desc, DO_desc, hq, LSEh, DELTAh, LAMBDAh, POS, stride_qn,
            stride_don, own_lo, own_hi, q_off, qk_scale, window, offs_d, d_mask, True, USE_WINDOW, ZERO_SUM, USE_DESC,
            BLOCK_S, BLOCK_D, DOT_PRECISION,
        )  # fmt: skip
        # descendants: one contiguous pre-order range, every query sees the whole segment
        dk, dv, c = _bwd_dkv_inner(
            dk, dv, c, k, v, pos_k, offs_n, n_mask, Qh, DOh, Q_desc, DO_desc, hq, LSEh, DELTAh, LAMBDAh, POS, stride_qn,
            stride_don, sub_lo, sub_end, q_off, qk_scale, window, offs_d, d_mask, False, USE_WINDOW, ZERO_SUM, USE_DESC,
            BLOCK_S, BLOCK_D, DOT_PRECISION,
        )  # fmt: skip

    if ZERO_SUM:
        dk = dk - c
    tl.store(
        DK + offs_n[:, None].to(tl.int64) * stride_dkn + h * stride_dkh + offs_d[None, :],
        (dk * sm_scale).to(DK.dtype.element_ty),
        mask=kd_mask,
    )
    tl.store(
        DV + offs_n[:, None].to(tl.int64) * stride_dvn + h * stride_dvh + offs_d[None, :],
        dv.to(DV.dtype.element_ty),
        mask=kd_mask,
    )


# ---------------------------------------------------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------------------------------------------------


def _dot_precision(dtype: torch.dtype) -> Optional[str]:
    # fp32 inputs: exact fp32 products (no TF32). Half-precision inputs: the backend default, since the setting only
    # applies to fp32 operands and ROCm's Triton rejects "tf32" (it accepts "ieee", "bf16x3" and "bf16x6").
    return "ieee" if dtype == torch.float32 else None


def _tma_compatible(x: torch.Tensor) -> bool:
    """A descriptor needs a unit last stride and 16-byte aligned base and strides (as `mlp.py` checks)."""
    size = x.element_size()
    return x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and all((s * size) % 16 == 0 for s in x.stride()[:-1])


@functools.cache
def _has_tma(device_index: int) -> bool:
    # Triton also runs descriptor loads on GPUs without TMA, but as plain loads that are slower than masked pointer loads
    return not is_hip() and torch.cuda.get_device_capability(device_index)[0] >= 9


def _descriptors(*tensors):
    """
    Host descriptors over `(N, H, D)` tensors where they become TMA loads, else the tensors themselves: no
    `triton.tools.tensor_descriptor`, a GPU without TMA (or ROCm), or a tensor a descriptor cannot address. The block
    shape is set per autotune config by `_set_descriptor_blocks`.
    """
    device = tensors[0].device
    use = (
        TensorDescriptor is not None
        and _has_tma(device.index if device.index is not None else torch.cuda.current_device())
        and all(_tma_compatible(x) for x in tensors)
    )
    if not use:
        return False, tensors
    return True, tuple(
        TensorDescriptor(x, list(x.shape), list(x.stride()), [1, 1, triton.next_power_of_2(x.shape[-1])])
        for x in tensors
    )


def _grid(tiles: tuple, heads: int):
    n_tiles = dict(zip(BLOCK_T_SIZES, (t.shape[0] for t in tiles)))
    return lambda config: (n_tiles[config["BLOCK_T"]], heads)


def _query_tiles(meta: TreeAttentionMetadata) -> tuple:
    return meta.tiles if meta.query_tiles is None else meta.query_tiles


def tree_attention_forward(
    q,
    k,
    v,
    meta: TreeAttentionMetadata,
    sm_scale: float,
    window: Optional[int],
    output_dtype: Optional[torch.dtype] = None,
):
    """
    `q`: the queries of tokens `[meta.query_start, meta.query_start + len(q))` of the row; `k`, `v`: the whole row.
    Returns `o` and the log-sum-exp `(H_q, len(q))` of those queries.
    """
    n, hq, d = q.shape
    n_row, h = k.shape[0], k.shape[1]
    block_d = triton.next_power_of_2(d)
    o = torch.empty(n, hq, d, dtype=output_dtype or q.dtype, device=q.device)
    lse = torch.empty(hq, n, dtype=torch.float32, device=q.device)
    use_desc, (k_desc, v_desc) = _descriptors(k, v)
    tiles = _query_tiles(meta)
    with device_context(q.device):
        _launcher(_tree_attn_fwd_kernel)[_grid(tiles, hq)](
            q, k, v, k_desc, v_desc, o, lse, meta.cu_seqlens, meta.parents, meta.positions, *tiles,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1),
            lse.stride(0), sm_scale * LOG2E, window or 0, meta.query_start, meta.query_start + n,
            triton.next_power_of_2(n_row), G=hq // h, HEAD_DIM=d, BLOCK_D=block_d, USE_WINDOW=window is not None,
            USE_DESC=use_desc, DOT_PRECISION=_dot_precision(q.dtype),
        )  # fmt: skip
    return o, lse


def tree_attention_backward(
    do,
    q,
    k,
    v,
    o,
    lse,
    meta: TreeAttentionMetadata,
    sm_scale: float,
    window: Optional[int],
    zero_sum_correction: bool = True,
    dkv_dtype: Optional[torch.dtype] = None,
):
    """
    Gradients for the queries of [`tree_attention_forward`]: `dq` of those queries, and `dk`, `dv` of the whole row from
    those queries alone (in `dkv_dtype`, default `k.dtype`).
    """
    n, hq, d = q.shape
    n_row, h = k.shape[0], k.shape[1]
    block_d = triton.next_power_of_2(d)
    zero_sum = zero_sum_correction and q.dtype != torch.float32
    common = dict(
        G=hq // h,
        HEAD_DIM=d,
        BLOCK_D=block_d,
        USE_WINDOW=window is not None,
        ZERO_SUM=zero_sum,
        DOT_PRECISION=_dot_precision(q.dtype),
    )

    delta = torch.empty_like(lse)
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty(k.shape, dtype=dkv_dtype or k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=dkv_dtype or v.dtype, device=v.device)
    lam = torch.empty_like(lse) if zero_sum else lse  # LAMBDA is never read without ZERO_SUM
    qk_scale = sm_scale * LOG2E
    q_off, q_end = meta.query_start, meta.query_start + n
    n_bucket = triton.next_power_of_2(n_row)
    use_desc_kv, (k_desc, v_desc) = _descriptors(k, v)
    use_desc_q, (q_desc, do_desc) = _descriptors(q, do)
    query_tiles = _query_tiles(meta)
    with device_context(q.device):
        _bwd_preprocess_kernel[(triton.cdiv(n, 64), hq)](
            o, do, delta, o.stride(0), o.stride(1), do.stride(0), do.stride(1), delta.stride(0), n,
            HEAD_DIM=d, BLOCK_D=block_d, BLOCK_M=64,
        )  # fmt: skip
        # dQ first: it computes the zero-sum lambda that dK/dV reads
        _launcher(_tree_attn_bwd_dq_kernel, "ZERO_SUM")[_grid(query_tiles, hq)](
            q, k, v, k_desc, v_desc, do, dq, lse, delta, lam, meta.cu_seqlens, meta.parents, meta.positions,
            *query_tiles, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), do.stride(0),
            do.stride(1), dq.stride(0), dq.stride(1), lse.stride(0), sm_scale, qk_scale, window or 0, q_off, q_end,
            n_bucket, USE_DESC=use_desc_kv, **common,
        )  # fmt: skip
        _launcher(_tree_attn_bwd_dkv_kernel, "ZERO_SUM")[_grid(meta.tiles, h)](
            q, k, v, do, q_desc, do_desc, dk, dv, lse, delta, lam, meta.cu_seqlens, meta.subtree_end, meta.positions,
            *meta.tiles, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), do.stride(0),
            do.stride(1), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1), lse.stride(0), sm_scale, qk_scale,
            window or 0, q_off, q_end, n_bucket, USE_DESC=use_desc_q, **common,
        )  # fmt: skip
    return dq, dk, dv


# ---------------------------------------------------------------------------------------------------------------------
# context / sequence parallelism
# ---------------------------------------------------------------------------------------------------------------------


def _all_gather_tokens(x: torch.Tensor, group) -> torch.Tensor:
    """`(n, H, D)` shards of the ranks, in rank order -> `(P * n, H, D)`."""
    out = torch.empty(
        (torch.distributed.get_world_size(group) * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_tokens(x: torch.Tensor, group) -> torch.Tensor:
    """`(P * n, H, D)` partial sums -> this rank's `(n, H, D)` shard of their sum over the ranks."""
    out = torch.empty(
        (x.shape[0] // torch.distributed.get_world_size(group), *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    torch.distributed.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _tokens_to_heads(x: torch.Tensor, group) -> torch.Tensor:
    """Ulysses all-to-all: `(n, H, D)` with every head of this rank's tokens -> `(P * n, H / P, D)`, its heads."""
    world = torch.distributed.get_world_size(group)
    n, heads, d = x.shape
    send = x.reshape(n, world, heads // world, d).transpose(0, 1).contiguous()  # head group j goes to rank j
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    return recv.view(world * n, heads // world, d)  # rank i's tokens come i-th, so the row is in order


def _heads_to_tokens(x: torch.Tensor, group) -> torch.Tensor:
    """The inverse of [`_tokens_to_heads`]: `(P * n, H / P, D)` -> `(n, H, D)`."""
    world = torch.distributed.get_world_size(group)
    n_row, heads, d = x.shape
    recv = torch.empty_like(x, memory_format=torch.contiguous_format)
    torch.distributed.all_to_all_single(recv, x.contiguous(), group=group)
    return recv.view(world, n_row // world, heads, d).transpose(0, 1).reshape(n_row // world, world * heads, d)


def _require_unit_inner_stride(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


class LigerTreeAttentionFunction(torch.autograd.Function):
    """
    Tree attention over a prefix-tree-packed row (see the module docstring).

    Args:
        q: `(N, H_q, D)` queries of the packed row. Any strides with a unit last stride (e.g. a transposed HF view).
        k: `(N, H_kv, D)` keys, `H_q` a multiple of `H_kv`.
        v: `(N, H_kv, D)` values.
        meta: [`TreeAttentionMetadata`] from [`build_tree_attention_metadata`]. Built with a `cp_group` or `sp_group`,
            `q`, `k`, `v` are this rank's `N / P` tokens of the row, and so is the output.
        sm_scale: softmax scale, `1 / sqrt(D)` when `None`.
        window: sliding window in tree positions (a key at depth `j` is visible from depth `i` when `i - j < window`).
        output_dtype: `q.dtype` (default) or `torch.float32`. An fp32 output is saved as is, so backward's
            `delta = rowsum(dO * O)` sees no rounding of `O`, at twice the memory for the saved output.

    Returns:
        `(N, H_q, D)` attention output.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(ctx, q, k, v, meta: TreeAttentionMetadata, sm_scale=None, window=None, output_dtype=None):
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError(f"query heads ({q.shape[1]}) must be a multiple of key/value heads ({k.shape[1]})")
        if output_dtype not in (None, q.dtype, torch.float32):
            raise ValueError(f"output_dtype must be None, {q.dtype} or torch.float32, got {output_dtype}")
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        q, k, v = (_require_unit_inner_stride(x) for x in (q, k, v))
        ctx.meta, ctx.sm_scale, ctx.window, ctx.kv_repeat = meta, sm_scale, window, 1

        if meta.sp_group is not None:
            world = torch.distributed.get_world_size(meta.sp_group)
            if q.shape[1] % world != 0:
                raise ValueError(f"{q.shape[1]} query heads do not split over {world} ranks")
            if k.shape[1] % world != 0:
                if world % k.shape[1] != 0:
                    raise ValueError(f"{k.shape[1]} key/value heads neither split over nor divide {world} ranks")
                # fewer key/value heads than ranks: each rank takes a copy of the one its query heads share
                ctx.kv_repeat = world // k.shape[1]
                k, v = (x.repeat_interleave(ctx.kv_repeat, dim=1) for x in (k, v))
            q, k, v = (_tokens_to_heads(x, meta.sp_group) for x in (q, k, v))
            o, lse = tree_attention_forward(q, k, v, meta, sm_scale, window, output_dtype)
            ctx.save_for_backward(q, k, v, o, lse)
            return _heads_to_tokens(o, meta.sp_group)

        if meta.cp_group is not None:
            # save the local shards only; backward gathers K/V again
            k_row, v_row = (_all_gather_tokens(x, meta.cp_group) for x in (k, v))
            o, lse = tree_attention_forward(q, k_row, v_row, meta, sm_scale, window, output_dtype)
        else:
            o, lse = tree_attention_forward(q, k, v, meta, sm_scale, window, output_dtype)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        meta = ctx.meta
        do = _require_unit_inner_stride(do)
        if meta.sp_group is not None:
            do = _tokens_to_heads(do, meta.sp_group)
            dq, dk, dv = tree_attention_backward(do, q, k, v, o, lse, meta, ctx.sm_scale, ctx.window)
            dq, dk, dv = (_heads_to_tokens(x, meta.sp_group) for x in (dq, dk, dv))
            if ctx.kv_repeat > 1:
                dk, dv = (x.unflatten(1, (-1, ctx.kv_repeat)).sum(2) for x in (dk, dv))
        elif meta.cp_group is not None:
            k_row, v_row = (_all_gather_tokens(x, meta.cp_group) for x in (k, v))
            # this rank's queries' share of the whole row's dK/dV, summed over the ranks in fp32
            dq, dk, dv = tree_attention_backward(
                do, q, k_row, v_row, o, lse, meta, ctx.sm_scale, ctx.window, dkv_dtype=torch.float32
            )
            dk, dv = (_reduce_scatter_tokens(x, meta.cp_group).to(k.dtype) for x in (dk, dv))
        else:
            dq, dk, dv = tree_attention_backward(do, q, k, v, o, lse, meta, ctx.sm_scale, ctx.window)
        return dq, dk, dv, None, None, None, None
