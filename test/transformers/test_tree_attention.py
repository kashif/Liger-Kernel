import math
import random

import pytest
import torch
import torch.nn.functional as F
import triton

from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import tree_attention as tree_attention_ops
from liger_kernel.ops.tree_attention import LigerTreeAttentionFunction
from liger_kernel.ops.tree_attention import build_tree_attention_metadata
from liger_kernel.ops.tree_attention import tree_attention_backward
from liger_kernel.ops.tree_attention import tree_attention_forward
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


@pytest.fixture(autouse=True)
def pinned_config(monkeypatch):
    """Run on the one portable config; `test_every_autotune_config_matches_fp64` covers the rest of the space."""
    monkeypatch.setenv(tree_attention_ops._AUTOTUNE_ENV, "0")


# TRL's `TestTreePacking` rows, packed by `TreeRow.pack` (a trie node is keyed on its token and the next one, so the
# samples ending in 202 and continuing with 202 401 fork one token early):
#   input_ids    [101 102 | 103 201 | 202 | 202 401 | 103 301 302]
#   position_ids [  0   1 |   2   3 |   4 |   4   5 |   2   3   4]
TRL_TREE = ([0, 2, 4, 5, 7, 10], [-1, 0, 1, 1, 0])
TRL_POSITIONS = [0, 1, 2, 3, 4, 4, 5, 2, 3, 4]

# Relative L2 error against the fp64 reference, for the output and every input gradient.
TOLERANCE = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.5e-2}
DTYPES = [
    torch.float32,
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
    ),
]


def random_tree(n_segments, max_len, seed):
    """A random prefix forest numbered in DFS pre-order (`parents[s] < s`), with many 1- and 2-token segments."""
    rng = random.Random(seed)
    lengths, parents, path = [], [], []
    for s in range(n_segments):
        if s == 0 or rng.random() < 0.15:
            path = []  # start a new root
        else:
            path = path[: rng.randrange(len(path)) + 1]  # branch off the current root path
        parents.append(path[-1] if path else -1)
        lengths.append(rng.choice([1, 2, rng.randint(1, max_len), rng.randint(1, max_len)]))
        path.append(s)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return cu, parents


def tree_mask(cu, parents, window=None):
    """Dense visibility of the tree: own segment causally, ancestor segments fully, optionally a window on depth."""
    cu, parents = torch.tensor(cu), torch.tensor(parents)
    n_tokens, n_segments = int(cu[-1]), parents.numel()
    segment = torch.searchsorted(cu, torch.arange(n_tokens), right=True) - 1
    is_ancestor = torch.eye(n_segments, dtype=torch.bool)  # is_ancestor[s, a]: a is s or one of its ancestors
    depth_start = torch.zeros(n_segments, dtype=torch.long)
    for s, p in enumerate(parents.tolist()):
        if p >= 0:
            is_ancestor[s] |= is_ancestor[p]
            depth_start[s] = depth_start[p] + cu[p + 1] - cu[p]
    token = torch.arange(n_tokens)
    mask = is_ancestor[segment[:, None], segment[None, :]] & (token[None, :] <= token[:, None])
    positions = depth_start[segment] + token - cu[segment]
    if window is not None:
        mask &= (positions[:, None] - positions[None, :]) < window
    return mask, positions


def leaf_paths(cu, parents):
    """Packed token indices of each root-to-leaf path, keyed by leaf segment: the samples of the row."""
    paths = {}
    for leaf in set(range(len(parents))) - set(parents):
        path, s = [], leaf
        while s >= 0:
            path.append(s)
            s = parents[s]
        paths[leaf] = torch.cat([torch.arange(cu[s], cu[s + 1]) for s in reversed(path)])
    return paths


def masked_attention(q, k, v, mask, scale):
    """Reference attention in (N, H, D) layout; K/V heads are repeated for GQA."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = torch.einsum("nhd,mhd->hnm", q, k) * scale
    return torch.einsum("hnm,mhd->nhd", scores.masked_fill(~mask, float("-inf")).softmax(-1), v)


def make_inputs(n_tokens, hq, hkv, head_dim, dtype, key_offset=0.0, hf_layout=False):
    def make(heads, offset=0.0):
        if hf_layout:  # HF hands (1, H, N, D); transposed to (N, H, D) it is a view with a non-unit token stride
            x = torch.randn(1, heads, n_tokens, head_dim, device=device).transpose(1, 2)[0]
        else:
            x = torch.randn(n_tokens, heads, head_dim, device=device)
        return (x + offset).to(dtype)

    return make(hq), make(hkv, key_offset), make(hkv), make(hq)


def relative_error(x, reference):
    # floored at an RMS of 0.1 (real outputs and gradients here are 0.3-1.5), so an exactly-zero gradient, as when every
    # query sees only itself, gets an absolute bound instead of a division by zero
    return ((x.double() - reference).norm() / reference.norm().clamp_min(0.1 * reference.numel() ** 0.5)).item()


def check_against_fp64(
    cu, parents, hq, hkv, head_dim, dtype, window=None, hf_layout=False, key_offset=2.0, output_dtype=None
):
    """Liger tree attention forward and backward vs fp64 dense-mask attention on the same inputs."""
    n_tokens = cu[-1]
    q, k, v, do = make_inputs(n_tokens, hq, hkv, head_dim, dtype, key_offset, hf_layout)
    scale = 1.0 / math.sqrt(head_dim)
    meta = build_tree_attention_metadata(
        torch.tensor(cu, device=device), torch.tensor(parents, device=device), n_tokens=n_tokens
    )
    mask, positions = tree_mask(cu, parents, window)
    assert meta.positions.tolist() == positions.tolist()

    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
    actual = LigerTreeAttentionFunction.apply(*inputs, meta, scale, window, output_dtype)
    assert actual.dtype == (output_dtype or dtype)
    actual.backward(do.to(actual.dtype))
    reference = [x.detach().double().requires_grad_() for x in (q, k, v)]
    expected = masked_attention(*reference, mask.to(device), scale)
    expected.backward(do.double())

    for name, x, ref in [("o", actual, expected.detach())] + [
        (name, x.grad, ref.grad) for name, x, ref in zip(("dq", "dk", "dv"), inputs, reference)
    ]:
        error = relative_error(x, ref)
        assert error < TOLERANCE[dtype], f"{name}: relative error {error:.2e} >= {TOLERANCE[dtype]:.0e}"


@pytest.mark.parametrize(
    "n_segments, max_len",
    [
        (1, 300),  # one sample: plain causal attention
        (12, 90),  # branchy forest, segment boundaries off the tile grid
        (40, 9),  # many short segments, as when samples fork one token before they diverge
    ],
)
@pytest.mark.parametrize("hq, hkv", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [64, 80, 128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness(n_segments, max_len, hq, hkv, head_dim, dtype):
    cu, parents = random_tree(n_segments, max_len, seed=n_segments)
    cu = [0] + [c + 16 for c in cu[1:]]  # a 16-token first root, so no row is a lone token
    check_against_fp64(cu, parents, hq, hkv, head_dim, dtype)


@pytest.mark.parametrize("descriptors", [True, False], ids=["default-loads", "pointer-loads"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness_trl_packed_row(dtype, descriptors, monkeypatch):
    if not descriptors:  # TMA descriptor loads on sm90+ by default; force the pointer path everywhere else takes
        monkeypatch.setattr(tree_attention_ops, "TensorDescriptor", None)
    check_against_fp64(*TRL_TREE, hq=4, hkv=2, head_dim=64, dtype=dtype)
    _, positions = tree_mask(*TRL_TREE)
    assert positions.tolist() == TRL_POSITIONS


@pytest.mark.parametrize("window", [1, 7, 64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness_sliding_window(window, dtype):
    check_against_fp64(*random_tree(16, 80, seed=window), hq=4, hkv=2, head_dim=64, dtype=dtype, window=window)


@pytest.mark.parametrize("dtype", DTYPES[1:])
def test_correctness_fp32_output(dtype):
    check_against_fp64(*random_tree(12, 90, seed=4), hq=8, hkv=2, head_dim=64, dtype=dtype, output_dtype=torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness_hf_layout(dtype):
    check_against_fp64(*random_tree(10, 70, seed=3), hq=4, hkv=2, head_dim=64, dtype=dtype, hf_layout=True)


def test_each_leaf_matches_its_sample_forwarded_alone():
    """Every root-to-leaf path, run on its own as a causal sequence through SDPA, gives the leaf's packed outputs."""
    cu, parents = random_tree(15, 60, seed=11)
    q, k, v, _ = make_inputs(cu[-1], 4, 2, 64, torch.float32)
    meta = build_tree_attention_metadata(torch.tensor(cu, device=device), torch.tensor(parents, device=device))
    out = LigerTreeAttentionFunction.apply(q, k, v, meta, None, None)

    for leaf, tokens in leaf_paths(cu, parents).items():
        tokens = tokens.to(device)
        alone = F.scaled_dot_product_attention(
            *(x[tokens].transpose(0, 1) for x in (q, k, v)), is_causal=True, enable_gqa=True
        ).transpose(0, 1)
        n_leaf = cu[leaf + 1] - cu[leaf]
        torch.testing.assert_close(out[cu[leaf] : cu[leaf + 1]], alone[-n_leaf:], atol=1e-5, rtol=1e-5)


def test_backward_is_deterministic():
    cu, parents = random_tree(20, 50, seed=7)
    q, k, v, do = make_inputs(cu[-1], 8, 2, 64, torch.bfloat16 if supports_bfloat16() else torch.float16)
    meta = build_tree_attention_metadata(torch.tensor(cu, device=device), torch.tensor(parents, device=device))
    out, lse = tree_attention_forward(q, k, v, meta, 0.125, None)
    first, second = (tree_attention_backward(do, q, k, v, out, lse, meta, 0.125, None) for _ in range(2))
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device")
def test_zero_sum_projection_removes_bf16_gradient_leak():
    """
    With keys far from the origin and sharp attention, casting dS = P * (dP - delta) to bf16 leaves a nonzero row mass
    that reaches dQ and dK through the key/query means (Chen et al., arXiv:2609.34272). The projection removes it.
    """
    cu, parents = random_tree(8, 120, seed=5)
    n_tokens, head_dim, scale = cu[-1], 64, 0.125
    q = (torch.randn(n_tokens, 4, head_dim, device=device) * 3).to(torch.bfloat16)
    k = (torch.randn(n_tokens, 4, head_dim, device=device) * 0.5 + 8.0).to(torch.bfloat16)
    v, do = (torch.randn(n_tokens, 4, head_dim, device=device).to(torch.bfloat16) for _ in range(2))
    meta = build_tree_attention_metadata(torch.tensor(cu, device=device), torch.tensor(parents, device=device))

    reference = [x.double().requires_grad_() for x in (q, k, v)]
    masked_attention(*reference, tree_mask(cu, parents)[0].to(device), scale).backward(do.double())
    out, lse = tree_attention_forward(q, k, v, meta, scale, None)
    errors = {}
    for projected in (True, False):
        dq, dk, _ = tree_attention_backward(do, q, k, v, out, lse, meta, scale, None, zero_sum_correction=projected)
        errors[projected] = (relative_error(dq, reference[0].grad), relative_error(dk, reference[1].grad))

    assert max(errors[True]) < TOLERANCE[torch.bfloat16], errors
    assert errors[True][0] < 0.5 * errors[False][0], errors


def tiny_llama():
    from transformers import LlamaConfig
    from transformers import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).to(device)


def packed_batch(packing, vocab_size):
    """
    `(input_ids, forward kwargs, samples)` for one kind of input; each sample is `(row, packed token indices)`.

    - tree: a TRL tree-packed row, samples its root-to-leaf paths
    - flattened: sequences packed end to end with transformers' `cu_seq_lens_q/k`, as DataCollatorWithFlattening does
    - batch: an unpadded `(B, L)` batch, samples its rows
    """
    if packing == "batch":
        input_ids = torch.randint(0, vocab_size, (3, 50), device=device)
        return input_ids, {}, [(row, torch.arange(50, device=device)) for row in range(3)]
    if packing == "tree":
        cu, parents = random_tree(12, 40, seed=2)
        _, positions = tree_mask(cu, parents)
        kwargs = {
            "tree_cu_seqlens_q": torch.tensor(cu, device=device),
            "tree_segment_parents": torch.tensor(parents, device=device),
        }
        samples = [(0, tokens.to(device)) for tokens in leaf_paths(cu, parents).values()]
    else:
        lengths = [37, 64, 5, 90]
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
        positions = torch.cat([torch.arange(length) for length in lengths])
        kwargs = {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": max(lengths), "max_length_k": max(lengths)}
        samples = [(0, torch.arange(cu[i], cu[i + 1], device=device)) for i in range(len(lengths))]
    input_ids = torch.randint(0, vocab_size, (1, len(positions)), device=device)
    return input_ids, {"position_ids": positions[None].to(device), **kwargs}, samples


@pytest.mark.parametrize("packing", ["tree", "flattened", "batch"])
def test_hf_drop_in_matches_each_sample_forwarded_alone(packing):
    """
    Swapping a tiny Llama's attention for `"liger_tree"` gives, for a tree-packed row, a flattened batch and a plain
    batch alike, every sample's logits and the summed loss gradients of forwarding the samples one by one through SDPA.
    """
    from liger_kernel.transformers import register_liger_tree_attention

    model = tiny_llama()
    input_ids, kwargs, samples = packed_batch(packing, model.config.vocab_size)

    def grads():
        out = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        return out

    model.set_attn_implementation(register_liger_tree_attention())
    logits = model(input_ids, **kwargs).logits
    sum(logits[row, tokens].square().mean() for row, tokens in samples).backward()
    liger_grads = grads()

    model.set_attn_implementation("sdpa")
    loss = 0
    for row, tokens in samples:
        alone = model(input_ids[row : row + 1, tokens]).logits[0]
        torch.testing.assert_close(logits[row, tokens], alone, atol=1e-4, rtol=1e-4)
        loss = loss + alone.square().mean()
    loss.backward()
    for name, grad in grads().items():
        torch.testing.assert_close(liger_grads[name], grad, atol=1e-4, rtol=1e-3, msg=name)


def test_hf_drop_in_generates_like_sdpa():
    """Prefill runs tree attention and each cached decoding step SDPA; greedy decoding matches SDPA's tokens."""
    from liger_kernel.transformers import register_liger_tree_attention

    model = tiny_llama().eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 20), device=device)
    generated = {}
    for implementation in (register_liger_tree_attention(), "sdpa"):
        model.set_attn_implementation(implementation)
        generated[implementation] = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=8, do_sample=False, pad_token_id=0
        )
    assert torch.equal(*generated.values())


def test_hf_drop_in_refuses_padded_batches():
    from liger_kernel.transformers import register_liger_tree_attention

    model = tiny_llama()
    model.set_attn_implementation(register_liger_tree_attention())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :4] = 0
    with pytest.raises(NotImplementedError, match="padded batches"):
        model(input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("config", tree_attention_ops._autotune_space(), ids=lambda c: "T{}-S{}-w{}-s{}".format(*c))
def test_every_autotune_config_matches_fp64(config, monkeypatch):
    """Whichever config the autotuner picks on a GPU computes the same attention and gradients."""
    monkeypatch.setattr(tree_attention_ops, "_DEFAULT_CONFIG", config)
    dtype = torch.bfloat16 if supports_bfloat16() else torch.float16
    try:
        check_against_fp64(*random_tree(12, 90, seed=12), hq=8, hkv=2, head_dim=128, dtype=dtype)
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"{config} does not fit this GPU; the autotuner skips it too")
