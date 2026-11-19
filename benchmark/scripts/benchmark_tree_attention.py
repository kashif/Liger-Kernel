"""
Tree attention over a prefix-tree-packed row: Liger's kernel against TRL's `tree_attention` (gather every segment's
root-to-segment K/V with `index_select`, then one FlashAttention varlen call, here `torch.nn.attention.varlen`).

A row is one GRPO group in TRL's tree layout: a prompt shared by `G` rollouts, each a chain of multi-turn segments.
The x-axis is `G`, so the packing ratio (raw / unique tokens) and the gathered K/V (sum of root-to-segment paths) grow
with it while the unique tokens grow only by the completions.
"""

import math

import torch

from torch.nn.attention.varlen import varlen_attn
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.tree_attention import LigerTreeAttentionFunction
from liger_kernel.ops.tree_attention import build_tree_attention_metadata
from liger_kernel.utils import infer_device

device = infer_device()

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def grpo_tree(n_rollouts, prompt_len, turn_len, n_turns):
    """TRL's DFS layout of one prompt and `n_rollouts` rollouts of `n_turns` turns (each turn ends a sample)."""
    lengths, parents = [prompt_len], [-1]
    for _ in range(n_rollouts):
        parent = 0
        for _ in range(n_turns):
            lengths.append(turn_len)
            parents.append(parent)
            parent = len(parents) - 1
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    paths, kv_index, cu_k = [], [], [0]
    for s, p in enumerate(parents):
        paths.append((paths[p] if p >= 0 else []) + list(range(cu[s], cu[s + 1])))
        kv_index.extend(paths[-1])
        cu_k.append(len(kv_index))
    return cu, parents, kv_index, cu_k


class TRLGatherTreeAttention(torch.nn.Module):
    """TRL's `tree_attention`: gather K/V along every segment's path, then one causal varlen attention call."""

    def __init__(self, cu, cu_k, kv_index, hq, hkv):
        super().__init__()
        self.cu = torch.tensor(cu, dtype=torch.int32, device=device)
        self.cu_k = torch.tensor(cu_k, dtype=torch.int32, device=device)
        self.kv_index = torch.tensor(kv_index, dtype=torch.long, device=device)
        self.max_q = max(b - a for a, b in zip(cu, cu[1:]))
        self.max_k = max(b - a for a, b in zip(cu_k, cu_k[1:]))
        self.hq, self.hkv = hq, hkv

    def forward(self, qkv):
        q, k, v = qkv.split([self.hq, self.hkv, self.hkv], dim=1)
        if torch.version.hip and flash_attn_varlen_func is not None:
            # on ROCm TRL runs FlashAttention through aiter; flash-attn's AMD backend stands in for it
            return flash_attn_varlen_func(
                q.contiguous(),
                k.index_select(0, self.kv_index),
                v.index_select(0, self.kv_index),
                self.cu,
                self.cu_k,
                self.max_q,
                self.max_k,
                softmax_scale=1.0 / math.sqrt(q.shape[-1]),
                causal=True,
            )
        return varlen_attn(
            q.contiguous(),
            k.index_select(0, self.kv_index),
            v.index_select(0, self.kv_index),
            self.cu,
            self.cu_k,
            self.max_q,
            self.max_k,
            scale=1.0 / math.sqrt(q.shape[-1]),
            window_size=(-1, 0),
            enable_gqa=self.hq != self.hkv,
        )


class LigerTreeAttention(torch.nn.Module):
    def __init__(self, cu, parents, hq, hkv):
        super().__init__()
        self.meta = build_tree_attention_metadata(
            torch.tensor(cu, device=device), torch.tensor(parents, device=device), n_tokens=cu[-1]
        )
        self.hq, self.hkv = hq, hkv

    def forward(self, qkv):
        q, k, v = qkv.split([self.hq, self.hkv, self.hkv], dim=1)
        return LigerTreeAttentionFunction.apply(q, k, v, self.meta)


def setup_tree_attention(input):
    cfg = input.extra_benchmark_config
    hq, hkv, head_dim = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    cu, parents, kv_index, cu_k = grpo_tree(input.x, cfg["prompt_len"], cfg["turn_len"], cfg["n_turns"])
    if input.kernel_provider == "liger":
        layer = LigerTreeAttention(cu, parents, hq, hkv)
    elif input.kernel_provider == "trl_gather":
        layer = TRLGatherTreeAttention(cu, cu_k, kv_index, hq, hkv)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for tree attention")
    qkv = torch.randn(cu[-1], hq + 2 * hkv, head_dim, device=device, dtype=cfg["dtype"], requires_grad=True)
    return qkv, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "tree_attention",
        "x_name": "G",
        "x_label": "rollouts per prompt",
        "x_values": [1, 2, 4, 8, 16],
        "kernel_providers": ["liger", "trl_gather"],
        "extra_benchmark_configs": [
            {
                # Qwen3-8B attention, a 1k-token prompt and three 256-token turns per rollout
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "prompt_len": 1024,
                "turn_len": 256,
                "n_turns": 3,
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_tree_attention),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_tree_attention),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
