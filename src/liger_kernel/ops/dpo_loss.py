import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _selective_log_softmax_kernel(
    LOGITS,
    INPUT_IDS,
    LOG_P,
    MASK,
    TEMPERATURE,
    stride_input_ids_b,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 4096,
):
    """Compute log probabilities for selected tokens (adapted from GRPO)"""
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l
    LOG_P += off_b * L + off_l

    if MASK is not None:
        MASK += off_b * stride_input_ids_b + off_l
        not_skip = tl.load(MASK)
        if not_skip == 0:
            return

    # Compute LSE using stable numerical method (from GRPO)
    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    # Get log probability for target token
    ids = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    logp = x - lse
    tl.store(LOG_P, logp)


@torch.no_grad
def fused_selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0, mask=None):
    """Compute log probabilities for selected tokens without grad (adapted from GRPO)"""
    assert logits.is_contiguous()
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
    kwargs = {"BLOCK_N": 2048, "num_stages": 4, "num_warps": 1}
    _selective_log_softmax_kernel[(B, L)](
        logits, input_ids, log_p, mask, temperature, input_ids.stride(0), L, N, **kwargs
    )
    return log_p


@triton.jit
def _dpo_loss_fwd_kernel(
    LOGITS,
    REF_LOGITS,
    INPUT_IDS,
    COMPLETION_MASK,
    CHOSEN_LOGPS,
    REJECTED_LOGPS,
    LSE,
    TEMPERATURE,
    USE_REF_MODEL: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 4096,
):
    """DPO loss forward kernel (adapted from GRPO pattern)"""
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    # Check completion mask
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return

    # Set up pointers
    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    LSE += off_b * L + off_l

    # Compute LSE for policy model (stable numerical method from GRPO)
    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    # Get log probability for target token
    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse

    # Reference model computation
    ref_logp = 0.0
    if USE_REF_MODEL:
        REF_LOGITS += off_b * (L + 1) * N + off_l * N

        # Compute LSE for reference model
        ref_m_i = float("-inf")
        ref_l_i = 0.0
        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            ref_logits = tl.load(REF_LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
            new_ref_m_i = tl.maximum(ref_m_i, tl.max(ref_logits))
            ref_alpha = tl.exp(ref_m_i - new_ref_m_i)
            ref_l_i = ref_l_i * ref_alpha + tl.sum(tl.exp(ref_logits - new_ref_m_i))
            ref_m_i = new_ref_m_i
        ref_lse = ref_m_i + tl.log(ref_l_i)

        ref_x = tl.load(REF_LOGITS + idx).to(tl.float32) / TEMPERATURE
        ref_logp = ref_x - ref_lse

    # Compute log ratio
    log_ratio = logp - ref_logp

    # Store LSE for backward pass
    tl.store(LSE, lse)

    # Determine if this is chosen or rejected and accumulate
    # Use batch index to determine pair assignment
    pair_idx = off_b // 2
    is_chosen = off_b % 2 == 0

    if is_chosen:
        CHOSEN_LOGPS += pair_idx
        tl.atomic_add(CHOSEN_LOGPS, log_ratio)
    else:
        REJECTED_LOGPS += pair_idx
        tl.atomic_add(REJECTED_LOGPS, log_ratio)


class DPOLossFunction(torch.autograd.Function):
    """Memory-efficient Triton-based DPO Loss Function (following GRPO pattern)"""

    @staticmethod
    def forward(
        ctx,
        logits,
        ref_logits,
        input_ids,
        completion_mask,
        beta=0.1,
        loss_type="sigmoid",
        use_ref_model=True,
        temperature=1.0,
    ):
        """
        Args:
            logits: Policy model logits [B, L+1, V] where B is even (chosen + rejected pairs)
            ref_logits: Reference model logits [B, L+1, V] (can be None)
            input_ids: Target token IDs [B, L]
            completion_mask: Mask for completion tokens [B, L]
            beta: Temperature parameter
            loss_type: Type of loss ("sigmoid", "apo_zero", "apo_down")
            use_ref_model: Whether to use reference model
            temperature: Temperature for logits
        """
        assert logits.is_contiguous() and input_ids.is_contiguous()
        assert completion_mask is None or completion_mask.is_contiguous()

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        assert B % 2 == 0, "Batch size must be even (chosen + rejected pairs)"
        N_PAIRS = B // 2

        # Handle reference logits
        if ref_logits is None or not use_ref_model:
            ref_logits = torch.zeros_like(logits)
            use_ref_model = False
        assert ref_logits.is_contiguous()

        # Prepare outputs
        chosen_logps = torch.zeros(N_PAIRS, device=logits.device, dtype=torch.float32)
        rejected_logps = torch.zeros(N_PAIRS, device=logits.device, dtype=torch.float32)
        lse = torch.zeros(B, L, device=logits.device, dtype=torch.float32)

        # Launch kernel (following GRPO pattern exactly)
        kwargs = {"BLOCK_N": 2048, "num_stages": 2, "num_warps": 1}
        _dpo_loss_fwd_kernel[(B, L)](
            logits,
            ref_logits,
            input_ids,
            completion_mask,
            chosen_logps,
            rejected_logps,
            lse,
            temperature,
            use_ref_model,  # USE_REF_MODEL: tl.constexpr
            L,  # L: tl.constexpr
            N,  # N: tl.constexpr
            **kwargs,
        )

        # Compute rewards
        chosen_rewards = beta * chosen_logps
        rejected_rewards = beta * rejected_logps

        # Compute final loss based on loss type
        if loss_type == "sigmoid":
            logits_diff = chosen_rewards - rejected_rewards
            final_loss = -F.logsigmoid(logits_diff).mean()
        elif loss_type == "apo_zero":
            losses_chosen = 1 - F.sigmoid(chosen_rewards)
            losses_rejected = F.sigmoid(rejected_rewards)
            final_loss = (losses_chosen + losses_rejected).mean()
        elif loss_type == "apo_down":
            losses_chosen = F.sigmoid(chosen_rewards)
            losses_rejected = 1 - F.sigmoid(chosen_rewards - rejected_rewards)
            final_loss = (losses_chosen + losses_rejected).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Save for backward
        ctx.save_for_backward(logits, ref_logits, input_ids, completion_mask, lse)
        ctx.infos = (beta, loss_type, use_ref_model, temperature)
        ctx.chosen_rewards = chosen_rewards
        ctx.rejected_rewards = rejected_rewards

        return final_loss, chosen_rewards, rejected_rewards

    @staticmethod
    def backward(ctx, grad_output, grad_chosen_rewards, grad_rejected_rewards):
        """Backward pass using PyTorch autograd (simpler than Triton backward kernel)"""
        logits, ref_logits, input_ids, completion_mask, lse = ctx.saved_tensors
        beta, loss_type, use_ref_model, temperature = ctx.infos

        # Create output tensor for gradients
        dlogits = torch.zeros_like(logits)

        # Use PyTorch autograd for gradient computation
        if logits.requires_grad:
            with torch.enable_grad():
                logits_copy = logits.detach().requires_grad_(True)

                # Recompute forward pass for gradient computation
                log_probs = fused_selective_log_softmax(logits_copy, input_ids, temperature, completion_mask)

                # Handle reference model
                if use_ref_model and ref_logits is not None:
                    ref_log_probs = fused_selective_log_softmax(
                        ref_logits.detach(), input_ids, temperature, completion_mask
                    )
                else:
                    ref_log_probs = torch.zeros_like(log_probs)

                # Compute sequence-level log probabilities
                if completion_mask is not None:
                    mask = completion_mask.float()
                    seq_log_probs = (log_probs * mask).sum(dim=-1)
                    ref_seq_log_probs = (ref_log_probs * mask).sum(dim=-1)
                else:
                    seq_log_probs = log_probs.sum(dim=-1)
                    ref_seq_log_probs = ref_log_probs.sum(dim=-1)

                # Split into chosen/rejected
                chosen_log_probs = seq_log_probs[::2]
                rejected_log_probs = seq_log_probs[1::2]
                ref_chosen_log_probs = ref_seq_log_probs[::2]
                ref_rejected_log_probs = ref_seq_log_probs[1::2]

                # Compute rewards
                chosen_logratios = chosen_log_probs - ref_chosen_log_probs
                rejected_logratios = rejected_log_probs - ref_rejected_log_probs
                chosen_rewards = beta * chosen_logratios
                rejected_rewards = beta * rejected_logratios

                # Compute loss
                if loss_type == "sigmoid":
                    logits_diff = chosen_rewards - rejected_rewards
                    loss = -F.logsigmoid(logits_diff).mean()
                elif loss_type == "apo_zero":
                    losses_chosen = 1 - F.sigmoid(chosen_rewards)
                    losses_rejected = F.sigmoid(rejected_rewards)
                    loss = (losses_chosen + losses_rejected).mean()
                elif loss_type == "apo_down":
                    losses_chosen = F.sigmoid(chosen_rewards)
                    losses_rejected = 1 - F.sigmoid(chosen_rewards - rejected_rewards)
                    loss = (losses_chosen + losses_rejected).mean()

                # Compute gradients
                if loss.requires_grad:
                    dlogits = torch.autograd.grad(loss, logits_copy, grad_output, retain_graph=False)[0]

        return dlogits, None, None, None, None, None, None, None


def triton_dpo_loss(
    logits,
    ref_logits=None,
    input_ids=None,
    completion_mask=None,
    beta=0.1,
    loss_type="sigmoid",
    use_ref_model=True,
    temperature=1.0,
):
    """
    Memory-efficient Triton-based DPO loss computation.

    Args:
        logits: Policy model logits [B, L+1, V] where B is even (chosen + rejected pairs)
        ref_logits: Reference model logits [B, L+1, V] (optional)
        input_ids: Target token IDs [B, L]
        completion_mask: Mask for completion tokens [B, L] (optional)
        beta: Temperature parameter
        loss_type: Type of loss ("sigmoid", "apo_zero", "apo_down")
        use_ref_model: Whether to use reference model
        temperature: Temperature for logits

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards)
    """
    return DPOLossFunction.apply(
        logits,
        ref_logits,
        input_ids,
        completion_mask,
        beta,
        loss_type,
        use_ref_model,
        temperature,
    )
