"""
Optimizer and learning-rate schedule (CS336 Assignment 1, Sections 4.2-4.4).

Boilerplate convention matches transformer.py: docstrings describe what to build;
you fill in the bodies. The SGD class below is the WORKED EXAMPLE from the PDF
(Section 4.2.1) — it is fully implemented so you can study the torch.optim.Optimizer
API (param_groups, self.state, step). Use it as the template for AdamW.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """WORKED EXAMPLE (already implemented) — SGD with a 1/sqrt(t+1) decaying step.

    Study the three API pieces you'll reuse in AdamW:
      - __init__ passes a `defaults` dict of hyperparameters to super().__init__.
      - step() loops over self.param_groups, then each p in group["params"].
      - self.state[p] is a per-parameter dict you use to stash running state (here, t).

    This is not a graded deliverable; it's here for reference.
    """

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer (Loshchilov & Hutter 2019, decoupled weight decay).

    Per parameter, keep two running moment estimates (same shape as the param):
      m — first moment (mean of gradients)
      v — second moment (mean of squared gradients)
    Both start at 0. Also track the step count t, starting effectively at 1 on the
    first update (the PDF's algorithm indexes t from 1).

    __init__:
      - Signature like AdamW(params, lr, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01).
        (betas is a (beta1, beta2) tuple.) Validate lr >= 0 if you like.
      - Store lr, betas, eps, weight_decay in the `defaults` dict and call super().

    step(): for each param group (read lr, beta1/beta2, eps, weight_decay from `group`)
    and each p with a gradient g = p.grad.data, apply — in this order (matches the PDF):
      1. Read state: m, v (init to zeros_like(p) on first step), and t (init 1).
      2. alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)   # bias-correction, folded
                                                              # into the step size.
      3. Decoupled weight decay FIRST:  p.data -= lr * weight_decay * p.data
         (note: uses raw lr, NOT alpha_t, and is independent of the gradient — this is
          the "W" in AdamW).
      4. m = beta1*m + (1-beta1)*g
      5. v = beta2*v + (1-beta2)*g*g
      6. p.data -= alpha_t * m / (sqrt(v) + eps)
      7. Write m, v back to state and store t+1.
    Do updates in place on p.data. Return the closure loss like SGD does.

    Wire tests/adapters.py: get_adamw_cls() should `return AdamW`.
    Test:  uv run pytest -k test_adamw
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group.get("lr")
            (beta_1, beta_2) = group.get("betas")
            eps = group.get("eps")
            weight_decay = group.get("weight_decay")

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))

                alpha_t = lr * math.sqrt(1-beta_2**t) / (1-beta_1**t)
                p.data -= lr * weight_decay * p.data
                m = beta_1 * m + (1-beta_1) * grad
                v = beta_2 * v + (1-beta_2) * (grad**2)
                p.data -= alpha_t*m/(torch.sqrt(v)+eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine-annealing LR schedule with linear warmup (LLaMA-style). Pure function of `it`.

    Let a_max = max_learning_rate, a_min = min_learning_rate, Tw = warmup_iters,
    Tc = cosine_cycle_iters. Return the learning rate for iteration `it`:

      - Warmup   (it < Tw):          a = (it / Tw) * a_max          # linear ramp 0 -> a_max
      - Cosine   (Tw <= it <= Tc):   a = a_min + 0.5 * (1 + cos(pi * (it - Tw)/(Tc - Tw)))
                                              * (a_max - a_min)      # a_max -> a_min
      - Post     (it > Tc):          a = a_min                      # flat floor

    No torch needed — plain math.cos is fine. Watch the Tw == 0 edge case if you divide.
    You call this each step in the training loop and write the result into
    optimizer.param_groups[i]["lr"] before optimizer.step().

    Test:  uv run pytest -k test_get_lr_cosine_schedule
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 1/2 * (1+math.cos((it-warmup_iters) / (cosine_cycle_iters-warmup_iters) * math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate


# =============================================================================
# Problem (adamw_accounting): Resource accounting for training with AdamW (2 pts)
# Written deliverable. All tensors are float32 (4 bytes each).
#
# Symbols:
#   B = batch_size            T = context_length        V = vocab_size
#   L = num_layers            D = d_model               H = num_heads
#   d_ff = 8/3 * D            (given)
#
# Our architecture per the assignment: untied token-embedding and LM-head (each
# V*D), pre-norm blocks, SwiGLU FFN (3 matrices), RoPE (parameter-free), no biases.
#
# -----------------------------------------------------------------------------
# (a) Peak memory, decomposed into parameters / gradients / optimizer state /
#     activations. (Counts are in number-of-float32-VALUES; multiply by 4 for bytes.)
#
# PARAMETERS  P:
#   token embedding + LM head : 2 * V * D
#   per block:
#       attention q,k,v,o proj : 4 * D^2
#       SwiGLU W1,W2,W3        : 3 * D * d_ff = 3 * D * (8/3 D) = 8 * D^2
#       2 RMSNorm gains        : 2 * D
#     -> per block             : 12 * D^2 + 2 * D
#   final RMSNorm             : D
#   ------------------------------------------------------------------
#   P = 2*V*D  +  L*(12*D^2 + 2*D)  +  D
#
# GRADIENTS:      one per parameter                       = P
# OPTIMIZER STATE: AdamW keeps m and v per parameter      = 2 * P
#   => parameters + gradients + optimizer state = 4 * P values (= 16*P bytes)
#
# ACTIVATIONS  A  (only the components the problem lists are counted):
#   Per transformer block:
#       2 RMSNorm outputs            : 2 * B*T*D
#       Q,K,V projections            : 3 * B*T*D
#       QK^T scores                  : 1 * B*H*T^2
#       softmax(scores)              : 1 * B*H*T^2
#       weighted sum over V          : 1 * B*T*D
#       output projection            : 1 * B*T*D
#       FFN: W1, SiLU(gate), gate*val, W3 : 4 * B*T*d_ff = 4 * B*T*(8/3 D) = (32/3)*B*T*D
#       FFN: W2 output               : 1 * B*T*D
#     -> BTD terms: (2+3+1+1+1) = 8*B*T*D ; plus (32/3)*B*T*D ; plus 2*B*H*T^2
#     -> per block = (56/3)*B*T*D + 2*B*H*T^2
#   final RMSNorm                    : B*T*D
#   output embedding (logits)        : B*T*V
#   cross-entropy on logits (softmax): B*T*V
#   ------------------------------------------------------------------
#   A = L * ( (56/3)*B*T*D + 2*B*H*T^2 )  +  B*T*D  +  2*B*T*V
#
# TOTAL peak memory (bytes) = 4 * (4*P + A) = 16*P + 4*A.
#
# -----------------------------------------------------------------------------
# (b) GPT-2 XL: L=48, D=1600, H=25, V=50257, T=1024.
#
#   P = 2*V*D + L*(12*D^2 + 2*D) + D = 1,635,537,600 ≈ 1.636e9 params.
#   Batch-independent term  b = 16*P = 26,168,601,600 bytes ≈ 26.17 GB
#       (params 4P + grads 4P + opt state 8P).
#
#   Per-batch activation term (bytes):
#     a = 4 * [ L*((56/3)*T*D + 2*H*T^2) + T*D + 2*T*V ]
#       = 4 * 4,089,153,536 = 16,356,614,144 bytes ≈ 16.36 GB / batch element.
#       (The 2*B*H*T^2 attention-score term dominates: T^2 = ~1.05e6.)
#
#   Memory model:  M(B) = 16.36 * B + 26.17   (GB)
#   Fit in 80 GB:  16.36*B + 26.17 <= 80  ->  B <= (80 - 26.17)/16.36 ≈ 3.29
#   => MAX BATCH SIZE = 3.
#
# -----------------------------------------------------------------------------
# (c) FLOPs per AdamW step:
#   Every op is elementwise over the P parameters (independent of B and T):
#     weight decay  p -= (lr*wd)*p          ~2 FLOPs/param
#     m = b1*m + (1-b1)*g                   ~3
#     v = b2*v + (1-b2)*g*g                 ~4
#     p -= alpha_t * m/(sqrt(v)+eps)        ~5   (counting sqrt, div as 1 each)
#   => ~14 FLOPs per parameter, i.e. on the order of 14*P ≈ O(P) FLOPs per step.
#   This is negligible next to the forward+backward pass, which is O(P * tokens).
#
# -----------------------------------------------------------------------------
# (d) Training-time estimate (MFU):
#   Forward FLOPs ≈ 2*P per token; backward = 2x forward; total ≈ 6*P per token
#   (Kaplan/Hoffmann 6*N*D rule).
#     tokens/step = B*T = 1024*1024 = 1,048,576
#     total tokens D = 1,048,576 * 400,000 = 4.194e11
#     total FLOPs = 6 * P * D = 6 * 1.636e9 * 4.194e11 ≈ 4.12e21 FLOPs
#   Effective throughput = 495 TFLOP/s * 0.50 MFU = 2.475e14 FLOP/s
#     time = 4.12e21 / 2.475e14 ≈ 1.663e7 s ≈ 4,620 hours ≈ 192 days on one H100.
# =============================================================================
