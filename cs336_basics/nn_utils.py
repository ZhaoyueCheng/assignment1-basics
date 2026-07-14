"""
Loss and gradient utilities (CS336 Assignment 1, Section 4.1 and 4.5).

Same boilerplate convention as transformer.py: each function's docstring tells
you what to build (math, shapes, numerical-stability steps) but not the code.
Fill in the bodies yourself, then run the pytest command in the docstring.

softmax already lives in transformer.py (Section 3.4.4); import and reuse it.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor
import torch.nn.functional as F


def cross_entropy(
    inputs: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
) -> Float[Tensor, ""]:
    """Average cross-entropy (negative log-likelihood) between logits and target ids.

    For one example with logit vector o (length vocab_size) and target index x, the
    loss is   -log softmax(o)[x]  =  -( o[x] - log sum_a exp(o[a]) ).

    Do NOT call softmax then log — that is numerically unstable. Instead:
      - Subtract max(o) over the vocab dim for stability (log-sum-exp trick). The max
        cancels out mathematically but keeps exp() from overflowing.
      - Compute log_sum_exp = log( sum_a exp(o[a] - max) ) + max  (or use the shifted
        form directly and let the max cancel).
      - The loss is  log_sum_exp - o[x]  (the "cancel log and exp" step: you never
        materialize the full softmax).
      - `targets` selects the correct-class logit o[x] per row; torch.gather or advanced
        indexing does this.

    Shapes: inputs may carry arbitrary leading/batch dims before the final vocab_size
    dim; targets carries the matching leading dims. Batch-like dims come FIRST, the
    vocab dim is LAST. Reduce (mean) over ALL batch dims and return a scalar tensor.

    Test:  uv run pytest -k test_cross_entropy
    """
    max_logits = torch.max(inputs, dim=-1).values
    logsumexp = torch.log(torch.sum(torch.exp(inputs - max_logits.unsqueeze(-1)), dim=-1)).squeeze(-1)
    targets_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - max_logits
    
    return torch.mean(-(targets_logits - logsumexp))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the GLOBAL l2 norm of all parameter gradients to at most max_l2_norm, in place.

    "Global" means you treat every parameter's .grad as if flattened and concatenated
    into one giant vector, and clip that single combined norm — not each tensor alone.

    Steps:
      - Collect p.grad for every parameter that has one (skip p.grad is None).
      - total_norm = sqrt( sum over params of sum(g**2) )  — the l2 norm of everything.
      - If total_norm <= max_l2_norm: do nothing.
      - Else scale every grad in place by  max_l2_norm / (total_norm + eps), with
        eps = 1e-6. Use g.mul_(scale) or g *= scale so the update is in place (the
        optimizer must see the modified .grad afterward).

    Returns nothing; it mutates the gradients.

    Test:  uv run pytest -k test_gradient_clipping
    """
    eps = 10**(-6)
    grads = [p.grad for p in parameters if p.grad is not None]
    grads_l2_norm = torch.sqrt(torch.sum(torch.stack([torch.square(g) for g in grads])))

    if grads_l2_norm > max_l2_norm:
      for p in parameters:
          if p.grad is None:
              continue
          
          p.grad = p.grad * max_l2_norm / (grads_l2_norm + eps)
          