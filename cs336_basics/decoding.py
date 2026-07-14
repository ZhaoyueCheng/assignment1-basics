"""
Text generation / decoding (CS336 Assignment 1, Section 6).

Autoregressively sample text from a trained TransformerLM, with temperature scaling
and top-p (nucleus) truncation. No provided pytest — validate by generating text
for the `generate` / `main_experiment` problems.
"""
from __future__ import annotations

import torch

from cs336_basics.transformer import TransformerLM


def filter_probs(
    probs: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Truncate a 1-D prob distribution (shape (vocab,)) with top-k and/or top-p, then
    renormalize. One mental model for both: FIND A THRESHOLD, ZERO EVERYTHING BELOW IT.

      - top-k: threshold = the k-th largest prob (torch.topk gives it directly).
      - top-p: sort desc, cumsum, count how many tokens are in the nucleus, then zero
               the rest by their ORIGINAL indices (no scatter needed for 1-D).

    We decode one sequence at a time, so probs is 1-D — that's what lets us index by
    integer indices directly. (For a batched (B, vocab) version you'd need scatter.)

    One sort serves both filters: each just decides how many of the top tokens to keep
    (`num_keep`), and we take the stricter (smaller) count.
    """
    probs = probs.clone()
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    num_keep = probs.shape[-1]                          # keep everything by default

    if top_k is not None:
        num_keep = min(num_keep, top_k)

    if top_p is not None:
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # keep tokens whose EXCLUSIVE prefix sum < p (always keeps top-1, includes the
        # token that crosses p).
        num_keep_p = int((cumsum - sorted_probs < top_p).sum())
        num_keep = min(num_keep, num_keep_p)

    probs[sorted_idx[num_keep:]] = 0.0                  # zero the rest via original idx
    return probs / probs.sum()


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    context_length: int | None = None,
) -> torch.Tensor:
    """Autoregressively sample a completion. `prompt` is a 1-D LongTensor of ids;
    returns prompt + generated ids. Call model.eval() at the call site.

    Test (no pytest): encode a prompt, generate, decode ids -> string, eyeball fluency.
    """
    model.eval()
    seq = prompt
    for _ in range(max_new_tokens):
        # 1. crop to the model's context window
        seq_cond = seq if context_length is None else seq[-context_length:]
        # 2. logits at the last position: (1, T, V) -> (V,)
        logits = model(seq_cond[None, :])[0, -1]
        # 3. temperature scale + softmax  (temperature <= 0 => greedy)
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            # 4. optional top-k / top-p truncation
            if top_k is not None or top_p is not None:
                probs = filter_probs(probs, top_k=top_k, top_p=top_p)
            # 5. sample one token
            next_id = torch.multinomial(probs, num_samples=1)   # (1,)
        # 6. append
        seq = torch.cat([seq, next_id])
        # 7. stop on EOS
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break
    return seq
