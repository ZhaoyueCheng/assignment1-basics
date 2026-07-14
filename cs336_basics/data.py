"""
Data loading (CS336 Assignment 1, Section 5.1).

Turns one long 1-D array of token ids into random (input, target) batches for
language-model training.
"""
from __future__ import annotations

import numpy.typing as npt
import torch
import numpy as np


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one training batch of next-token-prediction examples.

    `dataset` is a 1-D int array x = [x_0, x_1, ..., x_{n-1}] of token ids (often an
    np.memmap over a huge file — treat it as a normal array; don't copy the whole thing).

    Produce two LongTensors, each shape (batch_size, context_length):
      - inputs[b]  = x[i : i+context_length]
      - targets[b] = x[i+1 : i+context_length+1]   # inputs shifted left by one
    where each i is drawn independently and uniformly at random.

    Steps:
      - Valid start indices are 0 .. len(dataset) - context_length - 1 (inclusive), so
        that targets never runs off the end. Sample `batch_size` of them (np.random.randint
        or torch.randint).
      - Build the two index windows and gather. A clean trick: starts[:, None] +
        arange(context_length) gives a (batch_size, context_length) index matrix; index
        the dataset with it, then +1 for targets.
      - Convert to torch.long and move to `device`. For 'cuda' you can optionally use
        pin_memory + non_blocking, but plain .to(device) is fine and required for 'mps'/'cpu'.

    Test:  uv run pytest -k test_get_batch


    Loading the array elsewhere (train.py), NOT here:
      - Save tokenized data as a uint16 (vocab < 65536) numpy array with np.save.
      - Load with np.load(path, mmap_mode="r")  (or np.memmap(path, dtype=np.uint16,
        mode="r")) so you never pull the whole corpus into RAM.
      - Sanity-check that max(id) < vocab_size after loading.
    """
    batch_start_inds = np.random.randint(low=0, high=len(dataset) - context_length, size=batch_size)
    inputs, targets = [], []
    for start_ind in batch_start_inds:
        inputs.append(dataset[start_ind:start_ind+context_length])
        targets.append(dataset[start_ind+1:start_ind+1+context_length])
    
    return torch.LongTensor(inputs), torch.LongTensor(targets)