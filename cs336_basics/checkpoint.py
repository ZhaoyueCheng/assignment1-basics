"""
Checkpointing (CS336 Assignment 1, Section 5.2).

Save/restore everything needed to resume a training run: model weights, optimizer
state (AdamW's moments), and the iteration counter.
"""
from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Dump model + optimizer + iteration to `out` (a path or open binary file object).

    Build one dict and torch.save it, e.g.:
      { "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration }
    torch.save accepts either a filesystem path or a file-like object for `out`, so just
    pass `out` straight through.

    Test:  uv run pytest -k test_checkpointing
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a checkpoint saved by save_checkpoint; return the saved iteration number.

    Steps:
      - obj = torch.load(src)  (map_location="cpu" is a safe default; the training loop
        moves the model to its device afterward, or pass the device you want).
      - model.load_state_dict(obj["model"])
      - optimizer.load_state_dict(obj["optimizer"])
      - return obj["iteration"]

    Restoring optimizer state matters: without it AdamW's moment estimates reset and
    training briefly destabilizes on resume.

    Test:  uv run pytest -k test_checkpointing
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
