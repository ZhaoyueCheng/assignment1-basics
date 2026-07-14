"""
Training loop / experiment driver (CS336 Assignment 1, Sections 5.3 and 7).

This is a SCAFFOLD: argparse config, the loop skeleton, and TODO markers where you
plug in the components you built in the other modules. It is infrastructure (not a
graded algorithm), so it's spelled out more than the other files — but the training
step itself is left for you to wire up.

The default hyperparameters below are the TinyStories settings from Section 7.2.1 so
you don't have to go back to the PDF:
  vocab_size 10000 | context_length 256 | d_model 512 | d_ff 1344 | num_layers 4
  num_heads 16 | rope_theta 10000 | total tokens processed ~= 327,680,000
  (batch_size * total_steps * context_length ≈ 327.68M)

Low-resource (CPU/MPS): drop total tokens to ~40M and target val loss <= 2.0 instead
of 1.45. See Section 7.2.3 tips (torch.compile, no TF32 on mps).

Run:  uv run python -m cs336_basics.train --train-path ... --val-path ... [flags]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data: paths to tokenized uint16 .npy files (produced by your tokenizer offline).
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path", required=True)
    # Model architecture (TinyStories defaults, Section 7.2.1).
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # Optimizer / schedule (tune these — problems `learning_rate`, `batch_size_experiment`).
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--lr-max", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Infra.
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--checkpoint-path", default="data/checkpoints/model.pt")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--compile", action="store_true")
    # Ablation switches for Section 7.3 — thread these into TransformerLM as you add them.
    p.add_argument("--no-rmsnorm", action="store_true", help="layer_norm_ablation")
    p.add_argument("--post-norm", action="store_true", help="pre_norm_ablation")
    p.add_argument("--no-pos-emb", action="store_true", help="no_pos_emb (NoPE)")
    p.add_argument("--ffn", choices=["swiglu", "silu"], default="swiglu", help="swiglu_ablation")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_data, args) -> float:
    """Mean per-token cross-entropy over a few random val batches. Put model in eval()
    mode, sum losses over `args.eval_batches` batches from get_batch(val_data, ...),
    return the average. Remember model.train() afterward."""
    raise NotImplementedError


def main() -> None:
    args = parse_args()

    # 1. Load tokenized data memory-mapped (don't read the whole corpus into RAM).
    train_data = np.load(args.train_path, mmap_mode="r")   # dtype must match how
    val_data   = np.load(args.val_path,   mmap_mode="r")   # you saved it (uint16).

    # 2. Build the model and move it to the device.
    model = TransformerLM(vocab_size=..., context_length=..., d_model=..., num_layers=...,
                            num_heads=..., d_ff=..., rope_theta=...).to(args.device)
    # Pass the ablation flags through once your model supports them.
    # if args.compile: model = torch.compile(model)   # backend="aot_eager" on mps

    # 3. Build the optimizer with the AdamW hyperparameters.
    optimizer = AdamW(model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2),
                        eps=args.eps, weight_decay=args.weight_decay)

    # 4. Optionally resume: start_step = load_checkpoint(args.resume_from, model, optimizer)
    start_step = 0

    # 5. Training loop.
    t0 = time.time()
    for step in range(start_step, args.total_steps):
        # a. Set this step's LR from the cosine schedule and write it into the optimizer.
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min,
                                    args.warmup_steps, args.total_steps)
        for g in optimizer.param_groups: g["lr"] = lr

        # b. Sample a batch, forward, loss.
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)                       # (B, T, vocab)
        loss = cross_entropy(logits, y)         # cross_entropy reduces over B and T

        # c. Backward + clip + step.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # d. Periodic eval + logging (log step, wall-clock time.time()-t0, train loss,
        #    val loss — to console and/or wandb; problem `experiment_log`).
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args)
            print(...)

        # e. Periodic checkpoint.
        if step % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
        pass

    # 6. Final checkpoint.
    save_checkpoint(model, optimizer, args.total_steps, args.checkpoint_path)


if __name__ == "__main__":
    main()
