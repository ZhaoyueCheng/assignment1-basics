# CS336 Assignment 1 — Section 4 onward (working guide)

Everything you need so you don't have to reopen the PDF. Code goes in the stub files
listed below (rich docstrings tell you *what* to build; you write the bodies). The
test adapters in `tests/adapters.py` are already wired to these functions.

Run the full suite: `uv run pytest`. Run one problem: `uv run pytest -k <name>`.

| File | Covers |
|---|---|
| `cs336_basics/nn_utils.py` | cross-entropy, gradient clipping |
| `cs336_basics/optimizer.py` | SGD (worked example), AdamW, cosine LR schedule |
| `cs336_basics/data.py` | `get_batch` data loader |
| `cs336_basics/checkpoint.py` | save/load checkpoint |
| `cs336_basics/decoding.py` | text generation (temperature, top-p) |
| `cs336_basics/train.py` | training loop / experiment driver (scaffold) |

---

## 4. Training a Transformer LM

### 4.1 `cross_entropy` (1 pt) — `nn_utils.py`
Compute `-log softmax(o)[x]` averaged over the batch. Subtract the max logit for
stability, cancel the log/exp (use log-sum-exp; never materialize softmax), reduce over
all leading batch dims. Vocab dim is last, batch dims first.
**Test:** `uv run pytest -k test_cross_entropy`
*Also note (used at eval time): perplexity = `exp(mean cross-entropy)`.*

### 4.2 SGD — no deliverable code
`SGD` in `optimizer.py` is the fully-worked example. Study the `torch.optim.Optimizer`
API there (`param_groups`, `self.state[p]`, `step`).
**Written deliverable (`learning_rate_tuning`, 1 pt):** run the toy loop (minimize
`(weights**2).mean()`) for 10 steps at lr = 1e1, 1e2, 1e3. Report what happens — one
value decays fastest, larger ones diverge. Write 1–2 sentences on what you observed.

### 4.3 `AdamW` (2 pts) — `optimizer.py`
Subclass `torch.optim.Optimizer`. Per-param state: first moment `m`, second moment `v`
(both zero-init), step `t` (from 1). Order per step: bias-correct step size
`alpha_t = lr·√(1-β2ᵗ)/(1-β1ᵗ)`; decoupled weight decay `p -= lr·wd·p`; update `m`, `v`;
then `p -= alpha_t·m/(√v+ε)`. Defaults betas `(0.9, 0.95)`, eps `1e-8`, wd `0.01`.
**Test:** `uv run pytest -k test_adamw`

**Written deliverable (`adamw_accounting`, 2 pts):** with `d_ff = 8/3·d_model`, float32:
- (a) Peak memory = params + activations + gradients + optimizer state. Params & grads
  each = #params·4 bytes; AdamW state = 2·#params·4 bytes (m and v); activations scale
  with `batch_size·context_length` across the components the PDF lists (RMSNorms, QKV,
  QKᵀ, softmax, attn·V, out proj, FFN branches, final norm, logits). Give an algebraic
  expression in `batch_size, vocab_size, context_length, num_layers, d_model, num_heads`.
- (b) Plug in GPT-2 XL (48 layers, d_model 1600, 25 heads, vocab 50257, ctx 1024) →
  reduce to `a·batch_size + b`; solve for max batch_size under 80 GB.
- (c) FLOPs of one AdamW step ≈ a small constant × #params (it's a handful of
  element-wise ops per parameter — O(#params), tiny next to the forward/backward).
- (d) Training-time estimate: forward FLOPs ≈ `2·#params·tokens`; backward = 2× forward,
  so total ≈ `6·#params·tokens_per_step·steps`. With H100 peak 495 TFLOP/s × 0.5 MFU,
  400K steps, batch 1024 → hours = total_FLOPs / (0.5·495e12) / 3600.

### 4.4 `get_lr_cosine_schedule` (1 pt) — `optimizer.py`
Pure function of `it`. Linear warmup to `a_max` over `Tw`; cosine decay `a_max→a_min`
between `Tw` and `Tc`; flat `a_min` after. Formula in the docstring.
**Test:** `uv run pytest -k test_get_lr_cosine_schedule`

### 4.5 `gradient_clipping` (1 pt) — `nn_utils.py`
Clip the *global* l2 norm across all grads to `max_l2_norm` (scale factor
`max/(norm+1e-6)`), in place.
**Test:** `uv run pytest -k test_gradient_clipping`

---

## 5. Training loop

### 5.1 `get_batch` (2 pts) — `data.py`
Sample random windows: `inputs = x[i:i+ctx]`, `targets = x[i+1:i+ctx+1]`, shape
`(batch, ctx)`, on the requested device. Load the underlying array with `mmap_mode="r"`
in `train.py`, not here.
**Test:** `uv run pytest -k test_get_batch`

### 5.2 checkpointing (1 pt) — `checkpoint.py`
`save_checkpoint`: `torch.save({model, optimizer, iteration}, out)`.
`load_checkpoint`: `torch.load`, restore both `state_dict`s, return the iteration.
**Test:** `uv run pytest -k test_checkpointing`

### 5.3 training loop (4 pts) — `train.py`
Scaffold provided (argparse + loop skeleton with TODOs). Wire: memmap data → build
`TransformerLM` → `AdamW` → per-step set LR from schedule, `get_batch`, forward,
`cross_entropy`, `zero_grad`/`backward`/`gradient_clipping`/`step`, periodic eval +
checkpoint + logging (steps **and** wall-clock). No pytest — validated by the Section 7
runs.
**Run:** `uv run python -m cs336_basics.train --train-path ... --val-path ...`

---

## 6. Generating text — `decoding.py`

### `decoding` (3 pts)
Autoregressive `generate`: crop to last `context_length` tokens, take last-position
logits, temperature-scale, softmax, optional top-p (nucleus) truncation + renormalize,
`multinomial` sample, append, stop on `<|endoftext|>` or `max_new_tokens`.
No pytest — validate by decoding samples (used in `generate` / `main_experiment`).

---

## 7. Experiments (run via `train.py`; deliverables are curves + writeups)

**Setup — TinyStories defaults (already the argparse defaults):** vocab 10000, ctx 256,
d_model 512, d_ff 1344, 4 layers, 16 heads, rope θ 10000, ≈327.68M total tokens
(`batch·steps·ctx`). Correct+efficient ⇒ ~20–30 min on 1×B200.
**Low-resource (CPU/MPS):** cut to ~40M tokens, target val loss ≤ 2.0 (not 1.45); e.g.
`batch 32 × 5000 steps × ctx 256`. On mps don't set TF32; optionally
`torch.compile(model, backend="aot_eager")`. Make the cosine schedule end (`Tc`) at your
final step.

- **`experiment_log` (3 pts):** logging infra (console and/or wandb) + a log doc of what
  you tried.
- **`learning_rate` (3 pts):** LR sweep, learning curves; hit val loss ≤ 1.45 (or ≤ 2.0
  low-resource). Then push LR up to find divergence ("edge of stability").
- **`batch_size_experiment` (1 pt):** vary batch 1 → memory limit (incl. 64, 128);
  re-tune LR; curves + discussion.
- **`generate` (1 pt):** ≥256 tokens of sample text + comment on fluency and ≥2 factors
  affecting quality.
- **Ablations (`train.py` flags):**
  - `layer_norm_ablation` (1 pt) `--no-rmsnorm`: remove RMSNorm; curve at old best LR +
    curve at a stable lower LR; comment.
  - `pre_norm_ablation` (1 pt) `--post-norm`: post-norm vs pre-norm curves.
  - `no_pos_emb` (1 pt) `--no-pos-emb`: NoPE vs RoPE curves.
  - `swiglu_ablation` (1 pt) `--ffn silu`: SiLU FFN with `d_ff = 4·d_model` (to match
    param count) vs SwiGLU; curves + discussion.
    *(These flags exist in the scaffold; you still have to thread them into `TransformerLM`.)*
- **`main_experiment` (2 pts):** train on OpenWebText, same arch/steps; curve + why loss
  differs from TinyStories; generated text + why quality is worse.
- **`leaderboard` (6 pts):** your own mods; ≤45 min on B200, OWT data only; beat 5.0 loss.
  Ideas: weight tying (input/output embeddings), Llama/Qwen tricks, nanoGPT speedrun.

---

## Test cheat-sheet
```
uv run pytest                              # everything
uv run pytest -k test_cross_entropy
uv run pytest -k test_gradient_clipping
uv run pytest -k test_adamw
uv run pytest -k test_get_lr_cosine_schedule
uv run pytest -k test_get_batch
uv run pytest -k test_checkpointing
```
Sections 6 (decoding) and 7 (experiments) have no unit tests — validate by running
`train.py` and inspecting learning curves / generated text.
