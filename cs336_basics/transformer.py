"""
Transformer language model built from scratch (CS336 Assignment 1, Section 3).

You implement each module below. Each docstring describes, in words, what to
build and how — the math, the shapes, and the steps — but not the code itself.
Every module's docstring ends with the pytest command that checks it.

Attribute names matter: they must match the reference state_dict keys so that
`load_state_dict` works in tests/adapters.py. Keep these names:
  Linear                    -> self.weight                (shape: out_features, in_features)
  Embedding                 -> self.weight                (shape: num_embeddings, d_model)
  RMSNorm                   -> self.weight                (shape: d_model)          # the gain g
  SwiGLU                    -> self.w1, self.w2, self.w3  (each a Linear)
  MultiHeadSelfAttention    -> self.q_proj, self.k_proj, self.v_proj, self.output_proj  (each a Linear)
  TransformerBlock          -> self.ln1, self.attn, self.ln2, self.ffn
  TransformerLM             -> self.token_embeddings, self.layers, self.ln_final, self.lm_head

Recommended: use einops.einsum / rearrange for readable, batch-tolerant tensor ops.

Run the whole module test file with:  uv run pytest tests/test_model.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor


# ---------------------------------------------------------------------------
# 3.3.2  Linear
# ---------------------------------------------------------------------------
class Linear(nn.Module):
    """A linear (matrix-multiply) layer with NO bias, computing y = W x. Following most
    modern LLMs, there is no bias term.

    Guidance:
      - __init__: store one parameter `self.weight` holding W of shape
        (out_features, in_features) — store W itself, NOT its transpose. Initialize it
        in place with a truncated normal: mean 0, standard deviation
        sigma = sqrt(2 / (in_features + out_features)), truncated to [-3*sigma, 3*sigma]
        (torch.nn.init.trunc_normal_ takes mean, std, and bounds a and b).
      - forward: x is (..., in_features) with any number of leading batch-like
        dimensions. Produce (..., out_features) by contracting x's last dimension
        (in_features) against W's second dimension, leaving W's first dimension
        (out_features) as the new last dimension. This is the row-major form y = x W^T,
        but as a single contraction it needs no explicit transpose; leading dims pass
        through untouched.

    Test:  uv run pytest -k test_linear
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.trunc_normal_(tensor=self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# ---------------------------------------------------------------------------
# 3.3.3  Embedding
# ---------------------------------------------------------------------------
class Embedding(nn.Module):
    """A learned lookup table mapping integer token ids to dense vectors. This is the
    Transformer's first layer, turning a (batch, seq) grid of ids into a (batch, seq,
    d_model) grid of vectors.

    Guidance:
      - __init__: store one parameter `self.weight` of shape
        (num_embeddings, embedding_dim) — one row per vocabulary entry. Initialize it
        in place with a truncated normal: mean 0, standard deviation 1, truncated to
        [-3, 3].
      - forward: token_ids is an integer (long) tensor of arbitrary shape. Return the
        embedding rows for those ids, so the output shape is the input shape with an
        extra trailing embedding_dim dimension. Indexing `self.weight` with the id
        tensor does exactly this.

    Test:  uv run pytest -k test_embedding
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(tensor=self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# 3.4.1  RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (RMSNorm).

    For an activation vector a of length d_model, each component is rescaled as
        RMSNorm(a_i) = a_i / RMS(a) * g_i,
    where  RMS(a) = sqrt( (1/d_model) * sum_i a_i^2  +  eps ),
    and g is a learned per-channel gain (d_model values). Unlike LayerNorm, there is
    NO mean subtraction and NO bias — you only divide by the root-mean-square and
    apply the gain. eps (default 1e-5) keeps the denominator from being zero.

    Guidance:
      - __init__: save eps; create the gain parameter `self.weight` of shape
        (d_model,) initialized to all ones (identity scaling at the start).
      - forward: the upcast to float32 (and final downcast) is provided, because
        squaring activations can overflow in low precision. In between:
          * square x, take the mean over the LAST dimension keeping that dim (so it
            broadcasts), add eps, take the square root -> this is RMS, shape (..., 1);
          * divide x by RMS, then multiply element-wise by the gain `self.weight`.
        Return the result cast back to the input's original dtype. Input and output
        are both (..., d_model).

    Test:  uv run pytest -k test_rmsnorm
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Upcast to float32 to avoid overflow when squaring, then downcast at the end.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms_x * self.weight
        return x.to(in_dtype)


# ---------------------------------------------------------------------------
# 3.4.2  SiLU + SwiGLU feed-forward
# ---------------------------------------------------------------------------
def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """SiLU / Swish activation, applied element-wise:
        SiLU(x) = x * sigmoid(x) = x / (1 + e^-x).
    It behaves like a smooth version of ReLU (no hard corner at 0). Use the library
    sigmoid function rather than writing the exponential yourself, for numerical
    stability. Output has the same shape as the input.

    Test:  uv run pytest -k test_silu
    """
    # Numerically stable PyTorch version: return x * torch.sigmoid(x)
    return x / (1 + torch.exp(-x))


class SwiGLU(nn.Module):
    """Position-wise feed-forward network using the SwiGLU activation:
        FFN(x) = W2 ( SiLU(W1 x) ⊙ W3 x ),   where ⊙ is element-wise multiplication.
    This is a Gated Linear Unit: SiLU(W1 x) acts as a gate that is multiplied
    element-wise into the "value" branch W3 x, and W2 projects the result back down.
    Modern LLMs (Llama, Qwen) use this instead of the original ReLU FFN, and omit all
    biases.

    Guidance:
      - __init__: create three Linear sub-layers named exactly w1, w2, w3:
          self.w1 : d_model -> d_ff   (weight shape (d_ff, d_model))   # gate branch
          self.w2 : d_ff   -> d_model (weight shape (d_model, d_ff))   # down-projection
          self.w3 : d_model -> d_ff   (weight shape (d_ff, d_model))   # value branch
        d_ff is passed in by the caller (canonically about 8/3 * d_model, rounded to a
        multiple of 64 for hardware efficiency), so you do not compute it here.
      - forward: pass x through w1 and apply silu (the gate); pass x through w3 (the
        value); multiply them element-wise; pass that through w2. Input and output are
        both (..., d_model).

    Test:  uv run pytest -k test_swiglu
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        gate = silu(self.w1(x))
        x = gate * (self.w3(x))
        x = self.w2(x)
        return x


# ---------------------------------------------------------------------------
# 3.4.3  Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE). Injects position information by rotating each
    consecutive PAIR of dimensions of a query/key vector by an angle that grows with
    the token's absolute position. There are no learnable parameters.

    The vector of size d_k is treated as d_k/2 pairs. Pair index k (0..d_k/2 - 1) has
    frequency  Theta^(-(2k)/d_k), and the angle applied at token position i is
        theta_{i,k} = i * Theta^(-(2k)/d_k).
    So low-index pairs rotate fast and high-index pairs rotate slowly, giving each
    position a unique signature. Each pair (x_even, x_odd) is rotated by the 2x2 matrix
        [[cos, -sin],
         [sin,  cos]]   with that angle.
    Because the transformation is a rotation, the dot product between a query at
    position i and a key at position j ends up depending only on their RELATIVE offset
    i - j, which is what we want.

    Guidance:
      - __init__: d_k is even. Build the per-pair inverse frequencies: for the even
        indices 0, 2, 4, ..., d_k-2, take theta to the power of (index / d_k) and
        negate the exponent -> shape (d_k/2,). Take positions 0..max_seq_len-1 and form
        the outer product position * inv_freq -> angle table of shape
        (max_seq_len, d_k/2). Store its cosine and sine as NON-persistent buffers via
        register_buffer(..., persistent=False): non-persistent means they are not
        written to the state dict (they are recomputable) but still follow .to(device).
        Do NOT use nn.Parameter — these are fixed constants.
      - forward: x is (..., seq, d_k); token_positions is (..., seq) giving each
        element's position index. Gather the cos/sin rows at those positions ->
        (..., seq, d_k/2). View x as d_k/2 pairs along the last axis (even element and
        odd element per pair). Rotate every pair:
            out_even = x_even * cos - x_odd * sin
            out_odd  = x_even * sin + x_odd * cos
        Re-interleave the rotated even/odd values back into a last dimension of size
        d_k and return (same shape as x). Ensure cos/sin broadcast over any batch/head
        dimensions sitting in front of seq.

    Test:  uv run pytest -k test_rope
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len)[:, None]
        inv_freq = theta ** (-torch.arange(0, d_k, 2) / d_k)
        angles = positions * inv_freq

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_k"],
        token_positions: Int[Tensor, " ... seq"],
    ) -> Float[Tensor, " ... seq d_k"]:
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return rearrange(torch.stack([out_even, out_odd], dim=-1), "... d1 d2 -> ... (d1 d2)")


# ---------------------------------------------------------------------------
# 3.4.4  softmax + scaled dot-product attention
# ---------------------------------------------------------------------------
def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """Turn scores into a probability distribution along dimension `dim`:
        softmax(v)_i = exp(v_i) / sum_j exp(v_j).

    Guidance: exp of a large number overflows to inf, and inf/inf = NaN. Softmax is
    invariant to subtracting a constant from all inputs, so first subtract the maximum
    value along `dim` (kept as a size-1 dimension so it broadcasts) from every element
    before exponentiating — now the largest input is 0 and exp is safe. Then divide the
    exponentials by their sum along `dim`. Output has the same shape as the input, and
    the values along `dim` sum to 1.

    Test:  uv run pytest -k test_softmax
    """
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention:
        Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V.
    Each query compares itself against every key via a dot product; the scaled scores
    become a probability distribution over keys, which is used to take a weighted
    average of the value vectors. Dividing by sqrt(d_k) keeps the scores from growing
    with dimension (which would push softmax into tiny gradients).

    Guidance:
      - Scores: contract Q and K over their shared last dimension d_k to get a
        (..., queries, keys) tensor of dot products, then divide every entry by
        sqrt(d_k) (d_k = Q's last-dimension size).
      - Masking (optional): the boolean mask is (..., queries, keys) with the
        convention True = this query MAY attend to this key, False = it may not.
        Wherever the mask is False, set the score to negative infinity BEFORE the
        softmax, so those positions get exactly zero probability.
      - Weights: softmax over the KEYS dimension (the last one) so each query's weights
        sum to 1.
      - Output: contract the weights with V over the keys dimension -> (..., queries,
        d_v). Any number of leading batch/head dimensions just pass through.

    Tests:  uv run pytest -k "test_scaled_dot_product_attention or test_4d_scaled_dot_product_attention"
    """
    d_k = Q.shape[-1]
    qk_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        qk_scores = torch.where(mask, qk_scores, float("-inf"))
    qk_scores = softmax(qk_scores, dim=-1)
    attention = einsum(qk_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention


# ---------------------------------------------------------------------------
# 3.4.5  Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE on Q and K.

    Multi-head attention splits the model dimension into num_heads independent heads,
    runs scaled dot-product attention within each head in parallel, concatenates the
    per-head outputs, and applies a final output projection:
        MultiHead(x) = W_O Concat(head_1, ..., head_h),
        head_i = Attention(W_Q^i x, W_K^i x, W_V^i x).
    "Self"-attention means Q, K, V all come from the same input x. "Causal" means each
    position may only attend to itself and earlier positions, so the model can't peek
    at future tokens it must predict.

    Guidance:
      - __init__: require d_model % num_heads == 0; the per-head size is
        d_k = d_v = d_model // num_heads. Save num_heads (and d_k) and the optional
        rope module. Create four Linear layers, each (d_model, d_model), named exactly
        q_proj, k_proj, v_proj, output_proj. Because all heads are packed inside
        d_model, each of the Q/K/V projections is ONE matrix multiply covering every
        head at once (that's the whole point of doing it batched).
      - forward:
          1. Project x through q_proj, k_proj, v_proj -> each (..., seq, d_model).
          2. Split the last dimension into heads and move the head axis in front of
             seq, so each head is just another batch-like dimension:
             (..., num_heads, seq, d_k). (rearrange "... seq (heads d) -> ... heads seq d".)
          3. If a rope module was given, apply it to the per-head queries and keys ONLY
             (never the values). Use token_positions if given, else default to
             0, 1, ..., seq-1. Every head uses the same rotation.
          4. Build a causal mask of shape (seq, seq): entry (i, j) is True iff key
             position j <= query position i (a lower-triangular boolean, or compare a
             column-index row vector against a row-index column vector). It broadcasts
             over the batch and head dimensions.
          5. Run scaled_dot_product_attention on the per-head Q, K, V with that mask
             -> (..., num_heads, seq, d_v).
          6. Merge heads back into a single d_model dimension (inverse of step 2), then
             apply output_proj -> (..., seq, d_model).

    Test:  uv run pytest -k test_multihead_self_attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.rope = rope
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        seq = x.size(-2)
        q = rearrange(self.q_proj(x), "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)

        if token_positions is None:
            # Device-safe version: torch.arange(seq, device=x.device)[None, :]
            token_positions = torch.arange(seq)[None, :]
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask_i = torch.arange(seq)[:, None]
        mask_j = torch.arange(seq)[None, :]
        mask = mask_i >= mask_j
        # Device-safe PyTorch version:
        # mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        
        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = rearrange(attention, "... n seq d_v -> ... seq (n d_v)")
        
        return self.output_proj(attention)
        

# ---------------------------------------------------------------------------
# 3.5  Pre-norm Transformer block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """A pre-norm Transformer block: two residual sub-layers (attention, then
    feed-forward), each normalizing its INPUT rather than its output. Concretely:
        y = x + MultiHeadSelfAttention( RMSNorm(x) )
        z = y + SwiGLU( RMSNorm(y) )
    "Pre-norm" means the residual stream (x -> y -> z) is never itself normalized;
    normalization sits only on the branch feeding each sub-layer. This keeps a clean
    gradient path from input to output and makes training more stable — it's the
    standard used by GPT-3, Llama, PaLM, etc.

    Guidance:
      - __init__: create four sub-modules with these exact names:
          self.ln1  = RMSNorm(d_model)                           # norm before attention
          self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
          self.ln2  = RMSNorm(d_model)                           # norm before feed-forward
          self.ffn  = SwiGLU(d_model, d_ff)
        Pass the shared rope module straight through to attn.
      - forward: first sub-layer — normalize x with ln1, run attn (forward
        token_positions to it), add the result back to x. Second sub-layer — normalize
        the updated stream with ln2, run ffn, add it back. Return the result. Shape is
        unchanged throughout: (..., seq, d_model).

    Test:  uv run pytest -k test_transformer_block
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rope = rope
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z


# ---------------------------------------------------------------------------
# 3.1 / 3.5  Full Transformer LM
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    """The complete decoder-only Transformer language model. It embeds the input token
    ids into vectors, runs them through num_layers pre-norm Transformer blocks, applies
    a final RMSNorm, and projects to per-token vocabulary logits (the "LM head"). The
    logits at each position predict the NEXT token; a softmax + cross-entropy (done by
    the loss, not here) turns them into a training signal.

    Guidance:
      - __init__: create, with these exact names —
          self.token_embeddings = Embedding(vocab_size, d_model)
          one shared RotaryPositionalEmbedding built with the per-head dimension
            (d_model // num_heads), theta = rope_theta, max_seq_len = context_length.
            Build it ONCE and hand the same instance to every block so its cos/sin
            tables are computed a single time.
          self.layers = nn.ModuleList of num_layers TransformerBlock(d_model,
            num_heads, d_ff, rope).
          self.ln_final = RMSNorm(d_model)                       # after the last block
          self.lm_head  = Linear(d_model, vocab_size)            # output projection
      - forward: in_indices is a (batch, seq) integer tensor of token ids
        (seq <= context_length). Embed the ids, pass through each block in order (each
        block builds its own causal mask and default 0..seq-1 positions), apply
        ln_final, then lm_head. Return the raw logits of shape (batch, seq, vocab_size).
        Do NOT apply softmax here.

    Tests:  uv run pytest -k test_transformer_lm
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k = d_model // num_heads, max_seq_len=context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,rope) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: Int[Tensor, " batch seq"]) -> Float[Tensor, " batch seq vocab_size"]:
        x = self.token_embeddings(in_indices)
        for l in self.layers:
            x = l(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


# ===========================================================================
# Problem (transformer_accounting): Transformer LM resource accounting
# ===========================================================================
#
# Almost all Transformer FLOPs are matrix multiplies. Rule: given A (m x n) and
# B (n x p), computing A @ B costs 2*m*n*p FLOPs (each of the m*p outputs is a
# length-n dot product = n mults + n adds = 2n FLOPs).
#
# Notation:  L = context_length,  d = d_model,  d_ff,  V = vocab_size,
#            N = num_layers,  h = num_heads,  d_k = d_v = d/h.
# RMSNorm / RoPE / softmax / residual adds are not matmuls and are negligible.
#
# ---------------------------------------------------------------------------
# The matrix multiplies in one forward pass (per Transformer block)
# ---------------------------------------------------------------------------
#   Q, K, V, O projections : 4 x (L,d)@(d,d)          = 4 * 2*L*d*d  = 8*L*d^2
#     (all heads are packed inside d_model, so each projection is ONE d x d
#      matmul covering every head at once)
#   Attention scores QK^T  : h x (L,d_k)@(d_k,L)       = h * 2*L*L*d_k = 2*L^2*d
#   Weighted sum  scores@V : h x (L,L)@(L,d_v)         = h * 2*L*L*d_v = 2*L^2*d
#   FFN (SwiGLU) w1,w2,w3  : 3 x (L,d)@(d,d_ff)        = 3 * 2*L*d*d_ff = 6*L*d*d_ff
# LM head (once, at the end): (L,d)@(d,V)              = 2*L*d*V
#
# TOTAL = N*( 8*L*d^2  +  4*L^2*d  +  6*L*d*d_ff )  +  2*L*d*V
#
# ---------------------------------------------------------------------------
# (a) GPT-2 XL parameters & memory  (V=50257, L=1024, N=48, d=1600, h=25,
#     d_ff=4288 = nearest multiple of 64 to 8/3*d).
#   params = V*d (embed) + N*(4*d^2 + 3*d*d_ff + 2*d) + d + d*V (head)
#          = 1,640,452,800  ~= 1.64B parameters.
#   fp32 (4 bytes/param) => ~6.56 GB (6.11 GiB) just to load the weights.
#
# ---------------------------------------------------------------------------
# (b) GPT-2 XL forward-pass FLOPs (L=1024):
#   FFN                      : 2.023e12   57.5%
#   Attn projections (QKVO)  : 1.007e12   28.6%
#   Attn scores (QK^T + @V)  : 3.221e11    9.2%
#   LM head                  : 1.647e11    4.7%
#   TOTAL                    : 3.517e12  (~3.52 TFLOPs)
#
# ---------------------------------------------------------------------------
# (c) Which parts cost the most FLOPs?
#   The feed-forward networks dominate (~57.5%). Together with the attention
#   projections, the position-wise linear layers are ~86% of all FLOPs, while
#   the actual attention interaction (QK^T + @V) is only ~9%.
#
# ---------------------------------------------------------------------------
# (d) Breakdown across GPT-2 sizes (all L=1024, V=50257):
#                        small        medium       large        XL
#   (N, d, h, d_ff)   (12,768,12,   (24,1024,16,  (36,1280,20, (48,1600,25,
#                        2048)         2752)        3392)        4288)
#   Attn projections    19.9%        24.8%        27.3%        28.6%
#   Attn scores         13.3%        12.4%        10.9%         9.2%
#   FFN                 39.8%        50.1%        54.3%        57.5%
#   LM head             27.1%        12.7%         7.4%         4.7%
#   Total FLOPs        0.292T       0.830T       1.769T       3.517T
#   As the model grows, the LM head's share shrinks fast (its 2*L*d*V grows only
#   linearly in d, while the blocks grow ~N*d^2), so the per-block linear layers
#   -- especially the FFN -- take a proportionally larger share. The attention-
#   score term also shrinks slightly (proportional to L^2*d vs the d^2 growth of
#   the rest).
#
# ---------------------------------------------------------------------------
# (e) GPT-2 XL with context_length = 16,384:
#   TOTAL ~= 1.336e14 (133.6 TFLOPs) -- about 38x more than L=1024, even though
#   L only grew 16x. The extra factor is the attention-score term, which is
#   QUADRATIC in L while everything else is linear. Its share jumps 9.2% -> 61.7%
#   and now dominates; FFN falls to 24.2%, projections to 12.1%, head to 2.0%.
# ===========================================================================
