import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    WHAT: Rotary Position Embeddings (RoPE).
    WHY: Instead of ADDING position info to embeddings,
         we ROTATE Q and K vectors by position-dependent angles.
         The dot product q_i · k_j then depends ONLY on (j-i),
         which is exactly what attention should care about.

         Paper: "RoFormer" (Su et al., 2021)
         Used in: LLaMA 1/2/3, Mistral, Mixtral, Qwen 1/2, Gemma

         How it works at a glance:
         1. For each pair of dimensions (0,1), (2,3), (4,5), ...
         2. Rotate by angle = position * frequency
         3. Lower dims rotate fast (local position)
            Higher dims rotate slow (global position)
         4. The dot product naturally depends on relative distance
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        WHAT: Precompute rotation frequencies for fast lookup.

        Args:
            d_model:     Head dimension (e.g., 64 for GPT-2). Must be even.
            max_seq_len: Precompute angles for positions 0..max_seq_len-1.
            theta:       Base frequency. 10000 is standard. Controls the
                         spread between fast and slow rotation frequencies.
        """
        super().__init__()

        # WHAT: Verify d_model is even (must have pairs to rotate)
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) must be even for RoPE. "
            f"Each pair of dimensions needs a partner to rotate with."
        )

        # WHAT: Create dimension indices: [0, 2, 4, ..., d_model-2]
        # WHY: Each pair (2i, 2i+1) gets the same rotation frequency.
        #      We only need half the indices because pairs share.
        dim_indices = torch.arange(0, d_model, 2).float()

        # WHAT: Compute rotation frequencies
        # WHY: theta_i = 1 / (theta ^ (2i / d_model))
        #
        #      i=0:  1 / 10000^(0/64)      = 1.0      → fast rotation (local)
        #      i=30: 1 / 10000^(60/64)     ≈ 0.0001   → slow rotation (global)
        #
        #      This multi-scale approach means some dimensions
        #      capture local word order while others capture
        #      long-range position relationships.
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # WHAT: Precompute angles for all positions
        # WHY: Computing cos/sin during training is expensive.
        #      Precomputing them once and caching is 100x faster.
        positions = torch.arange(max_seq_len).float()     # [0, 1, 2, ..., 2047]

        # WHAT: Outer product: each position x each frequency
        #       freqs[p, i] = p * inv_freq[i] = angle for position p, dim pair i
        #       Shape: [max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # WHAT: Duplicate to full dimension
        # WHY: Each dim pair (2i, 2i+1) gets the same angle,
        #      so we copy each angle: [θ0, θ1, θ2, ...] -> [θ0, θ0, θ1, θ1, ...]
        emb = freqs.repeat_interleave(2, dim=-1)         # [max_seq_len, d_model]

        # WHAT: Cache cos and sin for all positions
        # WHY: register_buffer means these move with model.to(device)
        #      and are saved with model.state_dict(), but are NOT
        #      trainable parameters (no gradients needed).
        self.register_buffer("cos_cached", emb.cos())   # cos of each angle
        self.register_buffer("sin_cached", emb.sin())   # sin of each angle

    @staticmethod
    @staticmethod
    def rotate_half(x):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        WHAT: Apply RoPE to queries or keys.

        Input:  [batch, num_heads, seq_len, head_dim]
                x can be either Q or K (NOT V — values don't need position)
        Output: Same shape, rotated by position-dependent angles

        WHY applied only to Q and K:
        The attention score = Q_i · K_j controls WHICH values to attend to.
        We want this score to depend on relative position.
        The VALUE vectors carry content — position is irrelevant for the
        content itself. Position only matters for deciding which tokens
        to pay attention TO.
        """
        # WHAT: Extract cos and sin for current sequence length
        # WHY: If seq_len=512 but max_seq_len=2048, we only need
        #      the first 512 rows of the cached cos/sin tables.
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # WHAT: Add batch and head dimensions for broadcasting
        # WHY: cos/sin are [seq_len, head_dim]. We need them to
        #      multiply with x [batch, heads, seq_len, head_dim].
        #      unsqueeze(0).unsqueeze(0) adds dims at positions 0 and 1:
        #      [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #      Now they broadcast correctly over batch and heads.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # WHAT: Execute rotation: x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # WHY: This is mathematically equivalent to applying a 2D rotation
        #      matrix to each pair of dimensions, but implemented in pure
        #      element-wise operations — much faster and parallelizable.
        return (x * cos) + (self.rotate_half(x) * sin)