# Copyright (c) 2026. Non-Causal State Space Duality for Mamba-3.
# Based on VSSD's NC-SSD for Mamba-2 (VSSD/classification/models/mamba2.py:non_casual_linear_attn).
# Replaces the causal scan kernel with all-to-all matrix-multiplication attention.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_rope(q, k, angles, split_tensor_size, num_rope_angles):
    """
    Apply rotary position embeddings to Q and K tensors.

    For Mamba-3, only the first `split_tensor_size` dimensions of the last axis
    (d_state) receive RoPE; the remaining dimensions pass through unchanged.

    Args:
        q: (..., N) where N = d_state
        k: (..., N)
        angles: (..., S) where S = num_rope_angles = split_tensor_size // 2
        split_tensor_size: number of d_state dims to rotate
        num_rope_angles: number of rotation angle pairs

    Returns:
        q_roped, k_roped: same shapes as q, k
    """
    dtype = q.dtype
    q = q.float()
    k = k.float()

    # Split: rope portion + pass-through portion
    q_rope = q[..., :split_tensor_size]
    q_pass = q[..., split_tensor_size:]
    k_rope = k[..., :split_tensor_size]
    k_pass = k[..., split_tensor_size:]

    # Reshape rope portion to (..., 2, num_rope_angles)
    q_rope = q_rope.unflatten(-1, (2, num_rope_angles))
    k_rope = k_rope.unflatten(-1, (2, num_rope_angles))

    cos = angles.cos()
    sin = angles.sin()

    q1, q2 = q_rope[..., 0, :], q_rope[..., 1, :]
    k1, k2 = k_rope[..., 0, :], k_rope[..., 1, :]

    q_rot_1 = q1 * cos - q2 * sin
    q_rot_2 = q1 * sin + q2 * cos
    k_rot_1 = k1 * cos - k2 * sin
    k_rot_2 = k1 * sin + k2 * cos

    q_rot = torch.stack([q_rot_1, q_rot_2], dim=-2).flatten(-2)
    k_rot = torch.stack([k_rot_1, k_rot_2], dim=-2).flatten(-2)

    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)

    return q.to(dtype), k.to(dtype)


class NCSSD_Mamba3(nn.Module):
    """
    Non-Causal State Space Duality for Mamba-3.

    Replaces the causal scan kernel (mamba3_siso_combined / mamba3_mimo_combined)
    with matrix-multiplication attention, enabling all-to-all (non-causal / bidirectional)
    interactions. Adapted from VSSD's non_casual_linear_attn for Mamba-2.

    Core formulation:
        V_gated  = V * sigmoid(trap)
        dA_sqrt  = exp(ADT / 2)              (symmetric per-position decay)
        V_scaled = V_gated * dA_sqrt          (half-decay on V side)
        Q_scaled = Q * dA_sqrt                (half-decay on Q side)
        KV       = K^T @ V_scaled             (all-to-all key-value)
        Out      = Q_scaled @ KV + D * V      (attention + skip connection)

    The symmetric split exp(ADT_i/2) * exp(ADT_j/2) = exp((ADT_i+ADT_j)/2)
    gives a factorized non-causal decay that depends on both positions
    equally, unlike the causal kernel where decay is cumulative over the
    interval j→i.

    RoPE is applied to Q and K before the matrix multiply.
    Supports both SISO (mimo_rank=1) and MIMO (mimo_rank>1) modes.
    """

    def __init__(
        self,
        nheads: int,
        num_bc_heads: int,
        headdim: int,
        d_state: int,
        split_tensor_size: int,
        num_rope_angles: int,
        mimo_rank: int = 1,
        is_mimo: bool = False,
        is_outproj_norm: bool = False,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.nheads = nheads
        self.num_bc_heads = num_bc_heads
        self.headdim = headdim
        self.d_state = d_state
        self.split_tensor_size = split_tensor_size
        self.num_rope_angles = num_rope_angles
        self.mimo_rank = mimo_rank
        self.is_mimo = is_mimo
        self.is_outproj_norm = is_outproj_norm
        self.norm = norm  # RMSNormGated for outproj_norm
        self.ngroups_ratio = nheads // num_bc_heads

    def forward(
        self,
        x: torch.Tensor,        # (B, L, H, P) — V
        z: torch.Tensor,        # (B, L, H, P) — gate for outproj_norm
        B: torch.Tensor,        # (B, L, 1, G, N) SISO or (B, L, R, G, N) MIMO — K
        C: torch.Tensor,        # (B, L, 1, G, N) SISO or (B, L, R, G, N) MIMO — Q
        ADT: torch.Tensor,      # (B, H, L) — A * DT, position-wise decay
        trap: torch.Tensor,     # (B, H, L) — trap gating (pre-sigmoid)
        angles: torch.Tensor,   # (B, L, H, S) — RoPE angles
        D: torch.Tensor,        # (H,) — skip connection parameter
        B_bias: torch.Tensor,   # (H, 1, N) SISO or (H, R, N) MIMO
        C_bias: torch.Tensor,   # (H, 1, N) SISO or (H, R, N) MIMO
        mimo_x: torch.Tensor | None = None,  # (H, R, P) or None
        mimo_z: torch.Tensor | None = None,  # (H, R, P) or None
        mimo_o: torch.Tensor | None = None,  # (H, R, P) or None
    ) -> torch.Tensor:
        """
        Returns:
            y: (B, L, H, P) — output, ready for rearrange + out_proj
        """
        if self.is_mimo:
            return self._forward_mimo(
                x, z, B, C, ADT, trap, angles, D,
                B_bias, C_bias, mimo_x, mimo_z, mimo_o,
            )
        else:
            return self._forward_siso(
                x, z, B, C, ADT, trap, angles, D, B_bias, C_bias,
            )

    # ------------------------------------------------------------------
    # SISO path
    # ------------------------------------------------------------------
    def _forward_siso(self, x, z, B, C, ADT, trap, angles, D, B_bias, C_bias):
        batch, seqlen, nheads, headdim = x.shape

        # --- B, C: (B, L, 1, G, N) → (B, L, G, N) → expand G→H ---
        B = B.squeeze(2)  # (B, L, G, N)
        C = C.squeeze(2)
        B = B.repeat_interleave(self.ngroups_ratio, dim=2)  # (B, L, H, N)
        C = C.repeat_interleave(self.ngroups_ratio, dim=2)

        # Add per-head bias
        B_bias_s = B_bias.squeeze(1)  # (H, N)
        C_bias_s = C_bias.squeeze(1)
        B = B + B_bias_s.reshape(1, 1, nheads, -1)
        C = C + C_bias_s.reshape(1, 1, nheads, -1)

        # Permute to (B, H, L, N) for matmul
        K = B.permute(0, 2, 1, 3)
        Q = C.permute(0, 2, 1, 3)

        # --- RoPE ---
        angles_rope = angles.permute(0, 2, 1, 3)  # (B, H, L, S)
        Q, K = _apply_rope(Q, K, angles_rope, self.split_tensor_size, self.num_rope_angles)

        # --- Trap gating + symmetric ADT decay ---
        V = x.permute(0, 2, 1, 3)  # (B, H, L, P)
        trap_gate = torch.sigmoid(trap)  # (B, H, L)

        dA_sqrt = torch.exp(ADT * 0.5).unsqueeze(-1)  # (B, H, L, 1) — exp(ADT/2) ϵ (0,1]
        V_gated = V * trap_gate.unsqueeze(-1)  # (B, H, L, P)
        V_scaled = V_gated * dA_sqrt  # (B, H, L, P) — half-decay on V side

        Q_scaled = Q * dA_sqrt  # (B, H, L, N) — half-decay on Q side

        # --- All-to-all matrix multiply ---
        # Scale by 1/√(d_state * L) to keep magnitudes stable as seqlen varies:
        #   K^T@V sums over L positions → split 1/√L across K and V
        #   Q@KV sums over N positions → 1/√N from d_state
        scale = (self.d_state * seqlen) ** -0.5
        KV = torch.einsum("bhln,bhlp->bhnp", K, V_scaled) * scale  # (B, H, N, P)
        Out = torch.einsum("bhln,bhnp->bhlp", Q_scaled, KV)  # (B, H, L, P)

        # --- Skip connection: D * V (original V, ungated) ---
        Out = Out + D.view(1, nheads, 1, 1) * V

        # --- Permute back: (B, L, H, P) ---
        y = Out.permute(0, 2, 1, 3)

        # --- outproj_norm ---
        if self.is_outproj_norm:
            y_flat = y.reshape(batch, seqlen, -1)  # (B, L, d_inner)
            z_flat = z.reshape(batch, seqlen, -1)
            y_flat = self.norm(y_flat, z_flat)
            y = y_flat.reshape(batch, seqlen, nheads, headdim)

        return y

    # ------------------------------------------------------------------
    # MIMO path
    # ------------------------------------------------------------------
    def _forward_mimo(
        self, x, z, B, C, ADT, trap, angles, D,
        B_bias, C_bias, mimo_x, mimo_z, mimo_o,
    ):
        batch, seqlen, nheads, headdim = x.shape
        num_ranks = self.mimo_rank

        # --- V projection: x (B, L, H, P) → V_mimo (B, L, H, R, P) ---
        V_mimo = torch.einsum("blhp,hrp->blhrp", x, mimo_x)  # (B, L, H, R, P)

        # Permute to (B, H, R, L, P) for per-(head,rank) matmul
        V_mimo = V_mimo.permute(0, 2, 3, 1, 4)  # (B, H, R, L, P)

        # --- Trap gating + symmetric ADT decay ---
        trap_gate = torch.sigmoid(trap)  # (B, H, L)

        dA_sqrt = torch.exp(ADT * 0.5).unsqueeze(2).unsqueeze(-1)  # (B, H, 1, L, 1)
        V_gated = V_mimo * trap_gate.unsqueeze(2).unsqueeze(-1)  # (B, H, R, L, P)
        V_scaled = V_gated * dA_sqrt  # (B, H, R, L, P) — half-decay on V side

        # --- B, C: (B, L, R, G, N) → expand G→H ---
        B = B.repeat_interleave(self.ngroups_ratio, dim=3)  # (B, L, R, H, N)
        C = C.repeat_interleave(self.ngroups_ratio, dim=3)

        # Add per-head per-rank bias: B_bias (H, R, N) → (1,1,R,H,N)
        B = B + B_bias.permute(1, 0, 2).reshape(1, 1, num_ranks, nheads, -1)
        C = C + C_bias.permute(1, 0, 2).reshape(1, 1, num_ranks, nheads, -1)

        # Permute to (B, H, R, L, N)
        K = B.permute(0, 3, 2, 1, 4)
        Q = C.permute(0, 3, 2, 1, 4)

        # --- RoPE ---
        angles_rope = angles.permute(0, 2, 1, 3).unsqueeze(2)  # (B, H, 1, L, S)
        Q, K = _apply_rope(Q, K, angles_rope, self.split_tensor_size, self.num_rope_angles)

        # Apply symmetric half-decay to Q
        Q_scaled = Q * dA_sqrt  # (B, H, R, L, N)

        # --- All-to-all matrix multiply ---
        scale = (self.d_state * seqlen) ** -0.5
        KV = torch.einsum("bhrln,bhrlp->bhrnp", K, V_scaled) * scale  # (B, H, R, N, P)
        Out = torch.einsum("bhrln,bhrnp->bhrlp", Q_scaled, KV)  # (B, H, R, L, P)

        # --- Skip connection: D * V (original V_mimo, ungated) ---
        Out = Out + D.view(1, nheads, 1, 1, 1) * V_mimo

        # Permute to (B, L, R, H, P) for outproj_norm / mimo_o
        Out = Out.permute(0, 3, 2, 1, 4)  # (B, L, R, H, P)

        # --- outproj_norm (per-rank, before mimo_o) ---
        if self.is_outproj_norm:
            z_mimo = torch.einsum("blhp,hrp->blrhp", z.float(), mimo_z)  # (B, L, R, H, P)
            Out_flat = Out.float().reshape(batch, seqlen, num_ranks, -1)  # (B, L, R, d_inner)
            z_flat = z_mimo.reshape(batch, seqlen, num_ranks, -1)
            Out_flat = self.norm(Out_flat, z_flat)
            Out = Out_flat.reshape(batch, seqlen, num_ranks, nheads, headdim)

        # --- Output projection: mimo_o (H, R, P) ---
        y = torch.einsum("blrhp,hrp->blhp", Out, mimo_o)  # (B, L, H, P)

        return y
