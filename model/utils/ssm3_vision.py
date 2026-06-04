import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import (
        mamba3_mimo as mamba3_mimo_combined,
    )
except ImportError:
    mamba3_mimo_combined = None

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class Mamba3Vision(nn.Module):
    """
    Mamba-3 adapted for 2D vision tasks.

    Takes input of shape (B, C, H, W) and outputs (B, C, H, W).
    Uses a 2D depthwise convolution followed by 4-directional cross-scan
    to flatten the 2D image into 1D sequences, processes them through
    the Mamba-3 kernel, then cross-merges back to 2D.

    Compared to the original Mamba-3 (Block 3):
    - Removed: step(), allocate_inference_cache(), inference_params logic
    - RoPE: replaced 1D causal RoPE with 2D spatial RoPE (H+W coordinates)
    - Added: 2D depthwise convolution for local spatial context
    - Added: 4-directional cross-scan (CrossScan/CrossMerge)
    - Input/Output: (B, C, H, W) instead of (B, L, D)
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        rope_base=10000.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
        dropout=0.0,
        d_conv=3,
        act_layer=nn.SiLU,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        if not self.is_mimo:
            self.mimo_rank = 1
        else:
            assert mamba3_mimo_combined is not None, (
                "Fails to import Mamba-3 MIMO kernels. "
                "Please ensure you installed the necessary dependencies, such as TileLang."
            )

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        self.rope_fraction = rope_fraction
        self.rope_base = rope_base
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        if self.num_rope_angles < 1:
            self.num_rope_angles = 1
        self.rotary_dim_divisor = int(2 / rope_fraction)

        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(
            torch.zeros(
                (self.nheads, self.mimo_rank, self.d_state),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            torch.zeros(
                (self.nheads, self.mimo_rank, self.d_state),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )

        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            mimo_x_init_weights = (
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
                / self.mimo_rank
            )
            mimo_z_init_weights = torch.ones(
                self.nheads, self.mimo_rank, self.headdim, device=device
            )
            mimo_o_init_weights = (
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
                / self.mimo_rank
            )
            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=False, **factory_kwargs
        )

        self.d_conv = d_conv
        if self.d_conv > 0:
            self.conv2d = nn.Conv2d(
                self.d_model,
                self.d_model,
                kernel_size=d_conv,
                padding=d_conv // 2,
                groups=self.d_model,
                bias=False,
            )
        self.act = act_layer()

        self.dt_limit = kwargs.get("dt_limit", (0.0, float("inf")))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _compute_2d_rope_angles(self, H, W, B, device):
        """Compute 2D RoPE angles for all 4 cross-scan directions.

        In 1D RoPE, each token at position ``l`` gets angles ``l * theta_k``.
        In 2D RoPE (VSSD-style), each token at spatial ``(h, w)`` gets
        angles where the first half encodes row ``h`` and the second half
        encodes column ``w``, each multiplied by the same frequency band.

        We compute a grid ``(H, W, num_rope_angles)`` and then index into it
        for each scan direction, matching the CrossScan ordering.

        Returns:
            angles: ``(B*4, L, num_rope_angles)`` — stacked angles for all
                    scans and batch elements.
        """
        L = H * W
        K = self.num_rope_angles
        K2 = K // 2

        theta = 1.0 / (
            self.rope_base ** (torch.arange(start=0, end=K2, device=device) / K2)
        )

        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()

        angles_h = h_pos.view(H, 1, 1) * theta.view(1, 1, K2)
        angles_w = w_pos.view(1, W, 1) * theta.view(1, 1, K2)

        angles_grid = torch.zeros(H, W, K, device=device)
        angles_grid[:, :, :K2] = angles_h.expand(H, W, K2)
        angles_grid[:, :, K2:] = angles_w.expand(H, W, K2)

        i = torch.arange(L, device=device)
        h0 = i // W
        w0 = i % W
        angles_0 = angles_grid[h0, w0]

        h1 = i % H
        w1 = i // H
        angles_1 = angles_grid[h1, w1]

        angles_2 = torch.flip(angles_0, dims=[0])
        angles_3 = torch.flip(angles_1, dims=[0])

        angles_scans = torch.stack([angles_0, angles_1, angles_2, angles_3], dim=0)
        angles_scans = angles_scans.unsqueeze(0).expand(B, 4, L, K).reshape(B * 4, L, K)

        return angles_scans

    def _mamba3_core(self, u, angles):
        """
        Run Mamba-3 kernel on 1D sequences.

        Args:
            u: (B*4, L, d_model) — four-direction stacked input
            angles: (B*4, L, num_rope_angles) — 2D RoPE angles

        Returns:
            out: (B*4, L, d_model)
        """
        B4, L, _ = u.shape

        zxBCdtAtrap = self.in_proj(u)
        z, x, B_out, C_out, dd_dt, dd_A, trap = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B_out = rearrange(
            B_out, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads
        )
        C_out = rearrange(
            C_out, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads
        )
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        if self.dt_limit != (0.0, float("inf")):
            DT = torch.clamp(DT, min=self.dt_limit[0], max=self.dt_limit[1])
        ADT = torch.clamp(_A * DT, min=-100.0, max=0.0)
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        B_out = self.B_norm(B_out)
        C_out = self.C_norm(C_out)

        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C_out,
                K=B_out,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=False,
                cu_seqlens=None,
            )
            if self.is_outproj_norm:
                z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z = rearrange(z, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C_out.squeeze(2),
                K=B_out.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=False,
                cu_seqlens=None,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z)

        out = self.out_proj(y.to(x.dtype))
        return out

    def forward(self, u_2d: torch.Tensor):
        """
        Args:
            u_2d: (B, C, H, W) — image feature map

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = u_2d.shape

        if self.d_conv > 0:
            x_2d = self.conv2d(u_2d)
            x_2d = self.act(x_2d)
        else:
            x_2d = u_2d

        xs = CrossScan.apply(x_2d)
        xs = xs.view(B * 4, C, H * W).transpose(1, 2)

        angles = self._compute_2d_rope_angles(H, W, B, xs.device)

        out_flat = self._mamba3_core(xs, angles)

        out_flat = out_flat.transpose(1, 2).contiguous().view(B, 4, C, H, W)

        y = CrossMerge.apply(out_flat).transpose(1, 2)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2)

        y = self.dropout(y)
        return y
