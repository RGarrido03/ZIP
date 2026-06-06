"""VSSD-style Mamba3 NC-SSD backbone for MMDetection.

Combines:
- VSSD wrapping architecture (Stem, PatchMerging, sequence-based blocks, LayerNorm outputs)
- Mamba3 NC-SSD block (non-causal matrix-multiplication attention)

Design:
- Stem: ConvNeXt-style (from VSSD) — conv3×3 stride2, residual conv blocks, conv3×3 stride2
- Blocks: Mamba3 NC-SSD adapted to operate on (B, L, C) with explicit H,W (like VMAMBA2Block)
- Downsample: VSSD PatchMerging (1×1 expand, 3×3 DWConv stride2, 1×1 project)
- Output: LayerNorm per stage, reshape to (B, C, H, W)
"""

from copy import deepcopy

import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from model.mamba3 import Mamba3

# ---------------------------------------------------------------------------
# VSSD utilities (copied from human-context/inspiration/adapt/utils.py)
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0,
        norm=nn.BatchNorm2d,
        act_func=nn.ReLU,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class VSSDStem(nn.Module):
    """ConvNeXt-style stem from VSSD.

    conv1: 3×3, stride 2  →  C//2
    conv2: residual (3×3→3×3), stride 1  →  C//2
    conv3: 3×3 stride 2 → 4×C, then 1×1 → C
    Total stride: 4× down.
    Output: (B, L, C) flattened sequence.
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(
            in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Sequential(
            ConvLayer(
                embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1
            ),
            ConvLayer(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                act_func=None,
            ),
        )
        self.conv3 = nn.Sequential(
            ConvLayer(
                embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1
            ),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, act_func=None),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        return x


class VSSDPatchMerging(nn.Module):
    """VSSD downsampling: 1×1 expand → 3×3 DWConv stride 2 → 1×1 project.

    Input: (B, L, in_dim) sequence.
    Output: (B, L/4, out_dim) sequence.
    """

    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        mid_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer(in_dim, mid_dim, kernel_size=1),
            ConvLayer(
                mid_dim, mid_dim, kernel_size=3, stride=2, padding=1, groups=mid_dim
            ),
            ConvLayer(mid_dim, out_dim, kernel_size=1, act_func=None),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.conv(x).flatten(2).permute(0, 2, 1)  # (B, L, C)
        return x


# ---------------------------------------------------------------------------
# Mamba3 NC-SSD block adapted for (B, L, C) sequence input (VSSD style)
# ---------------------------------------------------------------------------


class Mamba3Block(nn.Module):
    """Mamba3 NC-SSD block operating on (B, L, C) with explicit H, W.

    Mirrors VMAMBA2Block's structure:
    - CPE (depthwise conv on reshaped 2D)
    - Pre-norm → Mamba3(NCSSD) with 2D spatial RoPE
    - Pre-norm → MLP (2-layer Conv2d_BN-style FFN)
    - DropPath on both branches
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ssm_d_state: int = 128,
        ssm_expand: int = 2,
        ssm_headdim: int = 64,
        ssm_ngroups: int = 1,
        ssm_is_mimo: bool = False,
        ssm_mimo_rank: int = 4,
        ssm_is_outproj_norm: bool = False,
        ssm_chunk_size: int = 64,
        ssm_d_conv: int = 3,
        ssm_rope_fraction: float = 0.5,
        ssm_rope_base: float = 10000.0,
        ssm_dt_limit: tuple = (0.001, 100.0),
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.ssm_branch = True
        self.mlp_branch = True
        self.rope_base = ssm_rope_base

        # NC-SSD Mamba3
        self.op = Mamba3(
            d_model=dim,
            d_state=ssm_d_state,
            expand=ssm_expand,
            headdim=ssm_headdim,
            ngroups=ssm_ngroups,
            is_mimo=ssm_is_mimo,
            mimo_rank=ssm_mimo_rank,
            is_outproj_norm=ssm_is_outproj_norm,
            use_ncssd=True,
            chunk_size=ssm_chunk_size,
            dropout=0.0,
            rope_fraction=ssm_rope_fraction,
            dt_limit=ssm_dt_limit,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.0),
        )

        # CPE: 2D depthwise conv for local context (matching VMAMBA2Block)
        self.d_conv = ssm_d_conv
        if self.d_conv > 0:
            self.cpe1 = nn.Conv2d(
                dim,
                dim,
                kernel_size=ssm_d_conv,
                padding=ssm_d_conv // 2,
                groups=dim,
                bias=False,
            )
            self.cpe2 = nn.Conv2d(
                dim,
                dim,
                kernel_size=ssm_d_conv,
                padding=ssm_d_conv // 2,
                groups=dim,
                bias=False,
            )

    def _compute_2d_rope_angles(self, H, W, B, device):
        L = H * W
        K = self.op.num_rope_angles
        K2 = K // 2
        nheads = self.op.nheads

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
        h_idx = i // W
        w_idx = i % W
        angles_flat = angles_grid[h_idx, w_idx]

        angles = angles_flat.unsqueeze(0).unsqueeze(2).expand(B, L, nheads, K)
        return angles.to(torch.float32)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
            H, W: spatial resolution
        Returns:
            (B, L, C)
        """
        B, L, C = x.shape

        # CPE before SSM branch
        if self.d_conv > 0:
            x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            x_cpe = self.cpe1(x_2d).flatten(2).permute(0, 2, 1)  # (B, L, C)
            x_ssm_in = x + x_cpe
        else:
            x_ssm_in = x

        # SSM branch
        if self.ssm_branch:
            angles_2d = self._compute_2d_rope_angles(H, W, B, x.device)
            ssm_out = self.op(self.norm1(x_ssm_in), external_angles=angles_2d)
            x = x + self.drop_path(ssm_out)

        # CPE before MLP branch
        if self.d_conv > 0:
            x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x_cpe = self.cpe2(x_2d).flatten(2).permute(0, 2, 1)
            x = x + x_cpe

        # MLP branch
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# Stage builder (VSSD-style: all blocks are Mamba3, all operate on sequences)
# ---------------------------------------------------------------------------


def _build_stage(
    dim: int,
    depth: int,
    drop_paths: list,
    mlp_ratio: float = 4.0,
    downsample=None,
    **mamba3_kwargs,
) -> nn.ModuleList:
    blocks = nn.ModuleList()
    for i in range(depth):
        blocks.append(
            Mamba3Block(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop_path=drop_paths[i],
                **mamba3_kwargs,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

MAMBA3_VSSD_ARCH = {
    "micro": {"embed_dims": [48, 96, 192, 384], "depths": [2, 2, 8, 4]},
    "tiny": {"embed_dims": [64, 128, 256, 512], "depths": [2, 4, 8, 4]},
    "small": {"embed_dims": [64, 128, 256, 512], "depths": [3, 4, 18, 5]},
    "base": {"embed_dims": [96, 192, 384, 768], "depths": [3, 4, 18, 5]},
}


class Mamba3VSSDNBackbone(nn.Module):
    """VSSD-style Mamba3 NC-SSD backbone.

    Uses the VSSD wrapping architecture (ConvNeXt stem, sequence-based blocks,
    PatchMerging downsamplers, LayerNorm outputs) with Mamba3 NC-SSD blocks.

    Args:
        arch: Architecture preset ('tiny', 'small', 'base', 'micro').
        layers: Depths for each stage (overrides arch).
        embed_dims: Channel dims per stage (overrides arch).
        out_indices: Stage indices to output. Default (0,1,2,3).
        ssm_d_state: SSM state dimension.
        ssm_expand: Mamba3 inner dim expansion ratio.
        ssm_headdim: Head dimension.
        ssm_ngroups: Number of BC groups.
        ssm_is_mimo: Use MIMO projection.
        ssm_is_outproj_norm: Use gated output norm.
        ssm_d_conv: CPE depthwise conv kernel (0 to disable).
        ssm_chunk_size: Chunk size.
        ssm_rope_fraction: RoPE fraction.
        ssm_rope_base: RoPE frequency base.
        ssm_dt_limit: DT clamping range.
        drop_path_rate: Max stochastic depth rate.
        mlp_ratios: MLP expansion per stage.
        pretrained: Path to pretrained checkpoint.
        init_cfg: MMEngine init config.
    """

    _arch_info = MAMBA3_VSSD_ARCH

    def __init__(
        self,
        layers: list | None = None,
        embed_dims: list | None = None,
        arch: str = "tiny",
        out_indices: tuple = (0, 1, 2, 3),
        ssm_d_state: int = 128,
        ssm_expand: int = 2,
        ssm_headdim: int = 64,
        ssm_ngroups: int = 1,
        ssm_is_mimo: bool = False,
        ssm_is_outproj_norm: bool = False,
        ssm_d_conv: int = 3,
        ssm_chunk_size: int = 64,
        ssm_rope_fraction: float = 0.5,
        ssm_rope_base: float = 10000.0,
        ssm_dt_limit: tuple = (0.001, 100.0),
        drop_path_rate: float = 0.1,
        mlp_ratios: float | list = 4.0,
        pretrained: str | None = None,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        if layers is None or embed_dims is None:
            info = self._arch_info[arch]
            layers = info["depths"]
            embed_dims = info["embed_dims"]

        num_stages = len(layers)

        if isinstance(mlp_ratios, (int, float)):
            mlp_ratios = [mlp_ratios] * num_stages

        self.out_indices = out_indices
        self.arch = arch
        self.pretrained = pretrained

        mamba3_kwargs = dict(
            ssm_d_state=ssm_d_state,
            ssm_expand=ssm_expand,
            ssm_headdim=ssm_headdim,
            ssm_ngroups=ssm_ngroups,
            ssm_is_mimo=ssm_is_mimo,
            ssm_is_outproj_norm=ssm_is_outproj_norm,
            ssm_d_conv=ssm_d_conv,
            ssm_chunk_size=ssm_chunk_size,
            ssm_rope_fraction=ssm_rope_fraction,
            ssm_rope_base=ssm_rope_base,
            ssm_dt_limit=ssm_dt_limit,
        )

        # VSSD ConvNeXt-style stem
        self.patch_embed = VSSDStem(in_chans=3, embed_dim=embed_dims[0])

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            depth = layers[i]
            stage_dpr = dpr[sum(layers[:i]) : sum(layers[: i + 1])]

            blocks = _build_stage(
                dim=embed_dims[i],
                depth=depth,
                drop_paths=stage_dpr,
                mlp_ratio=mlp_ratios[i],
                **mamba3_kwargs,
            )
            self.stages.append(blocks)

        # Downsample layers between stages (VSSD PatchMerging)
        self.downsamples = nn.ModuleList()
        for i in range(num_stages - 1):
            self.downsamples.append(VSSDPatchMerging(embed_dims[i], embed_dims[i + 1]))

        # Output norms (one per stage)
        self.out_norms = nn.ModuleList()
        for i in range(num_stages):
            self.out_norms.append(nn.LayerNorm(embed_dims[i]))

        self.out_channels = embed_dims

        for m in [self.patch_embed, *self.stages, *self.downsamples, *self.out_norms]:
            self._init_weights(m)

        self.init_cfg = deepcopy(init_cfg)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _compute_hw_after_stem(self, H, W):
        """VSSD stem does two stride-2 convs → H//4, W//4."""
        H = int((H - 1) / 2) + 1
        H = int((H - 1) / 2) + 1
        W = int((W - 1) / 2) + 1
        W = int((W - 1) / 2) + 1
        return H, W

    def _compute_hw_after_downsample(self, H, W):
        """PatchMerging uses 3×3 conv stride 2 with padding 1 → H//2, W//2."""
        H = int((H - 1) / 2) + 1
        W = int((W - 1) / 2) + 1
        return H, W

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)  # (B, L, C)
        H, W = self._compute_hw_after_stem(H, W)
        outs = []
        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, H, W)

            if i in self.out_indices:
                # Apply output norm and reshape to (B, C, H, W)
                out = self.out_norms[i](x)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

            if i < len(self.stages) - 1:
                x = self.downsamples[i](x, H, W)
                H, W = self._compute_hw_after_downsample(H, W)

        return tuple(outs)
