from pathlib import Path

import torch
from torch import Tensor, nn

from models.ebc.timm_models import TIMMModel
from models.mamba3_vssd_ncssd import Mamba3VSSDNBackbone

from ..utils import conv1x1


class EBC(nn.Module):
    def __init__(
        self,
        block_size: int,
        bins: list[tuple[float, float]],
        bin_centers: list[float],
        zero_inflated: bool = True,
        vpt_drop: float | None = None,
        input_size: int | None = None,
        norm: str = "none",
        act: str = "none",
    ) -> None:
        super().__init__()

        self.input_size: tuple[int, int] | None = None
        if input_size is not None:
            assert input_size > 0, (
                f"Expected input_size to be a positive integer, got {input_size}"
            )
            self.input_size = (input_size, input_size)

        assert len(bins) == len(bin_centers), (
            f"Expected bins and bin_centers to have the same length, got {len(bins)} and {len(bin_centers)}"
        )
        assert len(bins) >= 2, f"Expected at least 2 bins, got {len(bins)}"
        assert all(len(b) == 2 for b in bins), (
            f"Expected bins to be a list of tuples of length 2, got {bins}"
        )
        bins = [(float(b[0]), float(b[1])) for b in bins]
        assert all(bin[0] <= p <= bin[1] for bin, p in zip(bins, bin_centers)), (
            f"Expected bin_centers to be within the range of the corresponding bin, got {bins} and {bin_centers}"
        )

        self.block_size = block_size
        self.bins = bins
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32, requires_grad=False).view(
                1, -1, 1, 1
            ),
        )

        self.zero_inflated = zero_inflated
        self.vpt_drop = vpt_drop

        self.norm = norm
        self.act = act

        self.backbone = TIMMModel(
            "mamba3_micro", block_size=block_size, norm=norm, act=act
        )
        self._build_head()

    def restore_backbone_checkpoint(self, path: Path) -> None:
        self.backbone.restore_checkpoint(path)

    def _build_head(self) -> None:
        channels = 192  # TODO: Check if this is correct
        if self.zero_inflated:
            self.bin_head = conv1x1(
                in_channels=channels,
                out_channels=len(self.bins) - 1,
            )
            self.pi_head = conv1x1(
                in_channels=channels,
                out_channels=2,
            )  # this models structural 0s.
        else:
            self.bin_head = conv1x1(
                in_channels=channels,
                out_channels=len(self.bins),
            )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        x = self.backbone(x)

        if self.zero_inflated:
            logit_pi_maps = self.pi_head(x)  # shape: (B, 2, H, W)
            logit_maps = self.bin_head(x)  # shape: (B, C, H, W)
            lambda_maps = (logit_maps.softmax(dim=1) * self.bin_centers[:, 1:]).sum(
                dim=1, keepdim=True
            )  # shape: (B, 1, H, W)

            # logit_pi_maps.softmax(dim=1)[:, 0] is the probability of zeros
            den_maps = (
                logit_pi_maps.softmax(dim=1)[:, 1:] * lambda_maps
            )  # expectation of the Poisson distribution

            if self.training:
                return logit_pi_maps, logit_maps, lambda_maps, den_maps
            else:
                return den_maps

        else:
            logit_maps = self.bin_head(x)
            den_maps = (logit_maps.softmax(dim=1) * self.bin_centers).sum(
                dim=1, keepdim=True
            )

            if self.training:
                return logit_maps, den_maps
            else:
                return den_maps


def _ebc(
    block_size: int,
    bins: list[tuple[float, float]],
    bin_centers: list[float],
    zero_inflated: bool = True,
    vpt_drop: float | None = None,
    input_size: int | None = None,
    norm: str = "none",
    act: str = "none",
) -> EBC:
    return EBC(
        block_size=block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        vpt_drop=vpt_drop,
        input_size=input_size,
        norm=norm,
        act=act,
    )
