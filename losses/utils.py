import torch
from torch import Tensor
from typing import List, Tuple


def _reshape_density(density: Tensor, block_size: int) -> Tensor:
    assert len(density.shape) == 4, f"Expected 4D (B, 1, H, W) tensor, got {density.shape}"
    assert density.shape[1] == 1, f"Expected 1 channel, got {density.shape[1]}"
    assert density.shape[2] % block_size == 0, f"Expected height to be divisible by {block_size}, got {density.shape[2]}"
    assert density.shape[3] % block_size == 0, f"Expected width to be divisible by {block_size}, got {density.shape[3]}"
    return density.reshape(density.shape[0], 1, density.shape[2] // block_size, block_size, density.shape[3] // block_size, block_size).sum(dim=(-1, -3))


def _bin_count(density_map: Tensor, bins: List[Tuple[int, int]]) -> Tensor:
    density_map = density_map.round()
    class_map = torch.zeros_like(density_map, dtype=torch.long)
    for idx, (low, high) in enumerate(bins):
        mask = (density_map >= low) & (density_map <= high)
        class_map[mask] = idx
    return class_map.squeeze(1)  # remove channel dimension
