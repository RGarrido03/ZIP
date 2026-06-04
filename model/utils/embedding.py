from torch import nn

from .utils import Conv2d_BN
from timm.layers.helpers import to_2tuple

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=2, padding=0,
                 in_chans=3, embed_dim=48):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = Conv2d_BN(in_chans, embed_dim, patch_size, stride,padding)

    def forward(self, x):
        x = self.proj(x)
        return x