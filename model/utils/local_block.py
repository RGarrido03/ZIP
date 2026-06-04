from timm.layers.weight_init import trunc_normal_
from timm.layers import DropPath
from torch import nn, ones
from .utils import RepDW, FFN


class LocalBlock(nn.Module):
    """Implementation of ConvEncoder with 3*3 and 1*1 convolutions."""

    def __init__(self, dim, hidden_dim=64, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.mlp(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
