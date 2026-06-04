import torch.nn as nn
from .utils import Conv2d_BN

def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        Conv2d_BN(in_chs, out_chs // 2, 3, 2, 1), 
        nn.GELU(),
        Conv2d_BN(out_chs // 2, out_chs, 3, 2, 1),
        nn.GELU(),)

