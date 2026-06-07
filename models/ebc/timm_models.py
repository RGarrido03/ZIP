from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from models.mamba3_vssd_ncssd import Mamba3VSSDNBackbone

from ..utils import (
    ConvDownsample,
    ConvRefine,
    ConvUpsample,
    LightConvDownsample,
    LightConvRefine,
    LightConvUpsample,
    LighterConvDownsample,
    LighterConvRefine,
    LighterConvUpsample,
    _get_activation,
    _get_norm_layer,
)

regular_models = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "convnext_nano",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "mobilenetv4_conv_large",
    "mamba3_tiny",
]

heavy_models = [
    "convnext_large",
    "convnext_xlarge",
    "convnext_xxlarge",
]

light_models = [
    "mobilenetv1_100",
    "mobilenetv1_125",
    "mobilenetv2_100",
    "mobilenetv2_140",
    "mobilenetv3_large_100",
    "mobilenetv4_conv_medium",
]

lighter_models = [
    "mobilenetv2_050",
    "mobilenetv3_small_050",
    "mobilenetv3_small_075",
    "mobilenetv3_small_100",
    "mobilenetv4_conv_small_050",
    "mobilenetv4_conv_small",
    "mamba3_micro",
]

supported_models = regular_models + heavy_models + light_models + lighter_models


refiner_in_channels = {
    # ResNet
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    # ConvNeXt
    "convnext_nano": 640,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_base": 1024,
    "convnext_large": 1536,
    "convnext_xlarge": 2048,
    "convnext_xxlarge": 3072,
    # MobileNet V1
    "mobilenetv1_100": 1024,
    "mobilenetv1_125": 1280,
    # MobileNet V2
    "mobilenetv2_050": 160,
    "mobilenetv2_100": 320,
    "mobilenetv2_140": 448,
    # MobileNet V3
    "mobilenetv3_small_050": 288,
    "mobilenetv3_small_075": 432,
    "mobilenetv3_small_100": 576,
    "mobilenetv3_large_100": 960,
    # MobileNet V4
    "mobilenetv4_conv_small_050": 480,
    "mobilenetv4_conv_small": 960,
    "mobilenetv4_conv_medium": 960,
    "mobilenetv4_conv_large": 960,
    "mamba3_micro": 224,
    "mamba3_tiny": 512,
}


refiner_out_channels = {
    # ResNet
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    # ConvNeXt
    "convnext_nano": 640,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_base": 1024,
    "convnext_large": 1536,
    "convnext_xlarge": 2048,
    "convnext_xxlarge": 3072,
    # MobileNet V1
    "mobilenetv1_100": 512,
    "mobilenetv1_125": 640,
    # MobileNet V2
    "mobilenetv2_050": 160,
    "mobilenetv2_100": 320,
    "mobilenetv2_140": 448,
    # MobileNet V3
    "mobilenetv3_small_050": 288,
    "mobilenetv3_small_075": 432,
    "mobilenetv3_small_100": 576,
    "mobilenetv3_large_100": 480,
    # MobileNet V4
    "mobilenetv4_conv_small_050": 480,
    "mobilenetv4_conv_small": 960,
    "mobilenetv4_conv_medium": 960,
    "mobilenetv4_conv_large": 960,
    "mamba3_micro": 224,
    "mamba3_tiny": 512,
}


groups = {
    # ResNet
    "resnet18": 1,
    "resnet34": 1,
    "resnet50": refiner_in_channels["resnet50"] // 512,
    "resnet101": refiner_in_channels["resnet101"] // 512,
    "resnet152": refiner_in_channels["resnet152"] // 512,
    # ConvNeXt
    "convnext_nano": 8,
    "convnext_tiny": 8,
    "convnext_small": 8,
    "convnext_base": 8,
    "convnext_large": refiner_in_channels["convnext_large"] // 512,
    "convnext_xlarge": refiner_in_channels["convnext_xlarge"] // 512,
    "convnext_xxlarge": refiner_in_channels["convnext_xxlarge"] // 512,
    # MobileNet V1
    "mobilenetv1_100": None,
    "mobilenetv1_125": None,
    # MobileNet V2
    "mobilenetv2_050": None,
    "mobilenetv2_100": None,
    "mobilenetv2_140": None,
    # MobileNet V3
    "mobilenetv3_small_050": None,
    "mobilenetv3_small_075": None,
    "mobilenetv3_small_100": None,
    "mobilenetv3_large_100": None,
    # MobileNet V4
    "mobilenetv4_conv_small_050": None,
    "mobilenetv4_conv_small": None,
    "mobilenetv4_conv_medium": None,
    "mobilenetv4_conv_large": 1,
    "mamba3_micro": 1,
    "mamba3_tiny": 1,
}


class TIMMModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        block_size: Optional[int] = None,
        norm: str = "none",
        act: str = "none",
    ) -> None:
        super().__init__()
        assert model_name in supported_models, (
            f"Backbone {model_name} not supported. Supported models are {supported_models}"
        )
        assert block_size is None or block_size in [8, 16, 32], (
            f"Block size should be one of [8, 16, 32], but got {block_size}."
        )
        self.model_name = model_name
        self.encoder = Mamba3VSSDNBackbone(
            arch=model_name.lstrip("mamba3_"),
            out_indices=(3,),
        )
        print("Encoder channels:", self.encoder.out_channels)
        self.encoder_channels = self.encoder.out_channels[-1]
        self.encoder_reduction = 32  # vibes
        self.block_size = (
            block_size if block_size is not None else self.encoder_reduction
        )

        if model_name in lighter_models:
            upsample_block = LighterConvUpsample
            downsample_block = LighterConvDownsample
            decoder_block = LighterConvRefine
        elif model_name in light_models:
            upsample_block = LightConvUpsample
            downsample_block = LightConvDownsample
            decoder_block = LightConvRefine
        else:
            upsample_block = partial(ConvUpsample, groups=groups[model_name])
            downsample_block = partial(ConvDownsample, groups=groups[model_name])
            decoder_block = partial(ConvRefine, groups=groups[model_name])

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(self.encoder)

        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(self.encoder)

        if self.block_size > self.encoder_reduction:
            if self.block_size > self.encoder_reduction * 2:
                assert self.block_size == self.encoder_reduction * 4, (
                    f"Block size {self.block_size} is not supported for model {self.model_name}. Supported block sizes are {self.encoder_reduction}, {self.encoder_reduction * 2}, and {self.encoder_reduction * 4}."
                )
                self.refiner = nn.Sequential(
                    downsample_block(
                        in_channels=self.encoder_channels,
                        out_channels=refiner_in_channels[self.model_name],
                        norm_layer=norm_layer,
                        activation=activation,
                    ),
                    downsample_block(
                        in_channels=refiner_in_channels[self.model_name],
                        out_channels=refiner_out_channels[self.model_name],
                        norm_layer=norm_layer,
                        activation=activation,
                    ),
                )
            else:
                assert self.block_size == self.encoder_reduction * 2, (
                    f"Block size {self.block_size} is not supported for model {self.model_name}. Supported block sizes are {self.encoder_reduction}, {self.encoder_reduction * 2}, and {self.encoder_reduction * 4}."
                )
                self.refiner = downsample_block(
                    in_channels=self.encoder_channels,
                    out_channels=refiner_out_channels[self.model_name],
                    norm_layer=norm_layer,
                    activation=activation,
                )

            self.refiner_channels = refiner_out_channels[self.model_name]

        elif self.block_size < self.encoder_reduction:
            if self.block_size < self.encoder_reduction // 2:
                assert self.block_size == self.encoder_reduction // 4, (
                    f"Block size {self.block_size} is not supported for model {self.model_name}. Supported block sizes are {self.encoder_reduction}, {self.encoder_reduction // 2}, and {self.encoder_reduction // 4}."
                )
                self.refiner = nn.Sequential(
                    upsample_block(
                        in_channels=self.encoder_channels,
                        out_channels=refiner_in_channels[self.model_name],
                        norm_layer=norm_layer,
                        activation=activation,
                    ),
                    upsample_block(
                        in_channels=refiner_in_channels[self.model_name],
                        out_channels=refiner_out_channels[self.model_name],
                        norm_layer=norm_layer,
                        activation=activation,
                    ),
                )
            else:
                assert self.block_size == self.encoder_reduction // 2, (
                    f"Block size {self.block_size} is not supported for model {self.model_name}. Supported block sizes are {self.encoder_reduction}, {self.encoder_reduction // 2}, and {self.encoder_reduction // 4}."
                )
                self.refiner = upsample_block(
                    in_channels=self.encoder_channels,
                    out_channels=refiner_out_channels[self.model_name],
                    norm_layer=norm_layer,
                    activation=activation,
                )

            self.refiner_channels = refiner_out_channels[self.model_name]

        else:
            self.refiner = nn.Identity()
            self.refiner_channels = self.encoder_channels

        self.refiner_reduction = self.block_size

        if self.refiner_channels <= 256:
            self.decoder = nn.Identity()
            self.decoder_channels = self.refiner_channels
        elif self.refiner_channels <= 512:
            self.decoder = decoder_block(
                in_channels=self.refiner_channels,
                out_channels=self.refiner_channels // 2,
                norm_layer=norm_layer,
                activation=activation,
            )
            self.decoder_channels = self.refiner_channels // 2
        elif self.refiner_channels <= 1024:
            self.decoder = nn.Sequential(
                decoder_block(
                    in_channels=self.refiner_channels,
                    out_channels=self.refiner_channels // 2,
                    norm_layer=norm_layer,
                    activation=activation,
                ),
                decoder_block(
                    in_channels=self.refiner_channels // 2,
                    out_channels=self.refiner_channels // 4,
                    norm_layer=norm_layer,
                    activation=activation,
                ),
            )
            self.decoder_channels = self.refiner_channels // 4
        else:
            self.decoder = nn.Sequential(
                decoder_block(
                    in_channels=self.refiner_channels,
                    out_channels=self.refiner_channels // 2,
                    norm_layer=norm_layer,
                    activation=activation,
                ),
                decoder_block(
                    in_channels=self.refiner_channels // 2,
                    out_channels=self.refiner_channels // 4,
                    norm_layer=norm_layer,
                    activation=activation,
                ),
                decoder_block(
                    in_channels=self.refiner_channels // 4,
                    out_channels=self.refiner_channels // 8,
                    norm_layer=norm_layer,
                    activation=activation,
                ),
            )
            self.decoder_channels = self.refiner_channels // 8

        self.decoder_reduction = self.refiner_reduction

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)[0]

    def refine(self, x: Tensor) -> Tensor:
        return self.refiner(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.refine(x)
        x = self.decode(x)
        return x

    def restore_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        weights = ckpt["model"]
        # Strip "backbone." prefix; rename ut_norms -> out_norms (older checkpoint naming)
        backbone_weights = {}
        for key, value in weights.items():
            if not key.startswith("backbone."):
                continue
            key = key[len("backbone."):]
            if key.startswith("ut_norms."):
                key = "out_norms." + key[len("ut_norms."):]
            backbone_weights[key] = value
        self.encoder.load_state_dict(backbone_weights, strict=True)

def _timm_model(
    model_name: str,
    block_size: Optional[int] = None,
    norm: str = "none",
    act: str = "none",
) -> TIMMModel:
    return TIMMModel(model_name, block_size=block_size, norm=norm, act=act)
