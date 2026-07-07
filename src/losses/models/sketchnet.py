'''
https://github.com/Mukosame/Anime2Sketch?tab=readme-ov-file
'''

from typing import Optional, List

import torch
from torch import nn


class UNetBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        input_channels: Optional[int] = None,
        submodule: Optional[nn.Module] = None,
        is_outermost: bool = False,
        is_innermost: bool = False,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.is_outermost = is_outermost
        if input_channels is None:
            input_channels = out_channels

        self.model = nn.Sequential(*self._build_layers(
            input_channels, out_channels, in_channels,
            submodule, is_outermost, is_innermost, use_dropout
        ))

    def _build_layers(
        self,
        input_channels: int,
        out_channels: int,
        in_channels: int,
        submodule: Optional[nn.Module],
        is_outermost: bool,
        is_innermost: bool,
        use_dropout: bool
    ) -> List[nn.Module]:
        down_conv = nn.Conv2d(input_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=True)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.InstanceNorm2d(in_channels, affine=False, track_running_stats=False)

        up_relu = nn.ReLU(True)
        up_norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

        if is_outermost:
            up_conv = nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1)
            return [down_conv, submodule, up_relu, up_conv, nn.Tanh()]
        elif is_innermost:
            up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
            return [down_relu, down_conv, up_relu, up_conv, up_norm]
        else:
            up_conv = nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
            layers = [down_relu, down_conv, down_norm, submodule, up_relu, up_conv, up_norm]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)


class SketchNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_downsampling: int,
        base_filters: int = 64,
        use_dropout: bool = False,
    ):
        super().__init__()
        block = UNetBlock(base_filters * 8, base_filters * 8, is_innermost=True)
        for _ in range(num_downsampling - 5):
            block = UNetBlock(base_filters * 8, base_filters * 8, submodule=block, use_dropout=use_dropout)
        block = UNetBlock(base_filters * 4, base_filters * 8, submodule=block)
        block = UNetBlock(base_filters * 2, base_filters * 4, submodule=block)
        block = UNetBlock(base_filters, base_filters * 2, submodule=block)
        self.model = UNetBlock(out_channels, base_filters, input_channels=in_channels, submodule=block, is_outermost=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
