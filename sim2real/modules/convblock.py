import torch

from sim2real.modules.film import FiLM
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        film: bool = False,
        freeze_film: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        padding = (kernel - 1) // 2

        self.activation = nn.ReLU()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)

        self.residual = residual

        if film:
            self.affine = FiLM(in_channels, freeze=freeze_film)
        else:
            self.affine = nn.Identity()

        if residual:
            self.residual = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor):
        h = self.conv(self.activation(self.affine(x)))
        if self.residual:
            return h + self.residual(x)
        else:
            return h


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        affine: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()

        padding = (kernel - 1) // 2

        self.activation = nn.ReLU()
        self.residual = residual

        if affine:
            self.affine1 = FiLM(in_channels)
            self.affine2 = FiLM(out_channels)
        else:
            self.affine1 = nn.Identity()
            self.affine2 = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride=1, padding=padding
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel, stride=1, padding=padding
        )

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.activation(self.affine1(x)))
        h = self.conv2(self.activation(self.affine2(h)))
        if self.residual:
            return h + self.residual(x)
        else:
            return h


from torch import nn
import torch


class TransposeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        # TODO: does this if do anything?
        if stride > 1:
            output_padding = stride // 2

        padding = kernel_size // 2

        self.transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transpose(x)
        return h
