""" Parts of the U-Net enc """

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_norm=False, use_bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_norm:
            NormLayer = ChannelNorm
        else:
            NormLayer = nn.Identity

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=use_bias),
            NormLayer(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            NormLayer(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownWithStride(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_norm=False, use_bias=True):
        super().__init__()
        if use_norm:
            NormLayer = ChannelNorm
        else:
            NormLayer = nn.Identity
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=use_bias),
            NormLayer(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            NormLayer(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_norm=False, use_bias=True):
        super().__init__()
        if use_norm:
            NormLayer = ChannelNorm
        else:
            NormLayer = nn.Identity

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.norm_layer1 = NormLayer(in_channels // 2)
        self.norm_layer2 = NormLayer(out_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_bias=use_bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.norm_layer1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.norm_layer2(x)
        return x


class UpSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class UpBilinear(nn.Module):
    def __init__(self, in_channels, out_channels, out_height=None, out_width=None):
        super().__init__()
        if out_width is not None and out_height is not None:
            self.up = nn.Upsample((out_height, out_width), mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, use_bias=True):
        super(OutConv, self).__init__()
        if use_norm:
            NormLayer = ChannelNorm
        else:
            NormLayer = nn.Identity
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)
        self.norm_layer = NormLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        return x


class ChannelNorm(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.norm = nn.LayerNorm(nchannels)

    def forward(self, x):
        assert x.dim() == 4
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
