""" Full assembly of the parts to form the complete network """

from .model_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_norm=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownWithStride(64, 128, use_norm=use_norm)
        self.down2 = DownWithStride(128, 256, use_norm=use_norm)
        self.down3 = DownWithStride(256, 512, use_norm=use_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownWithStride(512, 1024 // factor, use_norm=use_norm)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetx3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_norm=False):
        super(UNetx3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64, use_norm=use_norm)
        self.down1 = DownWithStride(64, 128, use_norm=use_norm)
        self.down2 = DownWithStride(128, 256, use_norm=use_norm)
        self.down3 = DownWithStride(256, 512 // factor, use_norm=use_norm)
        self.up2 = Up(512, 256 // factor, bilinear, use_norm=use_norm)
        self.up3 = Up(256, 128 // factor, bilinear, use_norm=use_norm)
        self.up4 = Up(128, 64, bilinear, use_norm=use_norm)

        self.outc = OutConv(64, n_classes, use_norm=use_norm)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownWithStride(64, 128)
        self.down2 = DownWithStride(128, 256)
        self.down3 = DownWithStride(256, 512)
        self.down4 = DownWithStride(512, latent_dim)

        self.seq = torch.nn.Sequential(
            self.inc,
            self.down1,
            self.down2,
            self.down3,
            self.down4
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Encoderx4(nn.Module):
    def __init__(self, in_channels, latent_dim, use_norm=False):
        super(Encoderx4, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.inc = DoubleConv(in_channels, 64, use_norm=use_norm)
        self.down1 = DownWithStride(64, 128, use_norm=use_norm)
        self.down2 = DownWithStride(128, latent_dim, use_norm=use_norm)

        self.seq = torch.nn.Sequential(
            self.inc,
            self.down1,
            self.down2,
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Encoderx8(nn.Module):
    def __init__(self, in_channels, latent_dim, use_norm=False, use_bias=True):
        super(Encoderx8, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.inc = DoubleConv(in_channels, 64, use_norm=use_norm, use_bias=use_bias)
        self.down1 = DownWithStride(64, 128, use_norm=use_norm, use_bias=use_bias)
        self.down2 = DownWithStride(128, 256, use_norm=use_norm, use_bias=use_bias)
        self.down3 = DownWithStride(256, latent_dim, use_norm=use_norm, use_bias=use_bias)

        self.seq = torch.nn.Sequential(
            self.inc,
            self.down1,
            DoubleConv(128, 128, use_norm=use_norm, use_bias=use_bias),
            self.down2,
            DoubleConv(256, 256, use_norm=use_norm, use_bias=use_bias),
            self.down3,
            DoubleConv(latent_dim, latent_dim, use_norm=use_norm, use_bias=use_bias),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Encoderx8Simp(nn.Module):
    def __init__(self, in_channels, latent_dim, use_norm=False, use_bias=True):
        super(Encoderx8Simp, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.inc = DoubleConv(in_channels, 64, use_norm=use_norm, use_bias=use_bias)
        self.down1 = DownWithStride(64, 128, use_norm=use_norm, use_bias=use_bias)
        self.down2 = DownWithStride(128, 256, use_norm=use_norm, use_bias=use_bias)
        self.down3 = DownWithStride(256, latent_dim, use_norm=use_norm, use_bias=use_bias)

        self.seq = torch.nn.Sequential(
            self.inc,
            self.down1,
            self.down2,
            self.down3,
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.up1 = UpBilinear(latent_dim, 512)
        self.up2 = UpBilinear(512, 256)
        self.up3 = UpBilinear(256, 128)
        self.up4 = UpBilinear(128, 64)
        self.outc = OutConv(64, out_channels)
        self.seq = torch.nn.Sequential(
            self.up1,
            self.up2,
            self.up3,
            self.up4,
            self.outc
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Decoderx8(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoderx8, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.up2 = UpBilinear(latent_dim, 256)
        self.up3 = UpBilinear(256, 128)
        self.up4 = UpBilinear(128, 64)
        self.outc = OutConv(64, out_channels)
        self.seq = torch.nn.Sequential(
            self.up2,
            self.up3,
            self.up4,
            self.outc
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class KQ_FFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            ChannelNorm(in_channels),
            nn.LeakyReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            ChannelNorm(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, ),
            ChannelNorm(in_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = x + self.layer1(x)
        x = x + self.layer2(x)
        return x


if __name__ == "__main__":
    enc = Encoder(3, 128)
    dec = Decoder(128, 3)
    input = torch.rand(2, 3, 384, 512)
    output = enc(input)
    pred = dec(output)
    print(output.shape, pred.shape)
