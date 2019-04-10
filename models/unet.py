import torch
import torch.nn as nn
import torch.nn.functional as F

from harmonic import SinConv


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class ups(nn.Module):
    def __init__(self, in_ch, out_ch, sins):
        super(ups, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        self.sinconv = SinConv(in_ch, out_ch, sins=sins, kernel_size=7, padding=3)
        self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2):
        #x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = F.upsample(x1, scale_factor=2, mode='bilinear', align_corners=True)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.sinconv(x)
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sins):
        super(outconv, self).__init__()
        self.sinconv = SinConv(in_ch, in_ch, sins=sins, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.sinconv(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, sins):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = ups(1024, 256, sins)
        self.up2 = ups(512, 128, sins)
        self.up3 = ups(256, 64, sins)
        self.up4 = ups(128, 64, sins)
        self.outc = outconv(64, len(sins), sins)

    def freeze_encoder(self, freeze):
        return None

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
        x = self.outc(x)
        return x
