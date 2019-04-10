# code borrow from https://github.com/Lextal/pspnet-pytorch
import torch
from harmonic import SinConv
from torch import nn
from torch.nn import functional as F

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class PSPUpsampleS(nn.Module):
    def __init__(self, in_channels, out_channels, sins):
        super(PSPUpsampleS, self).__init__()
        self.conv = nn.Sequential(
            SinConv(in_channels, out_channels, sins=sins, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self, x, y=None):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        if y is not None:
            p = torch.cat([p, y], dim=1)
        return self.conv(p)


def freeze_layer(layer, freeze=False):
    for param in layer.parameters():
        param.requires_grad = freeze


class PSPNet(nn.Module):
    def __init__(self, sins, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet152',
                 pretrained=True, path=None, extended=False):
        super(PSPNet, self).__init__()
        assert (len(sins) > 0)
        self.feats = getattr(extractors, backend)(pretrained, path)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        if extended:
            self.up_1 = PSPUpsampleS(1024, 192, sins=sins)
        else:
            self.up_1 = PSPUpsampleS(1024, 256, sins=sins)
        self.up_2 = PSPUpsampleS(256, 128, sins=sins)
        self.up_3 = PSPUpsampleS(128, 128, sins=sins)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = SinConv(128, len(sins), sins=sins, kernel_size=1)
        self.extended = extended

    def freeze_encoder(self, freeze):
        freeze_layer(self.feats, freeze)

    def forward(self, x):
        f, fc = self.feats(x)
        p = self.psp(f)
        p = self.up_1(p)
        if self.extended:
            p = self.up_2(p,fc)
        else:
            p = self.up_2(p)
        p = self.up_3(p)
        return self.final(p)
