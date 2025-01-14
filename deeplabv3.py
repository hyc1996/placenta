import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2, pretrained_backbone=True):
        super(DeepLabV3Plus, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(pretrained_backbone=pretrained_backbone)
        self.deeplabv3.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        return self.deeplabv3(x)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, 256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=None):
        super(ASPP, self).__init__()
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        layers = []

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for rate in atrous_rates:
            layers.append(ASPPConv(in_channels, out_channels, rate))

        self.convs = nn.ModuleList(layers)
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x_pool = self.global_pooling(x)
        x_pool = F.interpolate(x_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_aspp = [x_pool] + [conv(x) for conv in self.convs]
        x = torch.cat(x_aspp, dim=1)
        return self.out_conv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


if __name__ == '__main__':
    model = DeepLabV3Plus(num_classes=2)
    print(model)


