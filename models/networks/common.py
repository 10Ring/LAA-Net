#-*- coding: utf-8 -*-
import torch
import torch.nn as nn


BN_MOMENTUM = 0.1


def point_wise_block(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )


def conv_block(inplanes, outplanes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class InceptionBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, pool_size=3):
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.pool_size = pool_size
        super(InceptionBlock, self).__init__()

        self.pw_block = point_wise_block(self.inplanes, self.outplanes//4)
        self.mp_layer = nn.MaxPool2d(kernel_size=self.pool_size, stride=stride, padding=1)
        self.conv3_block = conv_block(self.outplanes//4, self.outplanes//4, kernel_size=3, stride=1, padding=1)
        self.conv5_block = conv_block(self.outplanes//4, self.outplanes//4, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x1 = self.pw_block(x)

        x2 = self.pw_block(x)
        x2 = self.conv3_block(x2)

        x3 = self.pw_block(x)
        x3 = self.conv5_block(x3)

        x4 = self.mp_layer(x)
        x4 = self.pw_block(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
