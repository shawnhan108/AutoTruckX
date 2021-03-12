import torch.nn as nn
from collections import OrderedDict

# Network extracted directly from 
# https://github.com/gupta-abhay/setr-pytorch/blob/ddb63efbd77b08f1aca7d8d7302c7fd231288be8/setr/ResNet.py

class PreActBottleneck(nn.Module):
    # Pre-activation (v2) bottleneck block.
    # "Identity Mappings in Deep Residual Networks", https://arxiv.org/pdf/1603.05027.pdf.

    def __init__(self, in_planes, out_planes=None, mid_planes=None, stride=1):
        super(PreActBottleneck, self).__init__()
        out_planes = out_planes or in_planes
        mid_planes = mid_planes or out_planes // 4

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes,mid_planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))

        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        return out + residual


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843):
        super(ResNetV2, self).__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=64*wf, out_planes=256*wf, mid_planes=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=256*wf, out_planes=256*wf, mid_planes=64*wf)) for i in range(2, block_units[0] + 1)],
        ))
        self.conv3 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=256*wf, out_planes=512*wf, mid_planes=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=512*wf, out_planes=512*wf, mid_planes=128*wf)) for i in range(2, block_units[1] + 1)],
        ))
        self.conv4 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=512*wf, out_planes=1024*wf, mid_planes=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=1024*wf, out_planes=1024*wf, mid_planes=256*wf)) for i in range(2, block_units[2] + 1)],
        ))
        self.conv5 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=1024*wf, out_planes=2048*wf, mid_planes=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=2048*wf, out_planes=2048*wf, mid_planes=512*wf)) for i in range(2, block_units[3] + 1)],
        ))

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.BatchNorm2d(2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))

    def forward(self, x, include_conv5=False, include_top=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if include_conv5:
            x = self.conv5(x)
        if include_top:
            x = self.head(x)

        if include_top and include_conv5:
            assert x.shape[-2:] == (1, 1,)
            return x[..., 0, 0]

        return x
