import torch 
import torch.nn as nn 
import numpy as np

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                        padding=0, stride=1, use_bn=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        padding=padding, stride=stride, bias= not(use_bn))
        relu = nn.ReLU(inplace=True)
        
        if use_bn:
            bn = nn.BatchNorm2d(out_channels)
            super(Conv2dReLU, self).__init__(conv, bn, relu)
        else:
            super(Conv2dReLU, self).__init__(conv, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_bn=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3,
                                padding=1, use_bn=use_bn)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))

        return x

class Decoder(nn.Module):
    def __init__(self, dim, decoder_channels, skip_num, skip_channels):
        super().__init__()
        head_channel_num = 512
        self.conv = Conv2dReLU(dim, head_channel_num, kernel_size=3, padding=1, use_bn=True)
        self.skip_num = skip_num

        in_channels = [head_channel_num] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if skip_num:
            for i in range(4-skip_num):
                skip_channels[3-i] = 0 
        else:
            skip_channels = [0] * 4
        
        self.decoder = nn.ModuleList([
            DecoderBlock(i, o, s) for i, o, s in zip(in_channels, out_channels, skip_channels)
        ])
    
    def forward(self, x, features=None):
        B, pd, D = x.size()               # N x HW/pd^2 x D
        x = x.permute(0, 2, 1).contiguous().view(B, D, int(np.sqrt(pd)), int(np.sqrt(pd))) # N x dim x H/pd x W/pd
        x = self.conv(x) 

        # start adding connections from encoder features
        for i, block in enumerate(self.decoder):
            skip = features[i] if ((features is not None) and (i < self.skip_num)) else None
            x = block(x, skip=skip)
        
        return x
