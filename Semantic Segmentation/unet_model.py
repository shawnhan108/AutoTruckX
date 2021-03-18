import torch 
import torch.nn as nn


class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.down1 = self.conv(in_channels=3, out_channels=64)
        self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = self.conv(in_channels=64, out_channels=128)
        self.down2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = self.conv(in_channels=128, out_channels=256)
        self.down3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = self.conv(in_channels=256, out_channels=512)
        self.down4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = self.conv(in_channels=512, out_channels=1024)

        self.up1_deconv = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1 = self.conv(in_channels=1024, out_channels=512)
        self.up2_deconv = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = self.conv(in_channels=512, out_channels=256)
        self.up3_deconv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = self.conv(in_channels=256, out_channels=128)
        self.up4_deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = self.conv(in_channels=128, out_channels=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )
    
    def forward(self, x):
        down1out = self.down1(x)
        down2out = self.down2(self.down1_pool(down1out))
        down3out = self.down3(self.down2_pool(down2out))
        down4out = self.down4(self.down3_pool(down3out))

        midout = self.middle(self.down4_pool(down4out))

        up1out = self.up1_deconv(midout)
        up2out = self.up2_deconv(self.up1(torch.cat((up1out, down4out), dim=1)))
        up3out = self.up3_deconv(self.up2(torch.cat((up2out, down3out), dim=1)))
        up4out = self.up4_deconv(self.up3(torch.cat((up3out, down2out), dim=1)))

        output = self.output(self.up4(torch.cat((up4out, down1out), dim=1)))

        return output

"""
t = UNet(13)
print(t)
print(sum(p.numel() for p in t.parameters()))
x = torch.randn(16, 3, 256, 256)
print(x.size())
x = t(x)
print(x.size())
"""
