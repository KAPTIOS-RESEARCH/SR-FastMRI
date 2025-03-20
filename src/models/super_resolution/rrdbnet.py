import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth=32):
        super(ResidualDenseBlock, self).__init__()
        self.growth = growth
        self.conv1 = nn.Conv2d(in_channels, growth, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels + growth, growth, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth, growth, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth, growth, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth, in_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2 
    
class RRDB(nn.Module):
    def __init__(self, in_channels=64, growth=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth)
        self.rdb2 = ResidualDenseBlock(in_channels, growth)
        self.rdb3 = ResidualDenseBlock(in_channels, growth)
        self.concat_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.rdb1(x)
        x2 = self.rdb2(x1)
        x3 = self.rdb3(x2)
        return self.concat_conv(torch.cat([x1, x2, x3], dim=1))

class RRDBNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, growth_channels=32, upscale_factor=2):
        super(RRDBNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        )
        self.rrdb = RRDB(num_features, growth_channels)
        self.out_conv = nn.Conv2d(num_features, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.rrdb(x1)
        x3 = self.out_conv(x2)
        x4 = self.upscale(x3)
        return x4