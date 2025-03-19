import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x  # Residual scaling

class RRDB(nn.Module):
    def __init__(self, channels=64, growth=32):
        super(RRDB, self).__init__()
        self.rrdb_layers = nn.Sequential(
            ResidualDenseBlock(channels, growth),
            ResidualDenseBlock(channels, growth),
            ResidualDenseBlock(channels, growth)
        )
    
    def forward(self, x):
        out = self.rrdb_layers(x)
        return out * 0.2 + x

class RealESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23, upscale_factor=4):
        super(RealESRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        upsample_layers = []
        for _ in range(int(upscale_factor // 2)):
            upsample_layers.append(nn.Conv2d(num_features, num_features * 4, 3, 1, 1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Conv2d(num_features, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.rrdb_blocks(x1)
        x3 = self.conv2(x2) + x1  # Residual connection
        x4 = self.upsample(x3)
        out = self.conv3(x4)
        return out