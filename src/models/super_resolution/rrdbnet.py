import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layers(block, n_layers):
    """
    Creates a sequential container with a specified number of identical layers.

    Args:
        block (nn.Module): The block to repeat.
        n_layers (int): The number of layers to create.

    Returns:
        nn.Sequential: A sequential container of the repeated blocks.
    """
    return nn.Sequential(*(block() for _ in range(n_layers)))

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block with 5 convolutional layers.

    Args:
        in_channels (int): Number of feature maps.
        out_channels (int): Growth channel (intermediate channels).
    """
    def __init__(self, in_channels = 64, out_channels = 32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(in_channels + 2 * out_channels, out_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(in_channels + 3 * out_channels, out_channels, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(in_channels + 4 * out_channels, in_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block composed of three ResidualDenseBlock blocks.

    Args:
        in_channels (int): Number of feature maps.
        out_channels (int): Growth channel.
    """
    def __init__(self, in_channels = 32, out_channels = 32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, out_channels)
        self.RDB2 = ResidualDenseBlock(in_channels, out_channels)
        self.RDB3 = ResidualDenseBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
class RRDBNet(nn.Module):
    """
    Network structure for image super-resolution based on Residual in Residual Dense Blocks.

    Args:
        in_nc (int): Number of input channels.
        out_nc (int): Number of output channels.
        nf (int): Number of feature maps.
        nb (int): Number of RRDB blocks.
        gc (int): Growth channel.
    """
    def __init__(self, in_channels, out_channels, n_feature_maps, n_blocks, growth_channel = 32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, in_channels = n_feature_maps, out_channels = growth_channel)

        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, n_feature_maps, 3, 1, 1, bias=True)

        # Residual in Residual Dense Blocks
        self.RRDB_trunk = make_layers(RRDB_block_f, n_blocks)
        self.trunk_conv = nn.Conv2d(n_feature_maps, n_feature_maps, 3, 1, 1, bias=True)

        # Upsampling layers
        self.upconv1 = nn.Conv2d(n_feature_maps, n_feature_maps, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(n_feature_maps, n_feature_maps, 3, 1, 1, bias=True)

        # High-resolution and final output layers
        self.HRconv = nn.Conv2d(n_feature_maps, n_feature_maps, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(n_feature_maps, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Initial feature extraction
        fea = self.conv_first(x)

        # Pass through RRDB trunk and add residual
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea += trunk

        # Upsampling by factor of 2 twice
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        # Final high-resolution output
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
