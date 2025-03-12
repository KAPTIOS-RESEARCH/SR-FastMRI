import torch
import torch.nn as nn

# -----------------------------------
# Basic Residual Block for EDSR
# -----------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_scale = 0.1
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

# -----------------------------------
# Upsampler using PixelShuffle
# -----------------------------------
class Upsampler(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        for _ in range(int(scale / 2)):
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.ReLU(inplace=True))
        super(Upsampler, self).__init__(*m)


# -----------------------------------
# EDSR Model
# -----------------------------------
class EDSR(nn.Module):
    def __init__(self, in_channels = 1, scale_factor=2, num_blocks=32, num_feats=256):
        super(EDSR, self).__init__()
        head_layer_list = [nn.Conv2d(in_channels, num_feats, kernel_size=3, stride=1, padding=1, bias=True)]
        body_layer_list = [ResidualBlock(num_feats, num_feats) for _ in range(num_blocks)]
        body_layer_list.append(nn.Conv2d(num_feats, num_feats, 3, 1, 1))
        tail_layer_list = [Upsampler(scale_factor, num_feats), nn.Conv2d(num_feats, in_channels, 3, 1, 1)]
        
        self.head = nn.Sequential(*head_layer_list)
        self.body = nn.Sequential(*body_layer_list)
        self.tail = nn.Sequential(*tail_layer_list)
        
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x