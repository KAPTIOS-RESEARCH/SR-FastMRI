import torch.nn as nn
from src.utils.device import get_available_device
from src.optimisation.losses.edge import SobelFilter
from src.optimisation.losses.perceptual import PerceptualLoss

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        device = get_available_device()
        self.l1_loss = nn.L1Loss()
        self.edge_detector = SobelFilter().to(device)
        self.perceptual_loss = PerceptualLoss().to(device)

    def preprocess_for_vgg(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def edge_loss(self, x, y):
        x_edge = self.edge_detector(x)
        y_edge = self.edge_detector(y)
        return self.l1_loss(x_edge, y_edge)

    def pixel_loss(self, x, y):
        return self.l1_loss(x, y)

    def feature_loss(self, x, y):
        x = self.preprocess_for_vgg(x)
        y = self.preprocess_for_vgg(y)
        return self.perceptual_loss(x, y)

    def forward(self, x, y):
        return {
            'edge_loss': self.edge_loss(x, y),
            'pixel_loss': self.pixel_loss(x, y),
            'feature_loss': self.feature_loss(x, y)
        }
        
class EPFLoss(nn.Module):
    def __init__(self, lambda_1 = 0.7, lambda_2 = 0.3, lambda_3 = 1):
        super(EPFLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.content = ContentLoss()
    
    def forward(self, x, y):
        content_loss = self.content(x, y)
        edge_loss = self.lambda_1 * content_loss['edge_loss']
        pixel_loss = self.lambda_2 * content_loss['pixel_loss']
        feature_loss = self.lambda_3 * content_loss['feature_loss']
        return edge_loss + pixel_loss + feature_loss