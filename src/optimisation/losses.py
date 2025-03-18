import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from src.utils.device import get_available_device
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSIM

# Sobel filter for edge detection
class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x_weights = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_weights = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.sobel_x.weight = nn.Parameter(sobel_x_weights.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weights.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2)

# VGG19 feature extractor for perceptual loss
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg19.children())[:35])  # Up to the 4th conv block
        for param in self.features.parameters():
            param.requires_grad = False

        # Normalization for VGG
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        return self.features(x)

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        device = get_available_device()
        self.l1_loss = nn.L1Loss()
        self.edge_detector = SobelFilter().to(device)
        self.feature_extractor = VGG19FeatureExtractor().to(device)

    def preprocess_for_vgg(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def edge_loss(self, A, B):
        A_edge = self.edge_detector(A)
        B_edge = self.edge_detector(B)
        return self.l1_loss(A_edge, B_edge)

    def pixel_loss(self, A, B):
        return self.l1_loss(A, B)

    def feature_loss(self, A, B):
        A = self.preprocess_for_vgg(A)
        B = self.preprocess_for_vgg(B)
        A_feat = self.feature_extractor(A)
        B_feat = self.feature_extractor(B)
        return self.l1_loss(A_feat, B_feat)

    def forward(self, A, B):
        return {
            'edge_loss': self.edge_loss(A, B),
            'pixel_loss': self.pixel_loss(A, B),
            'feature_loss': self.feature_loss(A, B)
        }

class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        self.lambda_1 = 0.7
        self.lambda_2 = 0.3
        self.lambda_3 = 1
        self.content = ContentLoss()
    
    def forward(self, x, y):
        content_loss = self.content(x, y)
        edge_loss = self.lambda_1 * content_loss['edge_loss']
        pixel_loss = self.lambda_2 * content_loss['pixel_loss']
        feature_loss = self.lambda_3 * content_loss['feature_loss']
        return edge_loss + pixel_loss + feature_loss


class SSIMMixedLoss(nn.Module):
    def __init__(self):
        super(SSIMMixedLoss, self).__init__()
        device = get_available_device()
        self.lambda_1 = 0.7
        self.lambda_2 = 0.3
        self.lambda_3 = 1
        self.content = ContentLoss()
        self.ssim = SSIM().to(device)
    
    def forward(self, x, y):
        content_loss = self.content(x, y)
        ssim_loss = 1 - self.ssim(x, y)
        edge_loss = self.lambda_1 * content_loss['edge_loss']
        pixel_loss = self.lambda_2 * content_loss['pixel_loss']        
        feature_loss = self.lambda_3 * ssim_loss
        return edge_loss + pixel_loss + feature_loss


class SRUNetLoss(nn.Module):
    def __init__(self):
        super(SRUNetLoss, self).__init__()
        device = get_available_device()
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM().to(device)
        
    def forward(self, x, y):
        l1_loss = self.l1_loss(x, y)
        ssim_loss = 1 - self.ssim(x, y)
        return l1_loss + 0.1 * ssim_loss
    
class L1MSSIMLoss(nn.Module):
    """Implementation from Loss Functions for Image Restoration with Neural Networks"""
    def __init__(self, alpha: float = 0.84):
        super(L1MSSIMLoss, self).__init__()
        device = get_available_device()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mssim = MSSIM().to(device)
        
    def forward(self, x, y):
        l1_loss = self.l1_loss(x, y)
        mssim_loss = 1 - self.mssim(x, y)
        return self.alpha * mssim_loss + (1 - self.alpha) * l1_loss
    
class L1EdgeMSSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.84):
        super(L1EdgeMSSIMLoss, self).__init__()
        device = get_available_device()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mssim = MSSIM().to(device)
        self.edge_detector = SobelFilter().to(device)
        
    def edge_loss(self, A, B):
        A_edge = self.edge_detector(A)
        B_edge = self.edge_detector(B)
        return self.l1_loss(A_edge, B_edge)
    
    def forward(self, x, y):
        edge_loss = 0.7 * self.edge_loss(x, y)
        l1_loss = self.l1_loss(x, y)
        mssim_loss = 1 - self.mssim(x, y)
        return self.alpha * mssim_loss + (1 - self.alpha) * l1_loss + edge_loss
    
