from torch import nn
from src.utils.device import get_available_device
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSIM
from src.optimisation.losses.edge import SobelFilter

class L1SSIM(nn.Module):
    def __init__(self):
        super(L1SSIM, self).__init__()
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


class L1MSSIMEdgeLoss(nn.Module):
    """Implementation from Loss Functions for Image Restoration with Neural Networks"""
    def __init__(self, alpha: float = 0.84):
        super(L1MSSIMEdgeLoss, self).__init__()
        device = get_available_device()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mssim = MSSIM().to(device)
        self.edge_detector = SobelFilter()
        
    def edge_loss(self, x, y):
        x_edge = self.edge_detector(x)
        y_edge = self.edge_detector(y)
        return self.l1_loss(x_edge, y_edge)
    
    def forward(self, x, y):
        edge_loss = self.edge_loss(x, y)
        l1_loss = self.l1_loss(x, y)
        mssim_loss = 1 - self.mssim(x, y)
        lamda = (1-self.alpha)/2
        return self.alpha * mssim_loss + lamda * l1_loss + lamda * edge_loss