import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PerceptualLoss(nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output
    """
    def __init__(self, conv_index: str = '22'):

        super(PerceptualLoss, self).__init__()
        vgg_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features
        modules = [m for m in vgg_features]
        
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        def _forward(x):
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)

        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss