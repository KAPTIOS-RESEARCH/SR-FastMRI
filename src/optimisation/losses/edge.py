from torch import nn
import torch

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
    
class LaplacianFilter(nn.Module):
    def __init__(self):
        super(LaplacianFilter, self).__init__()
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        laplacian_weights = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplacian.weight = nn.Parameter(laplacian_weights.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        return self.laplacian(x)