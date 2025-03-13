import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2, d=56, s=12, m=4):
        """
        scale_factor: Upscaling factor (e.g., 2, 3, or 4).
        d: Number of filters in the feature extraction layer.
        s: Number of filters in the shrinking layer.
        m: Number of mapping layers.
        """
        super(FSRCNN, self).__init__()
        
        # Feature extraction layer
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU()
        )
        
        # Shrinking layer
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )
        
        # Mapping layers
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU())
        self.mapping = nn.Sequential(*mapping_layers)
        
        # Expanding layer
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )
        
        # Deconvolution layer
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale_factor, padding=4, output_padding=1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x