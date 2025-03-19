from src.models.super_resolution.resrgan import RealESRGAN
from torchsummary import summary

model = RealESRGAN(1, 1, num_features=48, num_blocks=16, upscale_factor=2)
model.to('cuda')
summary(model, (1, 128, 128))