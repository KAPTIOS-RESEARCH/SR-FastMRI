from typing import NamedTuple, Tuple, Dict
import numpy as np
import torch
from .math import *
from src.data.utils.preprocessing import *
from .fft import *

class SRSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        max_value: Maximum image value
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    max_value: float

class SuperResolutionTransform:
    """Creates a low-resolution image of the kspace using a low-pass filter and gaussian noise
    """
    def __call__(
        self,
        kspace: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        scaling_factor: int = 2,
        low_pass_radius: float = 30., 
        target_snr: float = 20.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        
        kspace_tensor = to_tensor(kspace)
        target_tensor = to_tensor(target)

        kspace_tensor = apply_low_pass_filter_torch(kspace_tensor, low_pass_radius)
        kspace_tensor = add_gaussian_noise_torch(kspace_tensor, target_snr)

        image_tensor = ifft2c_new(kspace_tensor)
        crop_size = (target_tensor.shape[0], target_tensor.shape[1])
        image_tensor = complex_center_crop(image_tensor, crop_size)
        image_tensor = complex_abs(image_tensor)
        image_tensor = resize_tensor(image_tensor, (crop_size[0]//scaling_factor, crop_size[1]//scaling_factor))
        image_tensor, mean, std = normalize_instance(image_tensor, eps=1e-11)
        image_tensor = image_tensor.clamp(-6, 6)

        target_torch = center_crop(target_tensor, crop_size)
        target_torch = normalize(target_torch, mean, std, eps=1e-11)
        target_torch = target_torch.clamp(-6, 6)
        
        return SRSample(image_tensor.unsqueeze(0), target_torch.unsqueeze(0), mean, std, max_value)