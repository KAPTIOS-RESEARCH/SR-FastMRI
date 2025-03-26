import numpy as np
import torch
from typing import Tuple, Union
import torch.nn.functional as F

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def resize_tensor(tensor, new_size=(256, 256)):
    """
    Resizes a 2D tensor to the specified size using bilinear interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape [H, W].
        new_size (tuple): Desired output size (height, width).

    Returns:
        torch.Tensor: Resized tensor of shape [new_size[0], new_size[1]].
    """
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
    return resized_tensor.squeeze(0).squeeze(0)

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def apply_low_pass_filter_torch(original_kspace: torch.Tensor, radius: float = 50.):
    """
    Low pass filter removes high spatial frequencies from k-space.

    Parameters:
        original_kspace (torch.Tensor): The k-space sample to filter of shape (H, W, 2)
        radius (float): Relative size of the k-space mask circle (percent)

    Returns:
        lr_kspace (torch.Tensor): The low-pass filtered k-space
        lr_image (torch.Tensor): The low-pass filtered image
    """
    lr_kspace = original_kspace.clone()
    H, W, _ = lr_kspace.shape
    
    if radius < 100:
        r = (np.hypot(H, W) / 2) * (radius / 100)
        y, x = torch.meshgrid(torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing='ij')
        mask = (x ** 2 + y ** 2) <= r ** 2
        mask = mask.unsqueeze(-1).expand(-1, -1, 2)
        lr_kspace[~mask] = 0
    
    return lr_kspace

def add_gaussian_noise_torch(kspace: torch.Tensor, snr_db: float = 20.):
    """
    Add Gaussian noise to the non-zero part of the k-space based on the desired SNR.

    Parameters:
        kspace (torch.Tensor): The low-pass filtered k-space of shape (H, W, 2)
        snr_db (float): Desired signal-to-noise ratio in dB.

    Returns:
        noisy_kspace (torch.Tensor): k-space with added Gaussian noise.
        noisy_image (torch.Tensor): Image with added Gaussian noise.
    """
    mask = (kspace != 0).any(dim=-1)  # Find non-zero elements in k-space
    kspace_nonzero = kspace[mask]
    
    signal_power = torch.mean(kspace_nonzero[..., 0]**2 + kspace_nonzero[..., 1]**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    noise_real = torch.sqrt(noise_power / 2) * torch.randn_like(kspace_nonzero[..., 0])
    noise_imag = torch.sqrt(noise_power / 2) * torch.randn_like(kspace_nonzero[..., 1])
    
    noise = torch.stack((noise_real, noise_imag), dim=-1)
    
    noisy_kspace = kspace.clone()
    noisy_kspace[mask] += noise
    return noisy_kspace