
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

class FastMRIImageManager:
    """This code was based on the work in https://github.com/birogeri/kspace-explorer"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        file_list = list(self.data_path.glob("*.h5"))
        self.data_samples = self.__load_data(file_list)

    def __load_data(self, file_list):
        samples = []
        for fname in file_list:
            with h5py.File(fname, "r") as hf:
                num_slices = hf['reconstruction_rss'].shape[0]
                for i in range(num_slices):
                    samples.append((fname, i))
        return samples    
    
    def get_kspace_sample(self, idx):
        fname, slice_idx = self.data_samples[idx]
        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][slice_idx]
        return kspace

    def ifft(self, kspace: np.ndarray):
        """Performs inverse FFT function (kspace to [magnitude] image)

        Performs iFFT on the input data and updates the display variables for
        the image domain (magnitude) image and the kspace as well.

        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            out (np.ndarray): Array to store values
        """
        return np.absolute(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace))))
    
    def fft(self, img: np.ndarray):
        """ Performs FFT function (image to kspace)

        Performs FFT function, FFT shift and stores the unmodified kspace data
        in a variable and also saves one copy for display and edit purposes.

        Parameters:
            img (np.ndarray): The NumPy ndarray to be transformed
            out (np.ndarray): Array to store output (must be same shape as img)
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    
    def display_images(self, images: List[tuple], figsize=(15, 5)):
        """ Plots the kspace

        Scale the kspace values with log and displays it in a matplotlib plot.

        Parameters:
            images (List[(np.array, str)]): The list of images to display
        """
        
        fig, axs = plt.subplots(1, len(images))
        fig.set_size_inches(figsize)
        for idx, (img, title) in enumerate(images):
            if len(images) > 1:
                axs[idx].imshow(img, cmap='grey')
                axs[idx].set_title(f'{title}')
                axs[idx].axis('off')
            else:
                axs.imshow(img, cmap='grey')
                axs.set_title(f'{title}')
                axs.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_image_path(self, image, coordinates=(100, 100), patch_size=(64, 64)):
        row, col = coordinates
        patch_height, patch_width = patch_size
        return image[row:row + patch_height, col:col + patch_width]
    
    def filling_centric(self, kspace: np.ndarray, value: float):
        """ Centric filling method
        Fills the center line first from left to right and then alternating one
        line above and one below.
        """
        ksp_centric = np.zeros_like(kspace)

        # reorder
        ksp_centric[0::2] = kspace[kspace.shape[0] // 2::]
        ksp_centric[1::2] = kspace[kspace.shape[0] // 2 - 1::-1]

        ksp_centric.flat[int(kspace.size * value / 100)::] = 0

        # original order
        kspace[(kspace.shape[0]) // 2 - 1::-1] = ksp_centric[1::2]
        kspace[(kspace.shape[0]) // 2::] = ksp_centric[0::2]
        return kspace
        
    def apply_low_pass_filter(self, original_kspace: np.array, radius: float = 50.):
        """Low pass filter removes the high spatial frequencies from k-space

        This function only keeps the center of kspace by removing values
        outside a circle of given size. The circle's radius is determined by
        the 'radius' float variable (0.0 - 100) as ratio of the lenght of
        the image diagonally

        Parameters:
            original_kspace (np.array): The kspace sample to filter
            radius (float): Relative size of the kspace mask circle (percent)
        Returns:
            lr_kspace (np.array): The low-pass filtered kspace
            lr_image (np.array): The low-pass filtered image
        """
        lr_kspace = original_kspace.copy()
        if radius < 100:
            r = np.hypot(*lr_kspace.shape) / 2 * radius / 100
            rows, cols = np.array(lr_kspace.shape, dtype=int)
            a, b = np.floor(np.array((rows, cols)) / 2).astype(int)
            y, x = np.ogrid[-a:rows - a, -b:cols - b]
            mask = x * x + y * y <= r * r
            lr_kspace[~mask] = 0
        lr_image = self.ifft(lr_kspace)
        return lr_kspace, lr_image

    
    def add_gaussian_noise(self, kspace, snr_db: float = 20.):
        """
        Add Gaussian noise to the non-zero part of the k-space based on the desired SNR.

        Parameters:
        - kspace (np.array): The radial low-pass filtered k-space.
        - snr_db (float): Desired signal-to-noise ratio in dB.

        Returns:
        - noisy_kspace (np.array): k-space with added Gaussian noise.
        - noisy_image (np.array): image with added Gaussian noise.
        """
        kspace_nonzero = kspace[kspace != 0]
        
        signal_power = np.mean(np.abs(kspace_nonzero) ** 2)
        
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise_real = np.sqrt(noise_power / 2) * np.random.normal(size=kspace_nonzero.shape)
        noise_imag = np.sqrt(noise_power / 2) * np.random.normal(size=kspace_nonzero.shape)
        
        noise = noise_real + 1j * noise_imag
        
        noisy_kspace = kspace.copy()
        
        noisy_kspace[kspace != 0] += noise
        noisy_image = self.ifft(noisy_kspace)
        return noisy_kspace, noisy_image

    def run_full_preprocessing(self, kspace, low_pass_radius: float = 20, snr_db: float = 20.):
        
        original_image = self.ifft(kspace)
        lr_kspace, lr_image = self.apply_low_pass_filter(kspace, low_pass_radius)
        noisy_kspace, noisy_image = self.add_gaussian_noise(lr_kspace, snr_db)
        return {
            'hr_kspace': kspace, 
            'hr_image': original_image, 
            'lr_kspace': lr_kspace,
            'lr_image': lr_image,
            'noisy_lr_kspace': noisy_kspace,
            'noisy_lr_image': noisy_image
        }