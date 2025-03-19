from pathlib import Path
from typing import (
    Union,
)
import h5py
import torch
from src.data.utils.fastMRI.transforms import SuperResolutionTransform
from onnxruntime.quantization import CalibrationDataReader
from src.utils.image import extract_patches
from src.data.utils.fastMRI.transforms import SRSample

class FastMRISuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        challenge: str = 'singlecoil',
        transform = None,
        lr_image_scale: int = 2
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('Challenge should be either "singlecoil" or "multicoil"')

        self.root = Path(root)
        self.lr_image_scale = lr_image_scale
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.files = list(self.root.glob("*.h5"))
        self.samples = self._load_samples()
        
        if not transform:
            self.transform = SuperResolutionTransform()
        else:
            self.transform = transform

    def _load_samples(self):
        samples = []
        for fname in self.files:
            with h5py.File(fname, "r") as hf:
                num_slices = hf[self.recons_key].shape[0]
                for i in range(num_slices):
                    samples.append((fname, i))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> SRSample:
        fname, slice_idx = self.samples[idx]
        with h5py.File(fname, "r") as hf:
            image = hf[self.recons_key][slice_idx]
        sample = self.transform(image, self.lr_image_scale)
        return sample
    
class FastMRISuperResolutionDataReader(CalibrationDataReader):
    def __init__(self, data_folder, num_samples=100):
        self.num_samples = num_samples
        self.data_folder = data_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.load_data()

    def load_data(self):
        dataset = FastMRISuperResolutionDataset(self.data_folder)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        batch_data = []
        for i, sample in enumerate(data_loader):
            if i >= self.num_samples:
                break
            batch_data.append(sample)

        self.enum_data_dicts = iter([{"input": sample.image.numpy()} for sample in batch_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)
    

class FastMRIPatchSuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], challenge: str = 'singlecoil', transform=None, lr_image_scale: int = 2, patch_size=64, overlap=0.5):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('Challenge should be either "singlecoil" or "multicoil"')

        self.root = Path(root)
        self.lr_image_scale = lr_image_scale
        self.patch_size = patch_size
        self.overlap = overlap
        self.recons_key = "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        self.files = list(self.root.glob("*.h5"))
        self.samples = self._load_samples()
        self.transform = transform if transform else SuperResolutionTransform()

    def _load_samples(self):
        samples = []
        for fname in self.files:
            with h5py.File(fname, "r") as hf:
                num_slices = hf[self.recons_key].shape[0]
                for i in range(num_slices):
                    image = hf[self.recons_key][i]
                    patches, _ = extract_patches(image, patch_size=self.patch_size, overlap=self.overlap)
                    for patch in patches:
                        samples.append(patch)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch = self.samples[idx]
        sample = self.transform(patch, self.lr_image_scale)
        return sample