from pathlib import Path
from typing import (
    Tuple,
    Union,
)
import h5py
import torch
from src.data.utils.fastMRI.transforms import SuperResolutionTransform
from onnxruntime.quantization import CalibrationDataReader

class FastMRISuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        challenge: str = 'singlecoil',
        transform = None,
        input_size: Tuple[int, int] = (128, 128)
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('Challenge should be either "singlecoil" or "multicoil"')

        self.root = Path(root)
        self.input_size = input_size
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

    def __getitem__(self, idx):
        fname, slice_idx = self.samples[idx]
        with h5py.File(fname, "r") as hf:
            image = hf[self.recons_key][slice_idx]
        image, target = self.transform(image, self.input_size)
        return image, target
    

model_path = "./exports/mnist-infer.onnx"
quantized_model_path = "./exports/mnist-quantized.onnx"

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
        for i, (lr, hr) in enumerate(data_loader):
            if i >= self.num_samples:
                break
            batch_data.append(lr.numpy())

        self.enum_data_dicts = iter([{"input": img} for img in batch_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)