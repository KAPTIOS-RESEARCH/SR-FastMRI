import torchvision
from torch.utils.data import DataLoader, Subset
from . import AbstractDataloader
from torchvision.datasets.cifar import CIFAR10
from src.data.utils.transforms import GaussianNoise


class ReconstructionCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_transform = GaussianNoise()
    
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        noisy_img = self.noise_transform(img.clone())
        return noisy_img, img
    
class ReconstructionCIFAR10Loader(AbstractDataloader):
    def __init__(self, 
                 data_dir: str, 
                 input_size: tuple = (32, 32), 
                 batch_size: int = 8, 
                 num_workers: int = 8,
                 debug: bool = True):
        
        super(ReconstructionCIFAR10Loader, self).__init__()
        self.data_dir = data_dir
        self.debug = debug
        
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers 

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
        
    def train(self):
        train_dataset = ReconstructionCIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transforms
        )
        
        if self.debug:
            train_dataset = Subset(train_dataset, range(100))
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = ReconstructionCIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transforms
        )
        
        if self.debug:
            val_dataset = Subset(val_dataset, range(100))
            
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()