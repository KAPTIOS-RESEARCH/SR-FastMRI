import pathlib
from torch.utils.data import DataLoader, Subset
from . import AbstractDataloader
from src.data.utils.fastMRI import *
from src.data.sets.super_resolution import FastMRISuperResolutionDataset

class SRKneeMRILoader(AbstractDataloader):
    def __init__(self, 
                 train_data_dir: str, 
                 val_data_dir: str, 
                 lr_image_scale: int = 2, 
                 batch_size: int = 4, 
                 num_workers: int = 4,
                 debug: bool = True):
        
        super(SRKneeMRILoader, self).__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.debug = debug
        
        self.lr_image_scale = lr_image_scale
        self.batch_size = batch_size
        self.num_workers = num_workers 

        self.challenge = "singlecoil"

    def train(self):
        train_dataset = FastMRISuperResolutionDataset(
            root=pathlib.Path(self.train_data_dir),
            challenge=self.challenge,
            lr_image_scale=self.lr_image_scale
        )
        
        if self.debug:
            train_dataset = Subset(train_dataset, range(100))
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = FastMRISuperResolutionDataset(
            root=pathlib.Path(self.val_data_dir),
            challenge=self.challenge,
            lr_image_scale=self.lr_image_scale
        )
        
        if self.debug:
            val_dataset = Subset(val_dataset, range(100))
            
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()