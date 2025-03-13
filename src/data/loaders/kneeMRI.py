import pathlib
from torch.utils.data import DataLoader, Subset
from . import AbstractDataloader
from src.data.utils.fastMRI import *
from src.data.sets.super_resolution import FastMRISuperResolutionDataset

class KneeMRILoader(AbstractDataloader):
    def __init__(self, 
                 train_data_dir: str, 
                 val_data_dir: str, 
                 input_size: tuple = (28, 28), 
                 batch_size: int = 8, 
                 num_workers: int = 2,
                 debug: bool = True,
                 accelerations: list = [4, 4],
                 center_fractions: list = [0.08, 0.04]):
        
        super(KneeMRILoader, self).__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.debug = debug
        
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers 

        self.challenge = "singlecoil"

        self.mask_func = RandomMaskFunc(
            center_fractions=center_fractions,
            accelerations=accelerations
        )

    def train(self):
        train_dataset = SliceDataset(
            root=pathlib.Path(self.train_data_dir),
            transform=UnetDataTransform(self.challenge, self.mask_func),
            challenge=self.challenge,
            input_size=self.input_size
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
        val_dataset = SliceDataset(
            root=pathlib.Path(self.val_data_dir),
            transform=UnetDataTransform(self.challenge, self.mask_func),
            challenge=self.challenge,
            input_size=self.input_size
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