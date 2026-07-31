import lightning.pytorch as pl

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.ToTensor()


    def prepare_data(self):
        datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True
        )
        
    def setup(self, stage=None):
        dataset = datasets.MNIST(
            self.data_dir, 
            train=True, 
            transform=self.transform
        )
        
        self.train_data, self.val_data = random_split(
            dataset, 
            [55000, 5000]
        )
        
        self.test_data = datasets.MNIST(
            self.data_dir, 
            train=False, 
            transform=self.transform
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=5
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5
        )