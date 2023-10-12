import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import cv2
import os

class xBDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_path): 
        self.batch_size = batch_size
        self.train_path = train_path

    def train_dataloader(self):
        dataloader = DataLoader(xBDDataset(dataset_path = self.train_path), batch_size=self.batch_size)
        return dataloader
    
    def val_dataloader(self): # FIXME: replace with correct dataloader
        dataloader = DataLoader(xBDDataset(dataset_path = self.train_path), batch_size=self.batch_size)
        return dataloader
    
    def test_dataloader(self): # FIXME: replace with correct dataloader
        dataloader = DataLoader(xBDDataset(dataset_path = self.train_path), batch_size=self.batch_size)
        return dataloader

    
class xBDDataset(Dataset):
    def __init__(self, dataset_path):
        self.predisaster_paths = []
        images_path = os.path.join(dataset_path, 'images/')
        for name in os.listdir(images_path):
            if ('_pre_disaster.png' in name):
                self.predisaster_paths.append(os.path.join(images_path, name))

    def __len__(self):
        return len(self.predisaster_paths)
    
    def __getitem__(self, index):
        fn = self.predisaster_paths[index]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_B = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)
        label = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        img = TF.to_tensor(img)
        img_B = TF.to_tensor(img_B)
        label = TF.to_tensor(label)
        return {'name': fn, 'A': img, 'B': img_B, 'L': label}