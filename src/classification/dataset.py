import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Transforms:
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        return self.transforms(image=np.array(img))['image']

class InferDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        super(InferDataset, self,).__init__()
        self.ext = ['.jpg', '.jpeg', '.png']
        self.folder = Path(folder_path)
        self.transform = transform # some infer transform
        self.imgs = [p for p in self.folder.iterdir() if p.suffix.lower() in self.ext]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        path = str(self.imgs[idx])
        if self.transform is not None:
            return self.transform(img), path
        return img, path


def get_dataset(data, pipeline):
    transform = Transforms(pipeline)
    dataset = ImageFolder(data['root'], transform=transform)
    if data.get('weighted_sampling', False):
        loader = DataLoader(dataset, batch_size=data['batch_size'],  
                            sampler=ImbalancedDatasetSampler(dataset),
                            num_workers=data['num_workers'], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=data['batch_size'], 
                            shuffle=data['shuffle'], num_workers=data['num_workers'], pin_memory=True)
    return loader

# def get_inference_dataset(cfg_data):
#     transform = get_transforms(cfg_data)
#     dataset = InferDataset(cfg_data['root'], transform=transform)
#     loader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
#                         num_workers=cfg_data['num_workers'], pin_memory=True)
#     return loader