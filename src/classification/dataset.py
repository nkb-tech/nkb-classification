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

def get_transforms(cfg_data):
    size = cfg_data['size']
    transform = A.Compose([
        # A.LongestMaxSize(size),
        # A.OneOrOther(first=A.Resize(size, size),
        #              second=A.RandomResizedCrop(size, size, scale=(0.5, 1.0))),
        A.Resize(size, size),
        # A.SafeRotate(limit=20, p=0.4),
        # A.Affine(scale=(0.5, 1.5), translate_percent=0.3, p=0.8),
        # A.Blur(blur_limit=7, p=0.3),
        # A.RandomSnow(snow_point_lower=0.0, 
        #              snow_point_upper=0.5, 
        #              brightness_coeff=1.5,
        #              p=0.3), 
        A.MotionBlur(blur_limit=7, 
                     allow_shifted=True,
                     p=0.3),
        A.RandomShadow(shadow_roi=(0, 0.0, 1, 1), 
                       num_shadows_lower=0, 
                       num_shadows_upper=2,
                       shadow_dimension=20,
                       p=0.3),
        A.RandomToneCurve(scale=0.8, p=0.5),
        # A.Solarize(threshold=128, p=0.3),
        # A.HueSaturationValue(),
        A.ImageCompression(quality_lower=80, quality_upper=100),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        # A.ChannelDropout(p=0.2),
        # A.PadIfNeeded(
        #     position=A.PadIfNeeded.PositionType.CENTER,
        #     min_height=size,
        #     min_width=size,
        #     value=0,
        #     border_mode=cv2.BORDER_CONSTANT,
        # ),
        ToTensorV2()
    ])
    return Transforms(transform)


def get_dataset(cfg_data):
    transform = get_transforms(cfg_data)
    dataset = ImageFolder(cfg_data['root'], transform=transform)
    if cfg_data.get('weighted_sampling', False):
        loader = DataLoader(dataset, batch_size=cfg_data['batch_size'],  
                            sampler=ImbalancedDatasetSampler(dataset),
                            num_workers=cfg_data['num_workers'], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
                            shuffle=cfg_data['shuffle'], num_workers=cfg_data['num_workers'], pin_memory=True)
    return loader

def get_inference_dataset(cfg_data):
    transform = get_transforms(cfg_data)
    dataset = InferDataset(cfg_data['root'], transform=transform)
    loader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
                        num_workers=cfg_data['num_workers'], pin_memory=True)
    return loader