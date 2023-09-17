import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
from PIL import Image
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder

import albumentations as A

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


class GroupsDataset(Dataset):
    def __init__(self,
                 root, 
                 ann_file,
                 dict_path,
                 transform=None):
        super().__init__()
        self.data_prefix = root
        self.ann_file = Path(root, ann_file)
        self.dict_path = Path(dict_path)
        self.data_infos = self.load_annotations()
        self.transform = transform
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        img = Image.open(self.data_infos[idx]['img_info']['filename'])
        label = self.data_infos[idx]['gt_label']
        if self.transform is not None:
            return self.transform(img), label
        return img, label
    
    def load_annotations(self):
        assert isinstance(self.ann_file, PosixPath)
        with self.ann_file.open('rb') as f:
            data = pkl.load(f)
        with self.dict_path.open('rb') as f:
            group_dict = pkl.load(f)
        self.inv_group = {}
        for k, v in group_dict.items():
            for v_i in v:
                self.inv_group[v_i] = k
        self.class_to_idx = {k: i for i, k in enumerate(group_dict.keys())}
        self.idx_to_class = {idx: lb for lb, idx in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        data_infos = []
        for sample in data:
            sample = Path(sample)
            filename = sample.name
            orig_label = sample.parent.name
            label = self.inv_group[orig_label]
            img_path = Path(self.data_prefix, 'images_lr', orig_label, filename)
            assert img_path.is_file(), f'File {img_path} does not exist.'
            info = {
                'img_prefix': self.data_prefix,
                'img_info': {'filename': str(img_path)},
                'gt_label': np.array(self.class_to_idx[label], dtype=np.int64),
                }
            data_infos.append(info)
            
        return data_infos
    

class AnnotatedMultilabelDataset(Dataset):
    def __init__(self, annotations_file, target_name, fold='test', classes=None, transform=None):
        self.table = pd.read_csv(annotations_file, index_col=0)
        self.table = self.table[self.table['fold'] == fold]
        self.target_name = target_name
        if classes is not None:
            self.classes = classes
        else:
            self.classes = [*set(self.table[target_name].values)]
        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}
        self.idx_to_class = {idx: lb for lb, idx in self.class_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        path_column_num = self.table.columns.get_loc('path')
        img_path = self.table.iloc[idx, path_column_num]
        img = Image.open(img_path)
        target_column_num = self.table.columns.get_loc(self.target_name)
        label = self.table.iloc[idx, target_column_num]
        label = np.array(self.class_to_idx[label], dtype=np.int64)
        if self.transform is not None:
            return self.transform(img), label
        return img, label
    
    def get_labels(self):
        return self.table[self.target_name].values