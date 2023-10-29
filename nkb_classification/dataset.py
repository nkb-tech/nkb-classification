import cv2
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
from PIL import Image
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

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
    def __init__(self, folder_path, train_annotations_file, target_names, transform=None):
        super(InferDataset, self,).__init__()
        self.ext = ['.jpg', '.jpeg', '.png']
        self.folder = Path(folder_path)
        self.train_ann_table = pd.read_csv(train_annotations_file, index_col=0) # in order to inherit the set of classes
                                                                                # that the model saw during training
        self.train_ann_table = self.train_ann_table[self.train_ann_table['fold'] == 'train']
        self.target_names = [*sorted(target_names)]
        self.classes = {target_name: np.sort(np.unique(self.train_ann_table[target_name].values)) 
                        for target_name in self.target_names}
        self.class_to_idx = {target_name: {k: i for i, k in enumerate(classes)} 
                             for target_name, classes in self.classes.items()}
        self.idx_to_class = {target_name: {idx: lb for lb, idx in class_to_idx.items()} 
                             for target_name, class_to_idx in self.class_to_idx.items()}
        self.transform = transform # some infer transform
        self.imgs = [str(p) for p in self.folder.iterdir() if p.suffix.lower() in self.ext]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    

class AnnotatedMultitargetDataset(Dataset):
    """
    AnnotatedMultitargetDataset provides a way to do multi-target classification.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image pahts and their target values
        target_names: list of target_names to consider
        fold: which fold in the dataset to work with (train, val, test, -1)
        trnsform: which transform to apply to the image before returning it in the __getitem__ method
    """
    def __init__(self, annotations_file, target_names, fold='test', transform=None):
        self.table = pd.read_csv(annotations_file, index_col=0)
        self.table = self.table[self.table['fold'] == fold]
        self.target_names = [*sorted(target_names)]
        self.classes = {target_name: np.sort(np.unique(self.table[target_name].values)) 
                        for target_name in self.target_names}
        self.class_to_idx = {target_name: {k: i for i, k in enumerate(classes)} 
                             for target_name, classes in self.classes.items()}
        self.idx_to_class = {target_name: {idx: lb for lb, idx in class_to_idx.items()} 
                             for target_name, class_to_idx in self.class_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        img_path = self.table.iloc[idx, self.table.columns.get_loc('path')]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = {
            target_name: np.array(self.class_to_idx[target_name][self.table.iloc[idx, self.table.columns.get_loc(target_name)]], dtype=np.int64)
            for target_name in self.target_names
        }
        if self.transform is not None:
            return self.transform(img), labels
        return img, labels
    
    def get_labels(self):
        return self.table[self.target_names].values
    

def get_dataset(data, pipeline):
    transform = Transforms(pipeline)
    if data['type'] == 'GroupsDataset':
        dataset = GroupsDataset(data['root'],
                                data['ann_file'], 
                                data['group_dict'],
                                transform=transform)
    elif data['type'] == 'AnnotatedMultitargetDataset':
        dataset = AnnotatedMultitargetDataset(data['ann_file'],
                                             data['target_names'],
                                             data['fold'],
                                             transform=transform)
    else:
        dataset = ImageFolder(data['root'], transform=transform)
    if data.get('weighted_sampling', False):
        loader = DataLoader(dataset, batch_size=data['batch_size'],  
                            sampler=ImbalancedDatasetSampler(dataset),
                            num_workers=data['num_workers'], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=data['batch_size'], 
                            shuffle=data['shuffle'], num_workers=data['num_workers'], pin_memory=True)
    return loader


def get_inference_dataset(data, pipeline):
    transform = Transforms(pipeline)
    dataset = InferDataset(data['root'], 
                           data['train_annotations_file'],
                           data['target_names'],
                           transform=transform)
    loader = DataLoader(dataset, batch_size=data['batch_size'], 
                        num_workers=data['num_workers'], pin_memory=True)
    return loader
