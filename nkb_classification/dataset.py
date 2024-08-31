import pickle as pkl
from pathlib import Path, PosixPath
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from nkb_classification.utils import load_classes, get_classes_configs


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


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
        return self.transforms(image=np.array(img))["image"]


class InferDataset(Dataset):
    def __init__(
        self,
        folder_path,
        transform=None,
    ):
        super(
            InferDataset,
            self,
        ).__init__()
        self.ext = [".jpg", ".jpeg", ".png"]
        self.folder = Path(folder_path)

        self.transform = transform  # some infer transform
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
    def __init__(self, root, ann_file, dict_path, transform=None, **kwargs):
        super().__init__()
        self.data_prefix = root
        self.ann_file = Path(root, ann_file)
        self.dict_path = Path(dict_path)
        self.data_infos = self.load_annotations()
        self.transform = transform

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        img = Image.open(self.data_infos[idx]["img_info"]["filename"])
        label = self.data_infos[idx]["gt_label"]
        if self.transform is not None:
            return self.transform(img), label
        return img, label

    def load_annotations(self):
        assert isinstance(self.ann_file, PosixPath)
        with self.ann_file.open("rb") as f:
            data = pkl.load(f)
        with self.dict_path.open("rb") as f:
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
            img_path = Path(self.data_prefix, "images_lr", orig_label, filename)
            assert img_path.is_file(), f"File {img_path} does not exist."
            info = {
                "img_prefix": self.data_prefix,
                "img_info": {"filename": str(img_path)},
                "gt_label": np.array(self.class_to_idx[label], dtype=np.int64),
            }
            data_infos.append(info)

        return data_infos


class AnnotatedSingletaskDataset(Dataset):
    """
    AnnotatedSingletargetDataset provides a way to do single-target classification.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image paths and their target values
        target_column: name of the column containing class annotations
        fold: which fold in the dataset to work with (train, val, test, -1)
        transform: which transform to apply to the image before returning it in the __getitem__ method
        image_base_dir: base directory of images. if = None, then absolute paths are expected in 'path' column.
        classes: optional config of classes (list, dict, or path to json). if None, then will be infered from annotations
    """

    def __init__(self, annotations_file, target_column, fold="test", transform=None, image_base_dir=None, classes=None, **kwargs):
        self.table = pd.read_csv(annotations_file)
        self.table = self.table[self.table["fold"] == fold]
        self.target_column = target_column

        if classes is not None:
            self.classes = load_classes(classes)
        else:
            self.classes = np.sort(np.unique(self.table[target_column].values)).tolist()

        self.class_to_idx, self.idx_to_class = get_classes_configs(self.classes)

        self.transform = transform

        if image_base_dir is not None:
            image_base_dir = Path(image_base_dir)
            self.table.path = self.table.path.apply(lambda r: str(image_base_dir / Path(r)))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_path = self.table.iloc[idx, self.table.columns.get_loc("path")]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = np.array(
            self.class_to_idx[self.table.iloc[idx, self.table.columns.get_loc(self.target_column)]],
            dtype=np.int64,
        )

        if self.transform is not None:
            return self.transform(img), labels
        return img, labels

    def get_labels(self):
        return self.table[self.target_column].values


class AnnotatedMultitaskDataset(Dataset):
    """
    AnnotatedMultitargetDataset provides a way to do multi-target classification.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image paths and their target values
        target_names: list of target_names to consider
        fold: which fold in the dataset to work with (train, val, test, -1)
        transform: which transform to apply to the image before returning it in the __getitem__ method
        classes: optional config of classes (list, dict, or path to json). if None, then will be infered from annotations
    """

    def __init__(self, annotations_file, target_names, fold="test", transform=None, image_base_dir=None, classes=None, **kwargs):
        self.table = pd.read_csv(annotations_file)
        self.table = self.table[self.table["fold"] == fold]
        self.target_names = [*sorted(target_names)]

        if classes is not None:
            self.classes = load_classes(classes)
        else:
            self.classes = {
                target_name: np.sort(np.unique(self.table[target_name].values)).tolist()
                for target_name in self.target_names
            }

        self.class_to_idx, self.idx_to_class = get_classes_configs(self.classes)

        self.transform = transform

        if image_base_dir is not None:
            image_base_dir = Path(image_base_dir)
            self.table.path = self.table.path.apply(lambda r: str(image_base_dir / Path(r)))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_path = self.table.iloc[idx, self.table.columns.get_loc("path")]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = {
            target_name: np.array(
                self.class_to_idx[target_name][self.table.iloc[idx, self.table.columns.get_loc(target_name)]],
                dtype=np.int64,
            )
            for target_name in self.target_names
        }

        # import ipdb; ipdb.set_trace()

        if self.transform is not None:
            return self.transform(img), labels
        return img, labels

    def get_labels(self):
        return self.table[self.target_names].values


def get_dataset(data, pipeline):
    transform = Transforms(pipeline)
    match data["type"]:
        case "GroupsDataset":
            dataset = GroupsDataset(
                # data["root"],
                # data["ann_file"],
                # data["group_dict"],
                transform=transform,
                **data,
            )
        case "AnnotatedMultitaskDataset":
            dataset = AnnotatedMultitaskDataset(
                # data["ann_file"],
                # data["target_names"],
                # data["fold"],
                transform=transform,
                **data,
            )
        case "AnnotatedSingletaskDataset":
            dataset = AnnotatedSingletaskDataset(
                # data["ann_file"],
                # data["target_column"],
                # data["fold"],
                transform=transform,
                **data,
            )

        case _:
            dataset = ImageFolder(data["root"], transform=transform)

    # TODO match case
    # kwargs to datasets
    # if data["type"] == "GroupsDataset":
    #     dataset = GroupsDataset(
    #         data["root"],
    #         data["ann_file"],
    #         data["group_dict"],
    #         transform=transform,
    #     )
    # elif data["type"] == "AnnotatedMultitaskDataset":
    #     dataset = AnnotatedMultitaskDataset(
    #         data["ann_file"],
    #         data["target_names"],
    #         data["fold"],
    #         transform=transform,
    #     )
    # elif data["type"] == "AnnotatedSingletaskDataset":
    #     dataset = AnnotatedSingletaskDataset(
    #         data["ann_file"],
    #         data["target_column"],
    #         data["fold"],
    #         transform=transform,
    #     )
    # else:
    #     dataset = ImageFolder(data["root"], transform=transform)

    drop_last = data.get("drop_last", False)
    if data.get("weighted_sampling", False):
        # TODO test this
        # get_labels
        loader = DataLoader(
            dataset,
            batch_size=data["batch_size"],
            sampler=ImbalancedDatasetSampler(dataset),
            num_workers=data["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=data["shuffle"],
            num_workers=data["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
        )
    return loader


def get_inference_dataset(data, pipeline):
    transform = Transforms(pipeline)
    dataset = InferDataset(
        folder_path=data["folder_path"],
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        pin_memory=True,
    )
    return loader
