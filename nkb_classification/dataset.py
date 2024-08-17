import pickle as pkl
from pathlib import Path, PosixPath
from typing import Callable

import albumentations as A
import cv2
import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import requests, zipfile, io
import os


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


        #TODO
        weights = 1.0 / label_to_count[df["label"]]
        # weights = 1.0 / len(label_to_count)

        
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
        train_annotations_file,
        target_names,
        transform=None,
    ):
        super(
            InferDataset,
            self,
        ).__init__()
        self.ext = [".jpg", ".jpeg", ".png"]
        self.folder = Path(folder_path)
        self.train_ann_table = pd.read_csv(
            train_annotations_file, index_col=0
        )  # in order to inherit the set of classes
        # that the model saw during training
        self.train_ann_table = self.train_ann_table[self.train_ann_table["fold"] == "train"]
        self.target_names = [*sorted(target_names)]
        self.classes = {
            target_name: np.sort(np.unique(self.train_ann_table[target_name].values)).tolist()
            for target_name in self.target_names
        }
        self.class_to_idx = {
            target_name: {k: i for i, k in enumerate(classes)} for target_name, classes in self.classes.items()
        }
        self.idx_to_class = {
            target_name: {idx: lb for lb, idx in class_to_idx.items()}
            for target_name, class_to_idx in self.class_to_idx.items()
        }
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
        trnsform: which transform to apply to the image before returning it in the __getitem__ method
        image_base_dir: base directory of images. if = None, then absolute paths are expected in 'path' column.
    """

    def __init__(self, annotations_file, target_column, fold="train", transform=None, image_base_dir=None, **kwargs):
        super().__init__()
        self.table = pd.read_csv(annotations_file, index_col=0)
        self.table = self.table[self.table["fold"] == fold]
        self.target_column = target_column
        self.classes = np.unique(self.table[target_column].values)

        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}

        self.idx_to_class = {idx: lb for lb, idx in self.class_to_idx.items()}

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


class AnnotatedYOLODataset(Dataset):
    """extractall
    AnnotatedYOLODataset provides a way to do single-target classification on YOLO crops.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image pahts and their target values
        target_column: name of the column containing class annotations
        fold: which fold in the dataset to work with (train, val, test, -1)
        transform: which transform to apply to the image before returning it in the __getitem__ method
        image_base_dir: base directory of images. if = None, then absolute paths are expected in 'path' column.
    """


    def __init__(
        self,
        annotations_file,
        target_column=None,
        fold="train",
        transform=None,
        image_base_dir=None,
        min_box_size=5,
        **kwargs
    ):
        super().__init__()

        self.ext = [".jpg", ".jpeg", ".png"]
        self.min_box_size = min_box_size

        assert fold in ('train', 'val', 'test'), \
            f'Got fold equals {fold}'
        
        self.fold = fold
        self.transform = transform

        assert os.path.exists(annotations_file), \
            f'Annotations file {annotations_file} does not exist.'
        with open(annotations_file, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.SafeLoader)

        if not target_column:
            self.idx_to_class = self.yaml_data["names"]
        else:
            self.idx_to_class =  {
                idx: lb
                for idx, lb in enumerate(target_column)
            }

        self.classes = list(self.idx_to_class.values())

        self.class_to_idx = {lb: idx for idx, lb in self.idx_to_class.items()}
        
        if not isinstance(self.yaml_data[self.fold], list):
            self.yaml_data[self.fold] = [self.yaml_data[self.fold]]

        image_base_dir = Path(image_base_dir) if image_base_dir is not None else Path("/")
        image_dir = list(map(lambda p: image_base_dir / self.yaml_data["path"] / p, self.yaml_data[self.fold]))
        
        self.list_bbox = []
        
        for image_dir in image_dir:

            image_dir = Path(image_dir)
            if not image_dir.is_dir():
                #loading marking dataset for yolo
                url = self.yaml_data["download"]
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.yaml_data["path"])
                print(f"Finish loading dataset by {self.yaml_data["download"]}")

            labels_dir = image_dir.parent / "labels"
            assert labels_dir.is_dir(), \
                f"Directory {labels_dir} does not exist"
        
            for image_filename in sorted(image_dir.iterdir()):
                if image_filename.suffix.lower() not in self.ext:
                    continue

                txt_file = labels_dir / (image_filename.stem + ".txt")
                # checking if it is a file
                if not txt_file.is_file():
                    continue

                with open(txt_file, 'r') as fp:
                    lines = fp.readlines()

                with Image.open(image_filename) as img:
                    img_width, img_height = img.size
                    image_size = (img_height, img_width)

                for line in lines:
                    label = int(line.split()[0])
                    x_center, y_center, width, height = tuple(map(float, line.split()[1:]))
                    x_min, y_min, x_max, y_max = self.bbox_xywhn2xyxy(x_center, y_center, width, height, image_size)

                    if not self.check_boxes_sizes_annotation(x_min, y_min, x_max, y_max):
                        continue

                    image_filename = str(image_filename)
                    self.list_bbox.append((image_filename, (x_min, y_min, x_max, y_max), label))

    def __len__(self):
        return len(self.list_bbox)

    def __getitem__(self, idx):
        image_filename, (x_min, y_min, x_max, y_max), label = self.list_bbox[idx]

        img = cv2.imread(image_filename)
        
        img = img[y_min:y_max, x_min:x_max]

        if self.transform is not None:
            return self.transform(img), label

        return img, label

    def get_labels(self):
        return np.array([
            label for _, _, label in self.list_bbox
        ])
    
    @staticmethod
    def bbox_xywhn2xyxy(x_center, y_center, width, height, image_size):
        image_height, image_width = image_size
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        return x_min, y_min, x_max, y_max
    
    def check_boxes_sizes_annotation(self, x_min, y_min, x_max, y_max):
        return x_max - x_min >= self.min_box_size and y_max - y_min >= self.min_box_size


class AnnotatedMultitaskDataset(Dataset):
    """
    AnnotatedMultitargetDataset provides a way to do multi-target classification.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image paths and their target values
        target_names: list of target_names to consider
        fold: which fold in the dataset to work with (train, val, test, -1)
        trnsform: which transform to apply to the image before returning it in the __getitem__ method
    """

    def __init__(self, annotations_file, target_names, fold="test", transform=None, image_base_dir=None, **kwargs):
        self.table = pd.read_csv(annotations_file, index_col=0)
        self.table = self.table[self.table["fold"] == fold]
        self.target_names = [*sorted(target_names)]
        self.classes = {
            target_name: np.sort(np.unique(self.table[target_name].values)).tolist()
            for target_name in self.target_names
        }
        self.class_to_idx = {
            target_name: {k: i for i, k in enumerate(classes)} for target_name, classes in self.classes.items()
        }
        self.idx_to_class = {
            target_name: {idx: lb for lb, idx in class_to_idx.items()}
            for target_name, class_to_idx in self.class_to_idx.items()
        }
        self.transform = transform

        if image_base_dir is not None:
            image_base_dir = Path(image_base_dir)
            self.table.path = self.table.path.apply(lambda r: str(image_base_dir / Path(r)))

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img_path = self.table.iloc[idx, self.table.columns.get_loc("path")]
        img = cv2.imread(img_path)
        labels = {
            target_name: np.array(
                self.class_to_idx[target_name][self.table.iloc[idx, self.table.columns.get_loc(target_name)]],
                dtype=np.int64,
            )
            for target_name in self.target_names
        }

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
        case "AnnotatedYOLODataset":
            dataset = AnnotatedYOLODataset(
                # data["ann_file"],
                # data["target_column"],
                # fold = data["fold"],
                transform=transform,
                **data,
            )
            # img,_ = dataset[0]
            # if dataset.fold == "val": cv2.imwrite("nkb-classification/test_crop.jpg", img)

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

    if data.get("weighted_sampling", False):
        # TODO test this
        # get_labels
        loader = DataLoader(
            dataset,
            batch_size=data["batch_size"],
            sampler=ImbalancedDatasetSampler(dataset),
            num_workers=data["num_workers"],
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=data["shuffle"],
            num_workers=data["num_workers"],
            pin_memory=True,
        )
    return loader


def get_inference_dataset(data, pipeline):
    transform = Transforms(pipeline)
    dataset = InferDataset(
        data["root"],
        data["train_annotations_file"],
        data["target_names"],
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        pin_memory=True,
    )
    return loader
