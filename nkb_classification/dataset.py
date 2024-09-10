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
import glob

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


class AnnotatedYOLODataset(Dataset):
    """extractall
    AnnotatedYOLODataset provides a way to do single-target classification on YOLO crops.

    Args:
        annotations_file: path to the annotation file, which contains a pandas dataframe
                          with image pahts and their target values
        fold: which fold in the dataset to work with (train, val, test, -1)
        transform: which transform to apply to the image before returning it in the __getitem__ method
        image_base_dir: base directory of images. if = None, then absolute paths are expected in 'path' column.
        min_box_size: bboxes with smaller linear size are ignored
        generate_backgrounds: option to generate random crops from image backgrounds
                          in oreder to train the classifier to distinguish false and true detections
        background_generating_prob: probability of taking a bakground crop from an image (in case generate_backgrounds == True).
                          If None, would be 1 / n_classes
        background_crop_sizes: min and max size of randomly chosen background crop (in case generate_backgrounds == True).
    """


    def __init__(
        self,
        annotations_file,
        fold="train",
        transform=None,
        image_base_dir=None,
        min_box_size=5,
        generate_backgrounds=False,
        background_generating_prob=None,
        background_crop_sizes=(0.1, 0.3),
        **kwargs
    ):
        super().__init__()

        self.ext = [".jpg", ".jpeg", ".png"]
        self.min_box_size = min_box_size

        assert fold in ('train', 'val', 'test'), \
            f'Got fold equals {fold}'
        
        self.fold = fold
        self.transform = transform

        self.generate_backgrounds = generate_backgrounds
        self.background_generating_prob = background_generating_prob
        self.background_crop_sizes = background_crop_sizes
        self.attempts_to_put_bakground_crop = 10  # for random background_crop not to have intersection with true bboxes

        assert os.path.exists(annotations_file), \
            f'Annotations file {annotations_file} does not exist.'
        with open(annotations_file, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.SafeLoader)

        self.idx_to_class = self.yaml_data["names"]
        assert set(self.idx_to_class.keys()) == set(range(len(self.idx_to_class))), \
        "Class indices should form range(0, num_classes) without skips"

        self.classes = list(None for _ in range(len(self.idx_to_class)))
        for idx, lb in self.idx_to_class.items():
            self.classes[idx] = lb

        self.class_to_idx = {lb: idx for idx, lb in self.idx_to_class.items()}

        if generate_backgrounds:
            bg_idx, bg_lb = len(self.classes), "<GENERATED>_background"
            self.classes.append(bg_lb)
            self.idx_to_class[bg_idx] = bg_lb
            self.class_to_idx[bg_lb] = bg_idx

        if self.background_generating_prob is None:
            self.background_generating_prob = 1 / len(self.classes)
        
        if not isinstance(self.yaml_data[self.fold], list):
            self.yaml_data[self.fold] = [self.yaml_data[self.fold]]

        image_base_dir = Path(image_base_dir) if image_base_dir is not None else Path("/")
        image_dirs = list(map(lambda p: image_base_dir / self.yaml_data["path"] / p, self.yaml_data[self.fold]))

        if len(image_dirs) == 1 and "download" in self.yaml_data and not image_dirs[0].is_dir():
            url = self.yaml_data["download"]
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.yaml_data["path"])
            print(f"Finish loading dataset by {self.yaml_data["download"]}")

        print("Scanning image directories")
        img_paths = self.get_img_files(image_dirs)
        
        self.list_bbox = []

        print("Scanning image annotations")
        for image_filename in sorted(img_paths):

            image_filename = Path(image_filename)
            labels_dir = image_filename.parent.parent / "labels"
            assert labels_dir.is_dir(), \
                f"Directory {labels_dir} does not exist"

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

            if self.generate_backgrounds:
                if np.random.rand() > self.background_generating_prob:
                    continue

                for attempt in range(self.attempts_to_put_bakground_crop):
                
                    bg_crop_size = np.random.uniform(*self.background_crop_sizes)

                    bg_x_min = np.random.randint(0, int(img_width * (1 - bg_crop_size)))
                    bg_y_min = np.random.randint(0, int(img_height * (1 - bg_crop_size)))
                    bg_x_max = bg_x_min + int(img_width * bg_crop_size)
                    bg_y_max = bg_y_min + int(img_height * bg_crop_size)

                    if not self.check_boxes_sizes_annotation(
                        bg_x_min, bg_y_min, bg_x_max, bg_y_max
                    ):
                        continue

                    bg_label = self.class_to_idx[self.classes[-1]]

                    successfully_put_bg_crop = True
                    for line in lines:  # check inetrsection with true object boxes
                        x_center, y_center, width, height = tuple(map(float, line.split()[1:]))
                        x_min, y_min, x_max, y_max = self.bbox_xywhn2xyxy(
                            x_center, y_center, width, height, image_size
                        )

                        if not self.bbox_intersect(
                            (bg_x_min, bg_y_min, bg_x_max, bg_y_max),
                            (x_min, y_min, x_max, y_max)
                        ):
                            successfully_put_bg_crop = False

                    if not successfully_put_bg_crop:
                        continue

                    self.list_bbox.append(
                        (image_filename, (bg_x_min, bg_y_min, bg_x_max, bg_y_max), bg_label)
                    )
                    break

    def __len__(self):
        return len(self.list_bbox)

    def __getitem__(self, idx):
        image_filename, (x_min, y_min, x_max, y_max), label = self.list_bbox[idx]

        img = cv2.imread(image_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    @staticmethod
    def bbox_intersect(bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        if x1_max < x2_min or x2_max < x1_min:
            return False
        if y1_max < y2_min or y2_max < y1_min:
            return False
        return True
    
    def check_boxes_sizes_annotation(self, x_min, y_min, x_max, y_max):
        return x_max - x_min >= self.min_box_size and y_max - y_min >= self.min_box_size
    
    def get_img_files(self, img_path):
        """
        Read yolo images dattaset directory.
        
        As implemented in https://github.com/ultralytics/ultralytics/blob/7a79680dcc1d9c8d8da1c3910fa1775110c41255/ultralytics/data/base.py#L99
        """
        HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
        IMG_FORMATS = tuple(map(lambda ext: ext.split('.')[-1], self.ext))  # image suffixes
        VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
        FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n{HELP_URL}") from e
        return im_files


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
