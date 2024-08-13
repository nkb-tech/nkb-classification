 # NKB Classification

Easy framework for computer vision classification tasks.

## Install ...

Download from git
```bash
git clone git@github.com:nkb-tech/nkb-classification.git nkb_classification
cd nkb_classification
```

### ... via virtualenv

Install the package
```bash
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .
```

### ... via docker

```bash
docker build --tag nkb-cls-im --file Dockerfile .
docker run \
    -itd \
    --ipc host \
    --gpus all \
    --name nkb-cls-cont \
    --volume <host>:<docker> \ # put here path to models/configs to bind with docker image
    nkb-cls-im
docker exec -it nkb-cls-cont /bin/bash
```

## Run train

```bash
cd nkb_classification
python3 -m train -cfg `cfg_path`
```

#### Train config guideline
All configuration needed for training are passed by train config.py file.
See sample files in configs/ directory.

_Important fields_:
* task : Determines task. Should be "single" for single task classfication (binary or multiclass), or 'multi' for multi task classification.
* target_column *only for single task* : Column with class labels.
* target_names *only for multi task* : Column names with class labels.
* experiment : Settings regarding saving model weights, training results and Comet ML logger. Comet logger can be set to None to get disabled.
* train_data / val_data: Train / val dataset settings. Look at comments in sample configs for details.
* train_pipeline : Image preprocessing/augmentations during training phase.
* val_pipeline : Image preprocessing/augmentations during validations phase.
* model : Model settings. To choose a model, look section Available backbone models.

## Run inference

```bash
cd nkb_classification
python3 -m inference -cfg `inference_cfg_path`
```

## Dataset format

#### CSV annotations
To use this format, choose AnnotatedSingletaskDataset or AnnotatedMultitaskDataset dataset type in train_data/val_data configs. The dataset is provided as a `csv` table with paths to the images and annotations for each image which include train/val/test partition. Specifically, it should have the following structure

|| column_0 | column_1 | column_2 | ... | column_n | path | fold |
|-|---|---|---|---|---|---|---|
|0|value_0_0|value_1_0|value_2_0|...|value_n_0|/home/user/data/img_0.jpg|train|
|1|value_0_1|value_1_1|value_2_1|...|value_n_1|/home/user/data/img_1.jpg|val|
|2|value_0_2|value_1_2|value_2_2|...|value_n_2|/home/user/data/img_2.jpg|test|
|3|value_0_3|value_1_3|value_2_3|...|value_n_3|/home/user/data/img_3.jpg|-1|

Objects with the `-1` fold value are ignored. Target columns are scpicfied in the `config` file by the `target_names` list. The path to the `csv` table is also provided through the `config` file.

#### Arange images by folders
To use this format, choose ImageFolder dataset type in train_data/val_data configs. In this case the structure of the dataset should be the following:
```
dataset_root/
├── class_0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
└── class_2/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg

```

## Available backbone models

1. All models from timm library. To get model names run timm.list_models() with optional argument pretrained=True.
Provide exactly same model name to config.py
2. All models from unicom library. List them with unicom.available_models().
To use them, provide "unicom *model_name*" (i.e. "unicom ViT-B/32") in config.py.

## Run onnx export

To enable export models to onnx or torchscript, run first:
```bash
python3 -m pip install -r requirements/optional.txt
```

After that, run:
```bash
cd nkb_classification
python3 -m export \
    --to onnx \  # supported [torchscript, onnx, engine]
    --config `config_model_path` \
    --opset 17 \
    --dynamic batch \  # supported [none, batch, all]
    --sim \  # simplify the graph or not
    --input-shape 1 3 224 224 \
    --device cuda:0 \
    --half \  # convert to fp16
    --save_path . \  # where to save the model
    --weights `weights_model_path`  # path to model weights
```

## Develop
To enable autocheck code before commit, run:
```bash
# install tensorrt if needed to export
python3 -m pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
python3 -m pip install -r requirements/optional.txt
pre-commit install
```
