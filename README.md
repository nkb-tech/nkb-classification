# NKB Classification

Easy framework for computer vision classification tasks.

## Install ...

Download from git
```bash
git clone git@github.com:nkb-tech/nkb-classification.git
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

## Run train

```bash
cd nkb_classification
python3 -m train -cfg `cfg_path`
```

## Run onnx export

To enable export models to onnx or torchscript, run first:
```bash
python3 -m pip install -r requirements/dev.txt
```

After that, run:
```bash
cd nkb_classification
python3 -m export \
    --to onnx \  # supported [torchscript, onnx]
    --config `config_model_path` \
    --opset 17 \
    --dynamic batch \  # supported [none, batch, all]
    --sim \  # simplify the graph or not
    --input-shape 1 3 224 224 \
    --device cuda:0 \
    --save_path . \  # where to save the model
    --weights `weights_model_path`  # path to model weights
```

## Develop
To enable autocheck code before commit, run:
```bash
python3 -m pip install -r requirements/dev.txt
pre-commit install
```