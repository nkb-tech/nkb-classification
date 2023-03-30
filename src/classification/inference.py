from pathlib import Path
from os.path import split
import yaml
import shutil
import sys

import torch

import argparse
from tqdm import tqdm

from src.classification.dataset import get_inference_dataset

import warnings
warnings.filterwarnings("ignore")


def inference(model, loader, 
              save_path, classes, device):
    model.eval()
    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, leave=False):
            imgs = imgs.float().to(device)
            preds = model(imgs)
            preds = preds.argmax(dim=1).to('cpu')
            for pred, img_path in zip(preds, img_paths):
                cls_name = classes[pred.item()]
                img_name = split(img_path)[-1]
                shutil.copy(img_path, Path(save_path, cls_name, img_name))

def read_py_config(path):
    path = Path(path)
    sys.path.append(str(path.parent))
    line = f'import {path.stem} as cfg'
    return line

def main():
    parser = argparse.ArgumentParser(description='Inference arguments')
    parser.add_argument('-cfg', '--config', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())
    data_loader = get_inference_dataset(cfg.inference_data, cfg.inference_pipeline)
    device = torch.device(cfg.device)
    classes = cfg.inference_data['classes']
    model = torch.jit.load(cfg.model['checkpoint']).to(torch.device(device))
    save_path = Path(cfg.save_path)
    for i, name in classes.items():
        save_path.joinpath(name).mkdir(exist_ok=True, parents=True)
    inference(model, data_loader, save_path, classes, device)

if __name__ == '__main__':
    main()

