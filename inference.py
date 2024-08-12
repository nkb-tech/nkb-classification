import argparse
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nkb_classification.dataset import get_inference_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import load_classes, get_classes_configs, read_py_config


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    classes: Union[list, dict],
    save_path: str,
    device: Union[torch.device, str],
    cfg: Any
) -> None:

    _, idx_to_class = get_classes_configs(classes)

    task = cfg.task
    assert task in ("single", "multi")
    if task == "single":
        target_column = cfg.target_column
        columns = [target_column]
    elif task == "multi":
        target_names = cfg.target_names
        assert set(target_names) == set(classes.keys())
        columns = target_names.copy()

    columns.append("path")
    inference_annotations = pd.DataFrame(columns=columns)

    model.eval()

    for imgs, img_paths in tqdm(loader, leave=False, desc="Inference"):
        imgs = imgs.float().to(device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=cfg.enable_mixed_presicion,
        ):
            preds = model(imgs)
        batch_annotations = []
        if task == "single":
            pred = preds
            pred = pred.argmax(dim=-1).cpu().numpy().tolist()
            pred = [idx_to_class[idx] for idx in pred]
            batch_annotations.append(pred)
        elif task == "multi":
            for target_name in target_names:
                pred = preds[target_name]
                pred = pred.argmax(dim=-1).cpu().numpy().tolist()
                pred = [idx_to_class[target_name][idx] for idx in pred]
                batch_annotations.append(pred)
        batch_annotations.append(list(img_paths))
        batch_annotations = np.vstack(batch_annotations).T
        inference_annotations = pd.concat(
            [
                inference_annotations,
                pd.DataFrame(batch_annotations, columns=columns),
            ]
        )
    inference_annotations.to_csv(Path(save_path, "inference_annotations.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument(
        "-cfg",
        "--config",
        help="Config file path",
        type=str,
        default="",
        required=True,
    )
    args = parser.parse_args()

    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())

    # get dataloader
    data_loader = get_inference_dataset(cfg.inference_data, cfg.inference_pipeline)

    # load classes config
    classes = load_classes(cfg.classes)

    # get model
    device = torch.device(cfg.device)
    model = get_model(cfg.model, classes, device, compile=cfg.compile)

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    inference(model, data_loader, classes, save_path, device, cfg)


if __name__ == "__main__":
    main()
