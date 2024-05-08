import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nkb_classification.dataset import get_inference_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import read_py_config


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    save_path: str,
    device: Union[torch.device, str],
) -> None:
    columns = loader.dataset.target_names.copy()
    columns.append("path")
    inference_annotations = pd.DataFrame(columns=columns)

    model.eval()

    for imgs, img_paths in tqdm(loader, leave=False, desc="Inference"):
        imgs = imgs.float().to(device)
        preds = model(imgs)
        batch_annotations = []
        for target_name in loader.dataset.target_names:
            pred = preds[target_name]
            pred = (
                pred.softmax(dim=-1)
                .argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )
            pred = [
                loader.dataset.idx_to_class[target_name][idx]
                for idx in pred
            ]
            batch_annotations.append(pred)
        batch_annotations.append(list(img_paths))
        batch_annotations = np.vstack(batch_annotations).T
        inference_annotations = pd.concat(
            [
                inference_annotations,
                pd.DataFrame(batch_annotations, columns=columns),
            ]
        )
    inference_annotations.to_csv(
        Path(save_path, "inference_annotations.csv"), index=False
    )


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
    data_loader = get_inference_dataset(
        cfg.inference_data, cfg.inference_pipeline
    )
    device = torch.device(cfg.device)

    # get model
    classes = data_loader.dataset.classes
    model = get_model(cfg.model, classes, device, compile=cfg.compile)

    # load weights
    model.load_state_dict(
        torch.load(cfg.model["checkpoint"], map_location="cpu")
    )

    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    inference(model, data_loader, save_path, device)


if __name__ == "__main__":
    main()
