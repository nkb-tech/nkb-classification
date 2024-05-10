import os
import argparse
from io import BytesIO

import onnx
import torch

from nkb_classification.dataset import get_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import read_py_config

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config", help="Config file path", type=str, default="", required=True
    )
    parser.add_argument("--to", type=str, required=True, help="torchscript or onnx")
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="PyTorch model weights"
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument(
        "--dynamic", type=str, default="batch", help="Dynamic axes for onnx export"
    )
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 224, 224],
        help="Model input shape only for api builder",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Export ONNX device")
    parser.add_argument("--save_path", type=str, default=".", help="Save path")
    args = parser.parse_args()
    assert args.dynamic in ["batch", "all", "none"]
    assert args.to in ["torchscript", "onnx"]
    assert len(args.input_shape) == 4
    return args


def main(args):
    # Load the model
    print(f"Export to {args.to}")

    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())

    # get model
    data_loader = get_dataset(cfg.train_data, cfg.train_pipeline)
    device = torch.device(args.device)

    # get model
    classes = data_loader.dataset.classes
    cfg.model["pretrained"] = False
    model = get_model(cfg.model, classes, device, compile=False)

    # load weights
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # warm-up
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)

    save_path = os.path.join(
        args.save_path, os.path.basename(args.weights.replace(".pth", f".{args.to}"))
    )

    if args.to == "torchscript":
        jit_model = torch.jit.script(
            model,
            example_inputs=[(fake_input,)],
        )
        torch.jit.save(jit_model, save_path)

    elif args.to == "onnx":
        output_names = list(classes.keys())

        dynamic_axes = None
        if args.dynamic != "none":
            # outputs
            dynamic_axes = {output_name: {0: "batch"} for output_name in output_names}
            # inputs
            dynamic_dims = {0: "batch"}
            if args.dynamic == "all":
                dynamic_dims[2] = "height"
                dynamic_dims[3] = "width"

            # merge inputs and outputs
            dynamic_axes.update({"images": dynamic_dims})

        with BytesIO() as f:
            torch.onnx.export(
                model,
                fake_input,
                f,
                opset_version=args.opset,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            f.seek(0)
            onnx_model = onnx.load(f)

        # check onnx model
        onnx.checker.check_model(onnx_model)
        onnx_model_optimized = onnx_model

        # try to optimize the graph
        if args.sim:
            try:
                onnx_model_optimized, check = onnxsim.simplify(onnx_model)
                assert check, "Assert `onnxsim.simplify` failed"
                print("Finish! Here is the difference:")
                onnxsim.model_info.print_simplifying_info(
                    onnx_model, onnx_model_optimized
                )
            except Exception as e:
                print(f"Simplifier failure: {e}")

        # save the final graph
        onnx.save(onnx_model_optimized, save_path)
    else:
        raise NotImplementedError(f"Got {args.to} not supported.")

    print(f"{args.to} export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
