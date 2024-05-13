import argparse
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Union

import torch

import nkb_classification
from nkb_classification.dataset import get_dataset
from nkb_classification.model import get_model
from nkb_classification.utils import export_formats, read_py_config


def str2bool(v: Union[bool, str, Any]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        help="Config file path",
        type=str,
        default="",
        required=True,
    )
    parser.add_argument(
        "--to",
        type=str,
        required=True,
        help="torchscript, onnx or engine (tensorrt)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="PyTorch model weights",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    parser.add_argument(
        "--dynamic",
        type=str,
        default="batch",
        help="Dynamic axes for onnx export",
    )
    parser.add_argument(
        "--sim", action="store_true", help="simplify onnx model"
    )
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 224, 224],
        help="Model input shape only for api builder",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Export ONNX device"
    )
    parser.add_argument(
        "--save_path", type=str, default=".", help="Save path"
    )
    parser.add_argument(
        "--half",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to compute in fp16 or not",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to verbose during export or not",
    )
    args = parser.parse_args()
    assert args.dynamic in ["batch", "all", "none"]
    assert args.to in ["torchscript", "onnx", "engine"]
    assert len(args.input_shape) == 4
    return args


def main(args):
    # Load the model
    print(f"Export to {args.to}")

    device = torch.device(args.device)

    # tensorrt device
    # if device.type == "cuda":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    #     device = torch.device("cuda:0")

    cfg_file = args.config
    exec(read_py_config(cfg_file), globals(), globals())

    # get model
    data_loader = get_dataset(cfg.train_data, cfg.train_pipeline)

    # get model
    classes = data_loader.dataset.classes

    cfg.model["pretrained"] = False
    model = get_model(cfg.model, classes, device, compile=False)

    # load weights
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    fmt = args.to.lower()  # to lowercase
    fmts = tuple(
        export_formats()["Argument"][1:]
    )  # available export formats
    flags = [x == fmt for x in fmts]
    jit_fmt, onnx_fmt, engine_fmt = flags  # export booleans

    if engine_fmt:
        onnx_fmt = True

    description = f'NKBTech classification {cfg.experiment["name"]} model trained on Dog_expo_Vladimir_02_07_2023'
    metadata = {
        "description": description,
        "author": "NKBTech",
        "date": datetime.now().isoformat(),
        "version": nkb_classification.__version__,
        "batch": args.input_shape[0],
        "imgsz": args.input_shape[2:4],
        "classes": classes,
    }  # model metadata

    # warm-up
    fake_input = torch.rand(
        args.input_shape, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        for _ in range(2):
            _ = model(fake_input)

    print("Warm up made successfully!")

    if onnx_fmt:
        import onnx

        output_names = list(classes.keys())

        dynamic_axes = None
        if args.dynamic != "none":
            # outputs
            dynamic_axes = {
                output_name: {0: "batch"}
                for output_name in output_names
            }
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
                verbose=args.verbose,
                do_constant_folding=True,
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
                import onnxsim

                onnx_model_optimized, check = onnxsim.simplify(
                    onnx_model
                )
                assert check, "Assert `onnxsim.simplify` failed"
                print("Finish! Here is the difference:")
                onnxsim.model_info.print_simplifying_info(
                    onnx_model, onnx_model_optimized
                )
            except Exception as e:
                print(f"Simplifier failure: {e}")

        # Metadata
        for k, v in metadata.items():
            meta = onnx_model_optimized.metadata_props.add()
            meta.key, meta.value = k, str(v)

        save_path = os.path.join(
            args.save_path,
            os.path.basename(args.weights.replace(".pth", ".onnx")),
        )

        # save the final graph
        onnx.save(onnx_model_optimized, save_path)

    if jit_fmt:
        jit_model = torch.jit.trace(
            model,
            example_inputs=fake_input,
            check_trace=True,
            strict=False,
        )
        extra_files = {"config.txt": json.dumps(metadata)}
        if args.sim:
            print(f"Optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(jit_model)._save_for_lite_interpreter(
                save_path, _extra_files=extra_files
            )
        else:
            jit_model.save(save_path, _extra_files=extra_files)
        torch.jit.save(jit_model, save_path)

    if engine_fmt:
        assert (
            device.type != "cpu"
        ), "export running on CPU but must be on GPU, i.e. use 'device=0'"
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.INFO)
        if args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(
            device
        ).total_memory

        flag = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse_from_file(save_path):
            raise RuntimeError(f"failed to load ONNX file: {save_path}")

        inputs = [
            network.get_input(i) for i in range(network.num_inputs)
        ]
        outputs = [
            network.get_output(i) for i in range(network.num_outputs)
        ]
        for inp in inputs:
            print(
                f'input "{inp.name}" with shape {inp.shape} {inp.dtype}'
            )
        for out in outputs:
            print(
                f'output "{out.name}" with shape {out.shape} {out.dtype}'
            )

        if args.dynamic != "none":
            shape = fake_input.shape
            if shape[0] <= 1:
                print(
                    f"WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'"
                )
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(
                    inp.name,
                    (1, *shape[1:]),
                    (max(1, shape[0] // 2), *shape[1:]),
                    shape,
                )
            config.add_optimization_profile(profile)

        print(
            f"Building FP{16 if builder.platform_has_fast_fp16 and args.half else 32} engine as {save_path}"
        )
        if builder.platform_has_fast_fp16 and args.half:
            config.set_flag(trt.BuilderFlag.FP16)

            # explicit set inputs to fp16
            for idx in range(network.num_inputs):
                input_l = network.get_input(idx)
                input_l.dtype = trt.float16

            # explicity set outputs to fp16 or uint16
            for idx in range(network.num_outputs):
                output_l = network.get_output(idx)
                if output_l.dtype == trt.float32:
                    output_l.dtype = trt.float16

        save_path = os.path.join(
            args.save_path,
            os.path.basename(args.weights.replace(".pth", ".engine")),
        )

        del model
        torch.cuda.empty_cache()

        # Write file
        with builder.build_engine(network, config) as engine, open(
            save_path, "wb"
        ) as t:
            # Metadata
            meta = json.dumps(metadata)
            t.write(
                len(meta).to_bytes(4, byteorder="little", signed=True)
            )
            t.write(meta.encode())
            # Model
            t.write(engine.serialize())

    print(f"{args.to} export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
