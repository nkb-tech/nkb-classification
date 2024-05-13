import argparse
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tensorrt as trt
import torch
import tqdm


class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }

    def __init__(
        self, weight: Union[str, Path], device: Optional[torch.device]
    ) -> None:
        super(TRTModule, self).__init__()
        self.weight = (
            Path(weight) if isinstance(weight, str) else weight
        )
        self.device = (
            device if device is not None else torch.device("cuda:0")
        )
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.INFO)
        logger.min_severity = trt.Logger.Severity.VERBOSE
        trt.init_libnvinfer_plugins(logger, namespace="")

        # Read file
        with open(self.weight, "rb") as f, trt.Runtime(
            logger
        ) as runtime:
            meta_len = int.from_bytes(
                f.read(4), byteorder="little"
            )  # read metadata length
            # metadata = json.loads(
            #     f.read(meta_len).decode("utf-8")
            # )  # read metadata
            model = runtime.deserialize_cuda_engine(
                f.read()
            )  # read engine
            import ipdb; ipdb.set_trace()

        context = model.create_execution_context()
        Binding = namedtuple(
            "Binding", ("name", "dtype", "shape", "data", "ptr")
        )
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(
                        i, tuple(model.get_profile_shape(0, i)[2])
                    )
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(
                self.device
            )
            bindings[name] = Binding(
                name, dtype, shape, im, int(im.data_ptr())
            )
        binding_addrs = OrderedDict(
            (n, d.ptr) for n, d in bindings.items()
        )
        batch_size = bindings["images"].shape[
            0
        ]  # if dynamic, this is instead max batch size

        self.dynamic = dynamic
        self.bindings = bindings
        self.model = model
        self.context = context
        self.output_names = output_names
        self.binding_addrs = binding_addrs
        self.fp16 = fp16

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = (
            profiler if profiler is not None else trt.Profiler()
        )

    def forward(self, im: torch.Tensor) -> List[torch.Tensor]:
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()

        if self.dynamic and im.shape != self.bindings["images"].shape:
            i = self.model.get_binding_index("images")
            self.context.set_binding_shape(
                i, im.shape
            )  # reshape if dynamic
            self.bindings["images"] = self.bindings["images"]._replace(
                shape=im.shape
            )
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(
                    tuple(self.context.get_binding_shape(i))
                )
        s = self.bindings["images"].shape
        assert (
            im.shape == s
        ), f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        self.stream.synchronize()
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        return y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Engine model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )
    args = parser.parse_args()
    return args


def main(args):
    rand_input = torch.rand(
        1, 3, 224, 224, dtype=torch.float32, device=args.device
    )

    model = TRTModule(args.weights, args.device)

    # warmup
    for _ in tqdm.tqdm(range(100)):
        _ = model(rand_input)


if __name__ == "__main__":
    main(parse_args())
