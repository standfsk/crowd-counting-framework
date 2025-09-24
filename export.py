import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
from onnxsim import simplify

import models
from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Export')
    parser.add_argument('--save-path', type=str, required=True, help="onnx save path")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640), help="input image size [width, height]")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'p2pnet'],
                        help="Model architecture to use.")
    parser.add_argument('--backbone', type=str, required=True, help="Backbone network for the model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--device', default="cpu", help="device to use. either gpu_id or cpu")
    args = parser.parse_args()

    run(args)

def run(args: argparse.Namespace) -> None:
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    onnx_save_path = args.save_path
    input_size = args.input_size # (width, height)
    input_tensor = torch.zeros((1, 3, input_size[1], input_size[0])).to(device)

    config = update_config(args).flatten()
    config.state_dict = ""
    model = getattr(models, args.network)(config)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    with torch.set_grad_enabled(False):
        torch_output = model(input_tensor)
        input_names = ["input"]
        output_names = [f"output{x+1}" for x in range(len(torch_output))]

        torch.onnx.export(
            model,
            input_tensor,
            onnx_save_path,
            opset_version=12,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={x: {0: "batch"} for x in input_names + output_names}
        )

    onnx_model = onnx.load(onnx_save_path)
    onnx_sim, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim, onnx_save_path)

    # check onnx
    # onnx_model = onnx.load(onnx_save_path)
    # onnx.checker.check_model(onnx_model)
    # with open('OnnxShape.txt', 'w') as f:
    #     f.write(f"{onnx.helper.printable_graph(onnx_model.graph)}")

    # ort_session = onnxruntime.InferenceSession(onnx_save_path, providers=["CUDAExecutionProvider"])
    # ort_outs = ort_session.run(None, {'input': input_tensor.detach().cpu().numpy()})
    # print(torch.sum(torch_output).item(), np.sum(ort_outs[0]))
    # np.testing.assert_allclose(torch.sum(torch_output).item(), np.sum(ort_outs[0]), rtol=1e-03, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    main()
