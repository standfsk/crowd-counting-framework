import argparse
import importlib
import os

from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-path', type=str, required=True, help="dataset name")
    parser.add_argument('--save-path', type=str, required=True, help="save path")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'steerer', 'ffnet'],
                        help="Model architecture to use.")
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--device', default="cpu", help="device to use. either gpu_id or cpu")

    args = parser.parse_args()

    # save path
    args.save_path = os.path.join("output", "inference", args.save_path)

    # load config
    config = update_config(args).flatten()

    # run
    run(config)

def run(config: object) -> None:
    inference_module = importlib.import_module(f'models.{config.network}.inference')
    inference_module.inference(config)

if __name__ == '__main__':
    main()