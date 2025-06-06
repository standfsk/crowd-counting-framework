import argparse
import importlib
import os

from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset', type=str, required=True, help="dataset name")
    parser.add_argument('--save-path', type=str, required=True, help="save path")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--input-size', type=int, default=640, help="input image size")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'p2pnet', 'steerer'],
                        help="Model architecture to use.")
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--device', default="cpu", help="device to use. either gpu_id or cpu")
    parser.add_argument('--save', action="store_true", help="save result image")
    parser.add_argument('--log', action="store_true", help="save log")

    args = parser.parse_args()

    # save path
    args.save_path = os.path.join("output", args.save_path)

    # load config
    config = update_config(args).flatten()

    # run
    run(config)

def run(config: object) -> None:
    test_module = importlib.import_module(f'models.{config.network}.test')
    test_module.test(config)

if __name__ == '__main__':
    main()