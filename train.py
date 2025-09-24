import argparse
import importlib
import os

import torch
import torch.multiprocessing as mp

from core.distributed import init_seeds, setup
from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--save-path', type=str, required=True, help="save path")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--input-size', type=int, default=512, help="input image size")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'steerer', 'ffnet'],
                        help="Model architecture to use.")
    parser.add_argument('--eval-start', type=int, default=0, help="Epoch to start evaluation.")
    parser.add_argument('--eval-freq', type=int, default=1, help="Frequency (in epochs) to run evaluation.")
    parser.add_argument('--save-freq', type=int, default=1, help="Frequency (in epochs) to save.")
    parser.add_argument('--local-rank', type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action="store_true")

    args = parser.parse_args()

    # Set up DDP
    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")

    # save path
    args.save_path = os.path.join("output", "train", args.save_path)

    # update_config
    config = update_config(args).flatten()

    # run
    if args.nprocs > 1:
        config.lr = config.lr * args.nprocs
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, config))
    else:
        run(0, 1, config)


def run(local_rank: int, nprocs: int, config: object) -> None:
    train_module = importlib.import_module(f'models.{config.network}.trainer')

    if nprocs > 1:
        print(f"Rank {local_rank} process among {nprocs} processes.")
        init_seeds(config.seed + local_rank)
        setup(local_rank, nprocs)
        print(f"Initialized successfully. Training with {nprocs} GPUs.")
    train_module.run(local_rank, nprocs, config)


if __name__ == '__main__':
    main()