import os

import torch
import torch.nn as nn
from core.checkpoint import load_checkpoint, save_checkpoint
from core.data import get_dataloader
from core.distributed import barrier, cleanup
from core.logging import get_writer, update_train_result, update_eval_result, log, get_logger, get_config
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .eval import evaluate
from .model import FFNet
from .train import train
from .utils import get_loss_fn, get_optimizer


def run(local_rank: int, nprocs: int, config: object) -> None:
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    ddp = nprocs > 1

    model = FFNet(config).to(device)
    grad_scaler = GradScaler() if config.amp else None
    loss_fn = get_loss_fn(config).to(device)
    optimizer, scheduler = get_optimizer(config, model)
    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(
        config, model, optimizer, scheduler, grad_scaler)

    if local_rank == 0:
        model_without_ddp = model
        writer = get_writer(config.save_path)
        logger = get_logger(os.path.join(config.save_path, "train.log"))
        logger.info(get_config(config.to_dict(), mute=False))
        val_loader = get_dataloader(config, split="val", ddp=False)

    config.batch_size = int(config.batch_size / nprocs)
    config.num_workers = int(config.num_workers / nprocs)
    train_loader, sampler = get_dataloader(config, split="train", ddp=ddp)

    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank],
                output_device=local_rank) if ddp else model

    for epoch in range(start_epoch, config.epochs + 1):  # start from 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        model, optimizer, loss_info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, local_rank, nprocs)
        scheduler.step()
        barrier(ddp)

        if local_rank == 0:
            eval = (epoch >= config.eval_start) and ((epoch - config.eval_start) % config.eval_freq == 0)
            update_train_result(epoch, loss_info, writer)
            log(logger, epoch, config.epochs, loss_info=loss_info)

            if eval:
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                model_without_ddp.load_state_dict(state_dict)
                curr_val_scores = evaluate(
                    model_without_ddp,
                    val_loader,
                    device,
                )
                hist_val_scores, best_val_scores = update_eval_result(epoch, curr_val_scores, hist_val_scores,
                                                                      best_val_scores, writer, state_dict,
                                                                      config.save_path)
                log(logger, epoch, config.epochs, None, curr_val_scores, best_val_scores)

            if (epoch % config.save_freq == 0):
                save_checkpoint(
                    epoch + 1,
                    model.module.state_dict() if ddp else model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict() if scheduler is not None else None,
                    loss_info,
                    hist_val_scores,
                    best_val_scores,
                    config.save_path,
                    grad_scaler.state_dict() if grad_scaler is not None else None,
                )

        barrier(ddp)

    if local_rank == 0:
        writer.close()
        print("Training completed. Best scores:")
        print(best_val_scores)

    cleanup(ddp)
