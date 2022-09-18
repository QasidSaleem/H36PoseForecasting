import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch

from utils import setup_data_and_model_from_args, get_callbacks
import constants

np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
pl.seed_everything(constants.SEED)

NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="number of epochs")
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=constants.LR, help="Learning Rate"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LR, choices=[
                            "MSELoss",
                            "L1LOSS"
                        ], help="Loss Function")
    # Schedulers parameters
    parser.add_argument("--one_cycle_max_lr", type=float, default=None)
    parser.add_argument("--one_cycle_total_steps", type=int, default=constants.ONE_CYCLE_TOTAL_STEPS)
    # Callbacks
    parser.add_argument("--use_es", action="store_true",
                        help="use early stopping or not")
    parser.add_argument("--n_checkpoints", type=int,
                        default=constants.N_CHECKPOINTS, help="number of checkpoints")
    parser.add_argument("--patience", type=int,
                        default=constants.PATIENCE, help="patience for early stopping/checkpointing")
    parser.add_argument("--mode", type=str,
                        default="min", choices=[
                            "min",
                            "max"
                        ], help="mode for early stopping/checkpointing")
    parser.add_argument("--monitor", type=str,
                        default="val_loss", choices=[
                            "val_loss",
                            "train_loss"
                        ], help="monitor for early stopping/checkpointing")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers for dataloaders"
    )
    parser.add_argument(
        "--pin_memory",
        type=str,
        default=constants.PINMEMORY,
        help="pin memory for dataloader")
    parser.add_argument("--help", "-h", action="help")
    return parser

def _run_experiment(args):
    callbacks = get_callbacks(args)
    logdir = f'{constants.WORKING_DIR}/{constants.LOG_DIR}/' \
        + f'{args["config"]["data"]["dataset"]}/' \
        + f'{args["config"]["model"]["name"]}/' \
        + f'{args["exp_name"]}'

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    _run_experiment(args)

if __name__ == "__main__":
    main()
