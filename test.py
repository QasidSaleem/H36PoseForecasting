import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from lit_models import LitModule


from lit_models import un_normalize_joints, convert_heatmaps_to_skelton
from utils import setup_data_and_model_from_args
import constants

NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
if NUM_AVAIL_GPUS:
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = None
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument(
        "--checkpoint", type=str, help="path of the model checkpoint"
    )
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    parser.add_argument(
        "--lr", type=float, default=constants.LR, help="Learning Rate"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LOSS, choices=[
                            "MSELoss",
                            "L1LOSS"
                        ], help="Loss Function")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--accelerator", type=str, default=ACCELERATOR, help="accelerator")
    parser.add_argument("--devices", type=int, default=None, help="number of devices")
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
    
    parser.add_argument(
        "--project_name",
        type=str,
        default="CUDALAB",
        help="W and b Sweep id")

    return parser

def run_test(args):
    data_module, lit_model, args = setup_data_and_model_from_args(args)
    checkpoint_path = args["checkpoint"]
    model = LitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args, model=lit_model.model, strict=False)
    data_module.setup()
    test_loader = data_module.val_dataloader()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1
    )
    predictions = trainer.predict(model, test_loader)
    # scaler = data_module.data_val.
    return predictions