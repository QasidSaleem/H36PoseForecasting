import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from lit_models import LitModule


from lit_models import un_normalize_joints, convert_heatmaps_to_skelton, evaluate
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
        help="If passed, logs experiment results to Weights & Biases",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        default=False,
        help="If passed predictions are saved as a numpy array",
    )

    parser.add_argument(
        "--save_visuals",
        action="store_true",
        default=False,
        help="If passed 10 random result images are saved",
    )

    parser.add_argument(
        "--lr", type=float, default=constants.LR, help="Learning Rate"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LOSS, choices=[
                            "MSELoss",
                            "L1Loss",
                            "SSIMLoss"
                        ], help="Loss Function")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--accelerator", type=str, default=ACCELERATOR, help="accelerator")
    parser.add_argument("--devices", type=int, default=None, help="number of gpu devices")
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
        help="W and b project name")

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
    if "Heatmaps" in args["config"]["model"]["name"]:
        predictions = convert_heatmaps_to_skelton(predictions, (1002, 1000), (64, 64))
    else:
        predictions = un_normalize_joints(args, predictions)
    
    # The shape will be num_examples, 30, 17, 2
    # 30 (0:10-> seeds, 10:20-> targets, 20:-> predictions)
    preds = predictions[:, 20:, :]
    targets = predictions[:, 10:20, :]

    eval_results = evaluate(preds, targets)
    print("evaluation results", eval_results)

    if args["save_visuals"]:
        image_dir = args["save_dir"]+ "/images"
        pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
        random_indicies = np.random.randint(len(np.random.randint(2, size=10)), size=10)
        figures = []
        for idx in random_indicies:
            pass




    return predictions