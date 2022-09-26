import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from utils import get_data_module, prepare_config, \
    get_lit_model, get_model
import constants
from configs import HYPERPARAM_CONFIGS

np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
pl.seed_everything(constants.SEED, workers=True)

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
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="number of epochs")
    parser.add_argument("--num_samples", type=int,
                        default=10, help="number of trials")
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
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
    return parser

def train_func(config, args):
    # args.update(config)
    for k,v in config.items():
        if k == "lr":
            args[k] = v
        elif k == "batch_size":
            args["config"]["data"][k] = v
        else:
            args["config"]["model"][k] = v
    data_module = get_data_module(args)
    model = get_model(args)
    lit_model = get_lit_model(model, args)
    logdir = f'{constants.WORKING_DIR}/{constants.LOG_DIR}/' \
        + f'{args["config"]["data"]["dataset"]}/' \
        + f'{args["config"]["model"]["name"]}/' \
        + f'{args["exp_name"]}'
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir)
    trainer = pl.Trainer(
        deterministic=True,
        accelerator=args["accelerator"],
        devices=args["devices"] if args["devices"] else constants.DEVICES,
        max_epochs=args["num_epochs"],
        logger=tb_logger,
        log_every_n_steps=constants.LOG_STEPS,
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {
                    "val_loss": "val_loss",
                },
                on="validation_end"
            )
        ],
    )
    trainer.fit(lit_model, data_module)


def tune_fun(args):
    args = prepare_config(args)
    hyper_config = HYPERPARAM_CONFIGS[args['config']['model']['name']]
    scheduler = ASHAScheduler(
        max_t=args["num_epochs"],
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns= list(hyper_config.keys()),
        metric_columns=["val_loss"]
    )
    train_fn_with_parameters = tune.with_parameters(
        train_func,
        args=args
    )
    resources_per_trial = {"cpu": NUM_AVAIL_CPUS, "gpu": NUM_AVAIL_GPUS}
    tuner = tune.Tuner(
        tune.with_resources(
           train_fn_with_parameters,
           resources=resources_per_trial 
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=args["num_samples"],
        ),
        run_config=air.RunConfig(
            name=f"tun_{args['exp_name']}",
            progress_reporter=reporter,
        ),
        param_space=hyper_config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    tune_fun(args)

if __name__ == "__main__":
    main()




