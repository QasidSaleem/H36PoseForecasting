import os
import argparse
import sys
import json
import importlib
import pathlib
import shutil

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import Human36_data_module
from lit_models import LitModule
import constants

def get_config(args):
    """Method to load json config file"""
    with open(args["config"]) as f:
        return json.load(f)

def prepare_config(args):
    """Adds configs data to args"""
    cfg_fname = args['config']
    args['config'] = get_config(args)
    if "joints" in args["config"]["data"]['dataset']:
        save_dir = f"{constants.WORKING_DIR}/checkpoints/{args['config']['data']['dataset']}/{args['config']['model']['name']}/{args['exp_name']}"
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args["config"]["data"]["s_fname"] = f'{save_dir}/{args["config"]["data"]["s_fname"]}'
        shutil.copy2(cfg_fname, save_dir)
    return args

def get_data_module(args):
    return Human36_data_module(**args)

def get_model(args):
    module = importlib.import_module(constants.MODEL_CLASS_MODULE)
    model_class = getattr(module, args["config"]["model"]["name"])
    return model_class(**args)

def get_lit_model(model, args):
    return LitModule(model, args)

def setup_data_and_model_from_args(args):
    args = prepare_config(args)
    data_module = get_data_module(args)
    model = get_model(args)
    lit_model = get_lit_model(model, args)
    return data_module, lit_model, args

def get_callbacks(args):
    filename = f""\
        + "{epoch:02d} -{" \
        + f"{args['monitor']}" \
        + ":.2f}"

    save_dir = f"{constants.WORKING_DIR}/checkpoints/{args['config']['data']['dataset']}/{args['config']['model']['name']}/{args['exp_name']}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=args["n_checkpoints"],
        save_last=True,
        monitor=args["monitor"],
        mode=args["mode"],
        dirpath=save_dir,
        filename=filename,
    )
    early_stopping_callback = EarlyStopping(
        monitor=args["monitor"],
        patience=args["patience"],
        mode=args["mode"],
    )
    progress_bar_callback = TQDMProgressBar(refresh_rate=20)
    callbacks = [
                checkpoint_callback,
                progress_bar_callback,
    ]
    if args["use_es"]:
        callbacks.append(early_stopping_callback)
    
    return callbacks