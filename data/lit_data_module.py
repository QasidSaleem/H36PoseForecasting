"""Data class for pytorch lightening"""
import argparse
import imp
import os
from pathlib import Path
from pickle import FALSE

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .torch_datasets import Human2dJoints, HumanHeatmaps


# Default arguments
BATCH_SIZE = 128
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS
SHUFFLE_DATA = True
PINMEMORY = False

class Human36_data_module(pl.LightningDataModule):
    """Lightning data class
    Learn more about at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)
        self.shuffle_data = self.args.get("shuffle_data", SHUFFLE_DATA)
        self.pin_mem = self.args.get("pin_memory", PINMEMORY)
    
    def setup(self, stage=None):
        dataset_name = self.args.get("dataset")

        if "joints" in dataset_name:
            self.data_train = Human2dJoints(mode="train", **self.args)
            self.data_val = Human2dJoints(mode="test", **self.args)
        elif "heatmaps" in dataset_name:
            self.data_train = HumanHeatmaps(**self.args)
            self.data_val = HumanHeatmaps(**self.args)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=self.shuffle_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
        )
    
    def test_dataloader(self):
        """Test and validation set are same"""
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
        )