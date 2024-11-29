import pytorch_lightning as pl
import torch.utils.data as data

# import data.prepare_dataset_utils.packed_dataset as packed_dataset
import glob
import os
import random
import os
from lightning.data import (
    StreamingDataset,
    CombinedStreamingDataset,
    StreamingDataLoader,
)
from litdata import TokensLoader


# def de_binarize(filenames, n_chunks=1, block_size=256):
#     return packed_dataset.PackedDataset(
#         filenames, n_chunks, block_size, seed=12345, shuffle=False
#     )


class DataModule(pl.LightningDataModule):
    def __init__(self, train_config, preprocess_config):
        super().__init__()
        self.train_config = train_config
        self.preprocess_config = preprocess_config

    def setup(self, stage: str = None):
        self.vocab_size = self.preprocess_config["vocab_size"]
        self.train = StreamingDataset(
            input_dir=self.train_config["train_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
        )
        self.val = StreamingDataset(
            input_dir=self.train_config["train_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
        )
        self.test = StreamingDataset(
            input_dir=self.train_config["train_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
        )

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train,
            batch_size=self.train_config["batch_size"],
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val,
            batch_size=self.train_config["batch_size"],
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.test,
            batch_size=self.train_config["batch_size"],
            pin_memory=True,
            num_workers=4,
        )
