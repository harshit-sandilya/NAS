import pytorch_lightning as pl
from lightning.data import (
    StreamingDataset,
    StreamingDataLoader,
)
from litdata import TokensLoader
import random


class DataModule(pl.LightningDataModule):
    def __init__(self, train_config, preprocess_config):
        super().__init__()
        self.train_config = train_config
        self.preprocess_config = preprocess_config
        # self.random_float = random.uniform(0.1, 1.0)
        self.random_float = 0.2

    def setup(self, stage: str = None):
        self.vocab_size = self.preprocess_config["vocab_size"]
        self.train = StreamingDataset(
            input_dir=self.train_config["train_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
            subsample=self.random_float,
        )
        self.test = self.train

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train,
            batch_size=self.train_config["batch_size"],
            pin_memory=True,
            num_workers=1,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.test,
            batch_size=self.train_config["batch_size"],
            pin_memory=True,
            num_workers=1,
        )
