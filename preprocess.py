import pytorch_lightning as pl
import torch.utils.data as data
import data.prepare_dataset_utils.packed_dataset as packed_dataset
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


def de_binarize(filenames, n_chunks=1, block_size=256):
    return packed_dataset.PackedDataset(
        filenames, n_chunks, block_size, seed=12345, shuffle=False
    )


# if __name__ == "__main__":

#     parser= argparse.ArgumentParser()
#     parser.add_argument("--src_path",type=str)
#     args = parser.parse_args()
#     src_path = Path (args.src_path)
#     filenames = glob.glob(os.path.join(src_path, '**', '*.bin'), recursive=True)
#     print(filenames)
#     # filenames = ["dataset_out/bigger_0000000000.bin","dataset_out/bigger_0000000001.bin","dataset_out/bigger_0000000002.bin","dataset_out/bigger_0000000003.bin"]
#     de_binarize(filenames)


class DataModule(pl.LightningDataModule):
    def __init__(self, train_config, preprocess_config):
        super().__init__()
        self.train_config = train_config
        self.preprocess_config = preprocess_config

    def setup(self, stage: str = None):
        # filenames = glob.glob(os.path.join(self.train_config["bin_path"], '**', '*.bin'), recursive=True)

        # self.val_frac = self.preprocess_config["val_percent"] / 100
        # self.test_frac = self.preprocess_config["test_percent"] / 100
        # L = len(filenames)

        # Test_Filenames = filenames[:int(L*self.test_frac)]
        # Val_Filenames = filenames[int(L*self.test_frac):int(L*self.test_frac)+int(L*self.val_frac)]
        # Train_Filenames = filenames[int(L*self.test_frac)+int(L*self.val_frac):]

    #     Test_Filenames = glob.glob(os.path.join(self.train_config["test_bin_path"], '**', '*.bin'), recursive=True)
    #     Val_Filenames = glob.glob(os.path.join(self.train_config["validate_bin_path"], '**', '*.bin'), recursive=True)
    #     Train_Filenames = glob.glob(os.path.join(self.train_config["train_bin_path"], '**', '*.bin'), recursive=True)

    #     random.shuffle(Test_Filenames)
    #     random.shuffle(Val_Filenames)
    #     random.shuffle(Train_Filenames)

    #     self.train = de_binarize(Train_Filenames,1,self.train_config["context_length"]+1)
    #     self.val = de_binarize(Val_Filenames,1,self.train_config["context_length"]+1)
    #     self.test = de_binarize(Test_Filenames,1,self.train_config["context_length"]+1)
    #     self.vocab_size = self.preprocess_config["vocab_size"]

    # def train_dataloader(self):
    #     return data.DataLoader(
    #         self.train,
    #         batch_size=self.train_config["batch_size"],
    #         shuffle=False,
    #         num_workers=4,
    #     )

    # def val_dataloader(self):
    #     return data.DataLoader(
    #         self.val,
    #         batch_size=self.train_config["batch_size"],
    #         shuffle=False,
    #         num_workers=4,
    #     )

    # def test_dataloader(self):
    #     return data.DataLoader(
    #         self.test,
    #         batch_size=self.train_config["batch_size"],
    #         shuffle=False,
    #         num_workers=4,
    #     )
    
        self.vocab_size = self.preprocess_config["vocab_size"]
        self.train = StreamingDataset(
            input_dir=self.train_config["train_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
        )
        self.val = StreamingDataset(
            input_dir=self.train_config["validate_bin_path"],
            item_loader=TokensLoader(
                block_size=self.train_config["context_length"] + 1
            ),
            shuffle=False,
        )
        self.test = StreamingDataset(
            input_dir=self.train_config["test_bin_path"],
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
