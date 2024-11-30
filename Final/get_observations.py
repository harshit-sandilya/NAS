import numpy as np
from train import train_model
from config_reader import Config
from preprocess import DataModule
import random

config = Config()
train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]

dataModules = []
entries = []
for file in train_file_list:
    config.train["train_bin_path"] = file
    dataModules.append(DataModule(config.train, config.preprocess))
    random_float = random.uniform(0.1, 1.0)
    dataModules[-1].setup(random_float)
    entries.append(len(dataModules[-1].train))

print(entries)

train_model([4, 4], config, dataModules[0])
