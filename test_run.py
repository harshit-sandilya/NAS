from config_reader import Config
from preprocess import DataModule
from train import train_model

config = Config()
train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]

dataModules = []
entries = []
for j in range(4):
    for file in train_file_list:
        config.train["train_bin_path"] = file
        dataModules.append(DataModule(config.train, config.preprocess))
        dataModules[-1].setup()
        entries.append(len(dataModules[-1].train))

print(train_model([8, 8], config, dataModules[0]))
