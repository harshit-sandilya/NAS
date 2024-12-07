from train import train_model
from config_reader import Config
from preprocess import DataModule
import math

config = Config()
config.train["train_bin_path"] = "dataset/train-1"
dataModule = DataModule(config.train, config.preprocess)
dataModule.setup()

for i in range(8):
    for j in range(8):
        loss, ppl, time = train_model([i, j], dataModule.train)
        print("=====================================")
        print("SAMPLE SIZE ===> ", len(dataModule.train))
        print("NO OF HEADS ===> ", j + 1)
        print("NO OF LAYERS ===> ", i + 1)
        print(
            "REWARD ===> ",
            math.exp(10 - loss)
            + math.exp(2 - ((time * 1000) / 6))
            + (10000 / (ppl + 1)),
        )
        print("LOSS ===> ", loss)
        print("PPL ===> ", ppl)
        print("TIME ===> ", time)
        print("=====================================")
