import os
import json
import torch


class Config:
    def __init__(self):
        self.config_files = self.get_files()
        self.preprocess, self.train, self.deepspeed = self.loads()
        self.dtype = self.get_dtype(self.train["precision"])
        # Process these to actual python objects

    def get_files(self):
        config_folder = "./config"
        config_files = []
        for file in os.listdir(config_folder):
            if file.endswith(".json"):
                config_files.append(os.path.join(config_folder, file))
        return config_files

    def loads(self):
        preprocess = {}
        train = {}
        deepspeed = {}
        for file in self.config_files:
            with open(file, "r") as f:
                config = json.load(f)
                if "preprocess" in file:
                    preprocess = config
                elif "train" in file:
                    train = config
                elif "deepspeed" in file:
                    deepspeed = config
        return preprocess, train, deepspeed

    def get_dtype(self, precision):
        if precision == "64" or precision == 64 or precision == "64-true":
            self.deepspeed = None
            return torch.float64
        elif precision == "32" or precision == 32 or precision == "32-true":
            return torch.float32
        elif (
            precision == "16"
            or precision == 16
            or precision == "16-true"
            or precision == "16-mixed"
        ):
            return torch.float16
        elif (
            precision == "bf16-true" or precision == "bf16-mixed" or precision == "bf16"
        ):
            self.deepspeed["bf16"] = {"enabled": True}
            return torch.bfloat16
        elif precision == "transformer-engine":
            self.deepspeed = None
            return torch.bfloat16
        elif precision == "transformer-engine-float16":
            self.deepspeed = None
            return torch.float16
        else:
            raise ValueError(f"Precision {precision} not supported.")
