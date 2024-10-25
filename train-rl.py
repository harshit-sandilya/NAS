import torch
import json
import time
import numpy as np

from stable_baselines3 import DDPG, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from env import Environment

train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]

chunk_sizes = []
for file in train_file_list:
    with open(f"{file}/index.json") as f:
        index = json.load(f)
        chunk_sizes.append(int(index["config"]["chunk_size"]))

print(chunk_sizes)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        self.logger.record("train/reward", self.locals["rewards"])
        return True


checkpoint_callback = CheckpointCallback(
    save_freq=1, save_path="./logs/", name_prefix="ddpg_nas"
)

env = Environment(train_file_list, chunk_sizes)
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="logs/ddpg")
model.learn(total_timesteps=50, callback=[TensorboardCallback(), checkpoint_callback])
model.save("ddpg_transformer")
