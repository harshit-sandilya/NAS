from stable_baselines3 import DQN
from env import Environment
from config_reader import Config
from preprocess import DataModule
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import re

config = Config()
train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]


dataModules = []
entries = []
for file in train_file_list:
    config.train["train_bin_path"] = file
    dataModules.append(DataModule(config.train, config.preprocess))
    dataModules[-1].setup()
    entries.append(len(dataModules[-1].train))


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        self.logger.record("train/reward", self.locals["rewards"])
        return True


checkpoint_callback = CheckpointCallback(
    save_freq=10,
    save_path="./logs/",
    name_prefix="dqn_nas",
    save_replay_buffer=True,
)

env = Environment(dataModules, entries, config)
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs/dqn", gamma=0.75)

if os.path.exists("dqn_transformer.zip"):
    model = DQN.load("dqn_transformer", env=env)

model.learn(
    total_timesteps=30,
    callback=[TensorboardCallback(), checkpoint_callback],
)

model.save("dqn_transformer")
