from stable_baselines3 import PPO
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
for j in range(4):
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
    save_freq=1,
    save_path="./logs/",
    name_prefix="ppo_nas",
    save_replay_buffer=True,
)

env = Environment(dataModules, entries, config)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo")


def get_latest_checkpoint(log_dir):
    checkpoint_files = [
        f for f in os.listdir(log_dir) if re.match(r"ppo_nas_\d+_steps\.zip", f)
    ]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(
        checkpoint_files, key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    return os.path.join(log_dir, latest_checkpoint)


latest_checkpoint = get_latest_checkpoint("./logs/")
if latest_checkpoint:
    model.load(latest_checkpoint)
    print(f"Loaded model from {latest_checkpoint}")

model.learn(
    total_timesteps=20000, callback=[TensorboardCallback(), checkpoint_callback]
)
model.save("ppo_transformer")
