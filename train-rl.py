from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import Environment
from config_reader import Config
from preprocess import DataModule
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import argparse

parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
parser.add_argument(
    "--mode",
    type=int,
    choices=[1, 2],
    required=True,
    help="Mode of operation: 1 or 2",
    default=1,
)
args = parser.parse_args()

config = Config()
if args.mode == 1:
    train_file_list = [f"dataset/train-{i}" for i in range(1, 26)]
elif args.mode == 2:
    train_file_list = [f"dataset/train-{i}" for i in range(26, 51)]

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
    save_freq=25,
    save_path="./logs/",
    name_prefix="dqn_nas",
    save_replay_buffer=True,
)


def make_env():
    return Environment(dataModules, entries, config)


env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_reward=True)

if os.path.exists("vec_normalize.pkl"):
    env = env.load("vec_normalize.pkl", env)

model = DQN("MlpPolicy", env, tensorboard_log="logs/dqn", gamma=0.1)
if os.path.exists("logs/dqn_nas_25_steps.zip"):
    model = model.load("logs/dqn_nas_25_steps.zip", env=env)
    model.load_replay_buffer("logs/dqn_nas_replay_buffer_25_steps.pkl")

model.learn(
    total_timesteps=25,
    callback=[TensorboardCallback(), checkpoint_callback],
)

model.save("dqn_transformer")
env.save("vec_normalize.pkl")
