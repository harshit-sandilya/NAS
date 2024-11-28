import torch
import json
import time
import numpy as np

from stable_baselines3 import DDPG, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from env import Environment
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parse action arguments")
    parser.add_argument(
        "--action", nargs=2, type=int, required=True, help="Action list of 2 integers"
    )
    args = parser.parse_args()
    return args.action


action = parse_args()

train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]

chunk_sizes = []
for file in train_file_list:
    with open(f"{file}/index.json") as f:
        index = json.load(f)
        chunk_sizes.append(int(index["config"]["chunk_size"]))

# Initialize environment
env = Environment(train_file_list, chunk_sizes)

# Load or initialize the model
model = DDPG.load("ddpg_transformer_100")  # Load the pre-trained model

chunk_file = "dataset/train-51"
observation, _ = env.reset()
observation, reward, done, _, info = env.step(action, chunk_file)
print("Reward: ", reward)
