import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import train_model
import torch
import math
import os

if os.path.exists("rewards.pt"):
    rewards = torch.load("rewards.pt")
    rewards = rewards.tolist()
else:
    rewards = []


class Environment(gym.Env):
    def __init__(self, dataLoaders, sample_sizes, config, last_step=0):
        super(Environment, self).__init__()
        self.observation_space = spaces.Discrete(10000, start=1)
        self.action_space = spaces.MultiDiscrete([8, 8])
        self.dataLoaders = dataLoaders
        self.sample_sizes = sample_sizes
        self.config = config
        self.current = 0
        self.last_step = last_step

    def reset(self, seed=None):
        global rewards
        super().reset(seed=seed)
        # self.current = self.last_step % len(self.dataLoaders)
        # rewards = rewards[: self.last_step]
        # self.last_step = 0
        self.current = 0
        return self.sample_sizes[self.current], {}

    def step(self, action):
        reward = self.calc_reward(action)
        self.current += 1
        done = self.current == len(self.dataLoaders) - 1
        info = {}
        return self.sample_sizes[self.current], reward, done, False, info

    def calc_reward(self, action):
        loss, ppl, time = train_model(action, self.dataLoaders[self.current])
        with torch.no_grad():
            torch.cuda.empty_cache()
        reward = (
            math.exp(10 - loss)
            + math.exp(2 - ((time * 1000) / 6))
            + (10000 / (ppl + 1))
        )
        rewards.append(reward)
        rewards_tensor = torch.tensor(rewards)
        torch.save(rewards_tensor, "rewards.pt")
        print("=====================================")
        print("STEP ===> ", self.current)
        print("NO OF HEADS ===> ", int(action[1] + 1))
        print("NO OF LAYERS ===> ", int(action[0] + 1))
        print("SAMPLE SIZE ===> ", self.sample_sizes[self.current])
        print("REWARD ===> ", reward)
        print("LOSS ===> ", loss)
        print("PPL ===> ", ppl)
        print("TIME ===> ", time)
        print("=====================================")
        return reward
