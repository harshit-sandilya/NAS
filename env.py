import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import train_model
import torch


class Environment(gym.Env):
    def __init__(self, dataLoaders, sample_sizes, config):
        super(Environment, self).__init__()
        self.observation_space = spaces.Discrete(10000, start=1)
        self.action_space = spaces.Box(low=1, high=8, shape=(2,), dtype=int)
        self.dataLoaders = dataLoaders
        self.sample_sizes = sample_sizes
        self.config = config
        self.current = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current = 0
        return self.sample_sizes[0], {}

    def step(self, action, chunk=None):
        reward = self.calc_reward(action, chunk)
        self.current += 1
        done = self.current == len(self.dataLoaders) - 1
        info = {}
        return self.sample_sizes[self.current], reward, done, False, info

    def calc_reward(self, action, chunk=None):
        print("NO OF HEADS ===> ", int(action[1]))
        print("NO OF LAYERS ===> ", int(action[0]))
        print("SAMPLE SIZE ===> ", self.sample_sizes[self.current])
        loss = train_model(action, self.config, self.dataLoaders[self.current])
        with torch.no_grad():
            torch.cuda.empty_cache()
        reward = 200 - loss * 10 - action[0] - action[1]
        print("REWARD ===> ", reward)
        print("LOSS ===> ", loss)
        return reward
