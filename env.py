import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import train_model
import torch
import math

rewards = []


class Environment(gym.Env):
    def __init__(self, dataLoaders, sample_sizes, config):
        super(Environment, self).__init__()
        self.observation_space = spaces.Discrete(10000, start=1)
        self.action_space = spaces.MultiDiscrete([8, 8])
        self.dataLoaders = dataLoaders
        self.sample_sizes = sample_sizes
        self.config = config
        self.current = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current = 0
        return self.sample_sizes[0], {}

    def step(self, action):
        reward = self.calc_reward(action)
        self.current += 1
        done = self.current == len(self.dataLoaders) - 1
        info = {}
        return self.sample_sizes[self.current], reward, done, False, info

    def calc_reward(self, action):
        print("NO OF HEADS ===> ", int(action[1] + 1))
        print("NO OF LAYERS ===> ", int(action[0] + 1))
        print("SAMPLE SIZE ===> ", self.sample_sizes[self.current])
        loss, time = train_model(action, self.dataLoaders[self.current])
        with torch.no_grad():
            torch.cuda.empty_cache()
        reward = math.exp(10 - loss) + math.exp(2 - ((time * 1000) / 6))
        rewards.append(reward)
        rewards_tensor = torch.tensor(rewards)
        torch.save(rewards_tensor, "rewards.pt")
        print("REWARD ===> ", reward)
        print("LOSS ===> ", loss)
        return reward
