import gymnasium as gym
from gymnasium import spaces
import numpy as np
from train import train_model
import torch

rewards = []
# rewards = torch.load("rewards.pt")
# rewards = rewards.tolist()


class Environment(gym.Env):
    def __init__(self, train_file_list, chunk_sizes):
        super(Environment, self).__init__()
        self.observation_space = spaces.Discrete(17000, start=1)
        self.action_space = spaces.Box(low=1, high=8, shape=(2,), dtype=int)
        self.train_file_list = train_file_list
        self.chunk_sizes = chunk_sizes
        self.current_chunk = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_chunk = 0
        return self.chunk_sizes[0], {}

    def step(self, action):
        reward = self.calc_reward(action)
        self.current_chunk += 1
        done = self.current_chunk == len(self.train_file_list) - 1
        info = {}
        return self.chunk_sizes[self.current_chunk], reward, done, False, info

    def calc_reward(self, action):
        print("NO OF LAYERS ===> ", action[0])
        print("NO OF HEADS ===> ", action[1])
        print("CHUNK SIZE ===> ", self.chunk_sizes[self.current_chunk])
        loss = train_model(action, self.train_file_list[self.current_chunk])
        with torch.no_grad():
            torch.cuda.empty_cache()
        reward = 200 - loss * 10 - action[0] - action[1]
        rewards.append(reward)
        tensor = torch.tensor(rewards)
        torch.save(tensor, "rewards.pt")
        print("REWARD ===> ", reward)
        print("LOSS ===> ", loss)
        return reward
