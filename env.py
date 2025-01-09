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
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(1,), dtype=np.int32
        )
        # self.observation_space = spaces.Discrete(10000, start=1)
        self.action_space = spaces.Discrete(64)
        self.dict = [(i, j) for i in range(1, 9) for j in range(1, 9)]
        self.dataLoaders = dataLoaders
        self.sample_sizes = sample_sizes
        self.config = config
        self.current = 0
        self.value = 0
        self.lamda = 0.8

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current = 0
        self.value = 0
        return self.sample_sizes[self.current], {}

    def step(self, action):
        reward = self.calc_reward(action)
        self.value = reward + 0.75 * self.value
        self.current += 1
        done = self.current == len(self.dataLoaders) - 1
        info = {}
        return self.sample_sizes[self.current], reward, done, False, info

    def calc_reward(self, action):
        loss, ppl, time = train_model(self.dict[action], self.dataLoaders[self.current])
        with torch.no_grad():
            torch.cuda.empty_cache()
        reward_loss = math.exp(10 - loss)
        reward_time = math.exp(2 - ((time * 1000) / 6))
        reward_ppl = 10000 / (ppl + 1)
        total_reward = reward_loss + reward_time + reward_ppl
        mean_reward = (reward_loss + reward_time + reward_ppl) / 3
        variance_penalty = (
            abs(reward_loss - mean_reward)
            + abs(reward_time - mean_reward)
            + abs(reward_ppl - mean_reward)
        ) / 3
        total_reward -= self.lamda * variance_penalty
        rewards.append(total_reward)
        rewards_tensor = torch.tensor(rewards)
        torch.save(rewards_tensor, "rewards.pt")
        print("=====================================")
        print("STEP ===> ", self.current)
        print("NO OF HEADS ===> ", int(self.dict[action][1] + 1))
        print("NO OF LAYERS ===> ", int(self.dict[action][0] + 1))
        print("SAMPLE SIZE ===> ", self.sample_sizes[self.current])
        print("REWARD ===> ", total_reward)
        print("LOSS ===> ", loss)
        print("PPL ===> ", ppl)
        print("TIME ===> ", time)
        print("VALUE FUNCTION ===> ", self.value)
        print("=====================================")
        return total_reward
