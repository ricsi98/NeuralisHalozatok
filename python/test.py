import torch
import torch.nn as nn
import torch.nn.functional as F
from env import CarEnv
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, inputSize, hiddenLayers):
        self.layers = [nn.Linear(inputSize, [hiddenLayers[0]])] # input
        for i, l in enumerate(hiddenLayers):
            if l != hiddenLayers[-1]:
                self.layers.append(nn.Linear(l, hiddenLayers[i+1]))
            else:
                self.layers.append(nn.Linear(l, 2)) # output
    
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x

    def learn(self, episode):
        pass

    def getAction(self, obs, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.uniform(2) * 2 - np.array([1, 1])
        return forward(obs)

if __name__ == '__main__':
    try:
        env = CarEnv()
        for j in range(1):
            env.reset()
            for i in range(500):
                print(env.step([1, 0]))
    finally:
        print('cleanup')
        env.dispose()
