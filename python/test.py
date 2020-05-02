import torch
import torch.nn as nn
import torch.nn.functional as F
from env import CarEnv
import numpy as np
import random
import itertools

ACTIONS = [x for x in itertools.product([-1,0,1], [-1,0,1])]

class DQN(nn.Module):
    def __init__(self, layers):
        super(DQN, self).__init__()
        self.layers = []
        for i, l in enumerate(layers[:-1]):
            self.layers.append(nn.Linear(l,layers[i+1]).double())
            self.add_module("layer"+str(i), self.layers[-1])
        
    
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x

    def learn(self, episode):
        pass

    def getAction(self, obs, epsilon):
        if np.random.uniform() < epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[torch.multinomial(self.forward(obs), 1).item()]

if __name__ == '__main__':
    try:
        env = CarEnv()
        agent = DQN([8,16,4,9])
        for j in range(1):
            obs, reward = env.reset()
            for i in range(500):
                obs, reward = env.step(agent.getAction(torch.tensor(obs), 0))
    finally:
        print('cleanup')
        env.dispose()
