import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def learn(self, episode, optimizer, gamma):
        loss = nn.MSELoss()
        for i, j in enumerate(episode):
            obs, action, reward, obs_ = j
            action = ACTIONS.index(action)
            optimizer.zero_grad()
            y_ = self.forward(obs)
            target = y_.clone()
            target[action] = reward if j == len(episode) else reward + gamma *torch.max(self.forward(obs_))
            target.detach()
            L = loss(y_, target)
            L.backward()
            optimizer.step()

    def getAction(self, obs, epsilon):
        if np.random.uniform() < epsilon:
            return random.choice(ACTIONS)
        #return ACTIONS[torch.argmax(self.forward(obs)).item()]
        out = self.forward(obs)
        out = torch.max(out, torch.zeros_like(out))
        out = F.softmax(out)
        actionIdx = torch.multinomial(out, 1).item()
        return ACTIONS[actionIdx]
