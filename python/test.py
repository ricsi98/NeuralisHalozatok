import torch
import numpy as np

from env import CarEnv
from models import *

def package(obs, action, reward, obs_):
    return torch.tensor(np.array(obs)), action, reward, torch.tensor(np.array(obs_))

def adaGamma(t):
    return max(0.2, min(1, 1 - np.log((t + 1) / 1000)))

if __name__ == '__main__':
    memory = []
    try:
        env = CarEnv()
        agent = DQN([10,16,32,24,9])
        opt = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.9)
        for j in range(3000):
            obs, reward = env.reset()
            sumrew = 0
            for i in range(350):
                action = agent.getAction(torch.tensor(obs), adaGamma(j*350 + i))
                obs_, reward = env.step(action)
                sumrew += reward
                memory.append(package(obs, action, reward, obs_))
            env.dispose()
            agent.learn(memory, opt, 0.9)
            memory = []
            print('epoch, sum reward, gamma: ', j,sumrew,adaGamma(j*350))
            if j % 200 == 0:
                torch.save(agent.state_dict(), 'snapshot' + str(j) + ".pt")
    finally:
        print('cleanup')
        env.dispose()
