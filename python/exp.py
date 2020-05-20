import torch
import numpy as np

from env import CarEnv
from models import *




if __name__ == '__main__':
    try:
        env = CarEnv()
        agent = BootstrappedDQN([19, 256, 256, 256, 64, 9], 3, 2)
        agent.load_state_dict(torch.load("snapshots/snapshot.pt"))
        for j in range(len(agent.heads)):
            agent.useHead(j)
            print('Using head ' + str(agent.heads.index(agent.selectedHead)))
            obs, reward = env.reset()
            sumrew = 0
            for i in range(350):
                action = agent.getAction(torch.tensor(obs), 0)
                obs_, reward = env.step(action)
                sumrew += reward
            env.dispose()
            print('sum reward', sumrew)
    finally:
        print('cleanup')
        env.dispose()
