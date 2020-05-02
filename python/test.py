from env import CarEnv
import torch
from models import *


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
