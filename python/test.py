import torch
import numpy as np

from env import CarEnv
from models import *

KEYBOARD = True
if KEYBOARD:
    import keyboard

def package(obs, action, reward, obs_):
    return torch.tensor(np.array(obs)), action, reward, torch.tensor(np.array(obs_))

class AdaEpsilon:
    def __init__(self):
        self.t = 0

    def getEpsilon(self, addOne=True):
        if addOne: self.t += 1
        return 1.0 / (self.t/3000 + 1)

eps = AdaEpsilon()

def getAction(agent, obs):
    if KEYBOARD:
        left = 1 if keyboard.is_pressed('a') else 0
        right = 1 if keyboard.is_pressed('d') else 0
        up = 1 if keyboard.is_pressed('w') else 0
        down = 1 if keyboard.is_pressed('s') else 0
        if up == 0 and down == 0 and right == 0 and left == 0:
            return agent.getAction(torch.tensor(obs), eps.getEpsilon())
        return (up - down, right - left)
    else:
        return agent.getAction(torch.tensor(obs), eps.getEpsilon())

if __name__ == '__main__':
    memory = []
    try:
        env = CarEnv()
        agent = DQN([19, 256, 256, 9])
        #BootstrappedDQN([19, 256, 256, 256, 64, 9], 3, 2)
        #agent = DQN2([19, 256, 256, 9])
        #print("HEADS", len(agent.heads))
        opt = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.6)
        for j in range(3000):
            #agent.useRandomHead()
            #print('Using head ' + str(agent.heads.index(agent.selectedHead)))
            obs, reward = env.reset()
            sumrew = 0
            for i in range(250):
                action = getAction(agent, obs)#agent.getAction(torch.tensor(obs), 0.2)
                obs_, reward = env.step(action)
                sumrew += reward
                memory.append(package(obs, action, reward, obs_))
            env.dispose()
            agent.learn(memory, opt, 0.9)
            memory = []
            print('epoch, sum reward, gamma: ', j,sumrew,eps.getEpsilon(False))
            if j % 200 == 0:
                torch.save(agent.state_dict(), 'snapshot' + str(j) + ".pt")
    finally:
        print('cleanup')
        env.dispose()
