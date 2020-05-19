import torch
import numpy as np

import keyboard

from env import CarEnv

def package(obs, action, reward, obs_):
    return torch.tensor(np.array(obs)), action, reward, torch.tensor(np.array(obs_))

def getAction():
    left = 1 if keyboard.is_pressed('a') else 0
    right = 1 if keyboard.is_pressed('d') else 0
    up = 1 if keyboard.is_pressed('w') else 0
    down = 1 if keyboard.is_pressed('s') else 0
    return (up - down, right - left)

def handleKeyboard():
    if keyboard.is_pressed('a'):
        print('pressed')

if __name__ == '__main__':
    memory = []
    try:
        env = CarEnv()
        obs, reward = env.reset()
        while True:
            action = getAction()
            obs_, reward = env.step(action)
            print("REW ", reward)
            memory.append(package(obs, action, reward, obs_))
        env.dispose()
        memory = []
    finally:
        print('cleanup')
        env.dispose()
