from env2 import CarEnv
from sac import SAC
import itertools
import torch

ACTIONS = [x for x in itertools.product([-1,0,1], [-1,0,1])]


n_episodes = 1000
n_steps = 10000
rews = []
env = CarEnv()
model = SAC(9, 19, 0.7, 0.45, 0.001)
for e in range(n_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    for s in range(n_steps):
        act_idx, _, _ = model.get_action(torch.tensor(obs, device=torch.device('cuda')))
        act = ACTIONS[act_idx]
        obs_, rew = env.step(act)
        episode_reward += rew
        transition = (torch.as_tensor(obs, dtype=torch.double, device=torch.device('cuda')), 
                      torch.as_tensor(ACTIONS.index(act), dtype=torch.long, device=torch.device('cuda')), 
                      torch.as_tensor(rew, dtype=torch.double, device=torch.device('cuda')), 
                      torch.as_tensor(obs_, dtype=torch.double, device=torch.device('cuda')), 
                      s == n_steps - 1)
    rews.append(episode_reward)
    print('episode %d reward: %.3f max reward: %.3f' % (e, episode_reward, max(rews)))
    #env.dispose()
model.save("END2")
