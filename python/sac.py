import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def copy_model(fr, to):
    to.load_state_dict(fr.state_dict())

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super(MLP, self).__init__()
        self.layers = [nn.Linear(input_dim, layers[0]).cuda()]
        for i, l in enumerate(layers[:-1]):
            self.layers.append(nn.Linear(l, layers[i+1]).cuda())
        self.layers.append(nn.Linear(layers[-1], output_dim).cuda())
        
        for i, l in enumerate(self.layers):
            self.add_module("l" + str(i), l)
            
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x
    

class SAC:
    def __init__(self, action_size, observation_size, gamma, alpha, lr):
        self.alpha, self.gamma = alpha, gamma
        self.action_size, self.observation_size = action_size, observation_size
        
        self.critic_local_1 = MLP(observation_size, action_size, [128, 128]).double().cuda()
        self.critic_local_2 = MLP(observation_size, action_size, [128, 128]).double().cuda()
        
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_local_1.parameters(), lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr)
        
        self.critic_target_1 = MLP(observation_size, action_size, [128, 128]).double().cuda()
        self.critic_target_2 = MLP(observation_size, action_size, [128, 128]).double().cuda()
        
        copy_model(self.critic_local_1, self.critic_target_1)
        copy_model(self.critic_local_2, self.critic_target_2)
        
        self.memory = []
        
        self.actor_local = MLP(observation_size, action_size, [128, 128]).double().cuda()
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr)
        
    def get_action(self, observation):
        action_probabilities = self.actor_local(observation)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        
        return action, (action_probabilities, log_action_probabilities), max_probability_action
    
    def calculate_critic_losses(self, obs, act, rew, obs_, done):
        with torch.no_grad():
            next_obs_action, (action_probabilities, log_action_probabilities), _ = self.get_action(obs_)
            qf1_next_target = self.critic_target_1(obs_)
            qf2_next_target = self.critic_target_2(obs_)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            next_q_value = rew
            if not done:
                next_q_value = next_q_value + self.gamma * min_qf_next_target
                
        qf1 = self.critic_local_1(obs)
        qf2 = self.critic_local_2(obs)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        return qf1_loss, qf2_loss
    
    def calculate_actor_loss(self, obs):
        action, (action_probabilities, log_action_probabilities), _ = self.get_action(obs)
        qf1_pi = self.critic_local_1(obs)
        qf2_pi = self.critic_local_2(obs)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = log_action_probabilities * action_probabilities
        return policy_loss, log_action_probabilities
    
    def learn_transition(self, transition):
        obs, act, rew, obs_, done = transition
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        q1_loss, q2_loss = self.calculate_critic_losses(obs, act, rew, obs_, done)
        q1_loss.backward()
        q2_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()
        
        self.actor_optimizer.zero_grad()
        a_loss, _ = self.calculate_actor_loss(obs)
        a_loss.backward()
        self.actor_optimizer.step()
        
        return q1_loss.item(), q2_loss.item(), a_loss.item()
    
    def save(self, suffix):
        torch.save(self.critic_local_1.state_dict(), 'c1local' + str(suffix) + '.pt')
        torch.save(self.critic_local_2.state_dict(), 'c2local' + str(suffix) + '.pt')
        torch.save(self.critic_target_1.state_dict(), 'c1target' + str(suffix) + '.pt')
        torch.save(self.critic_target_2.state_dict(), 'c2target' + str(suffix) + '.pt')
        torch.save(self.actor_local.state_dict(), 'alocal' + str(suffix) + '.pt')
        
   