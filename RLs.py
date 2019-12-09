"""
Created on Wed Sep 18 13:18:54 2019

@author: thoma
"""

from FitnessFunctions import OneMax, LeadingOnes, BinVal

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

# Problem Parameters
N = 10
fitness = OneMax()

# Define environment
class Environment():
    
    def __init__(self, N):
        self.N = N
        
    def reset(self):
        s = np.array([int(random.random() < 0.5) for _ in range(self.N)])
        self.current_state = s
        return deepcopy(self.current_state)
    
    def step(self, action):
        #self.current_state[action] = 1
        self.current_state[action] = int(not self.current_state[action])
        reward = fitness.eval1d(self.current_state)
        done = reward == fitness.best(self.N)
        return deepcopy(self.current_state), reward, done 
    
env = Environment(N)
    

# Define Network
class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs, learning_rate):
        super(PolicyNet, self).__init__()
        # network
        self.hidden1 = nn.Linear(n_inputs, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_outputs)
        
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)
    
    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))

# Net Parameters
alpha = 0.01
n_hidden1 = 50
n_hidden2 = 50

policy = PolicyNet(N, n_hidden1, n_hidden2, n_outputs=N, learning_rate=alpha)

# Training Parameters
num_episodes = 2000
rollout_limit = 200
discount_factor = 0.9
val_freq = 50

def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

# train policy network
try:
    training_rewards, losses = [], []
    print('start training')
    
    for i in range(num_episodes):
        rollout = []
        s = env.reset()
        for j in range(rollout_limit):
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():
                a_prob = policy(torch.from_numpy(np.atleast_2d(s)).float())
            a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action
            s1, r, done = env.step(a)
            rollout.append((s, a, r))
            s = s1
            if done: break
        
        # prepare batch
        rollout = np.array(rollout)
        states = np.vstack(rollout[:,0])
        actions = np.vstack(rollout[:,1])
        rewards = np.array(rollout[:,2], dtype=float)
        returns = compute_returns(rewards, discount_factor)
        
        # policy gradient update
        policy.optimizer.zero_grad()
        a_probs = policy(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
        loss = policy.loss(a_probs, torch.from_numpy(returns).float())
        loss.backward()
        policy.optimizer.step()
        
        # bookkeeping
        training_rewards.append(sum(rewards) / (j+1))
        losses.append(loss.item())
        
        # print
        if (i+1) % val_freq == 0:
            # validation
            validation_rewards = []
            for _ in range(10):
                s = env.reset()
                reward = 0
                for _ in range(rollout_limit):
                    with torch.no_grad():
                        a = policy(torch.from_numpy(np.atleast_2d(s)).float()).argmax().item()
                    s, r, done = env.step(a)
                    if done: break
                validation_rewards.append(r)
            print(f"Iteration: {i+1}, average final reward: {np.mean(validation_rewards)}, loss: {loss}")
    print('done')
except KeyboardInterrupt:
    print('interrupt')  


# plot results
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(range(1, len(training_rewards)+1), training_rewards, label='training reward')
plt.plot(moving_average(training_rewards))
plt.xlabel('episode'); plt.ylabel('reward')
plt.xlim((0, len(training_rewards)))
plt.legend(loc=4); plt.grid()
plt.subplot(212)
plt.plot(range(1, len(losses)+1), losses, label='loss')
plt.plot(moving_average(losses))
plt.xlabel('episode'); plt.ylabel('loss')
plt.xlim((0, len(losses)))
plt.legend(loc=4); plt.grid()
plt.tight_layout(); plt.show()
