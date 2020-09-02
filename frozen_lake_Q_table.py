# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:18:43 2020
simple Q-learning code frozen lake 
@author: Mon pc
"""
import random
import os
import gym
import time
import numpy as np

env = gym.make('FrozenLake-v0')
epsilon =0.95
total_episodes=1000
max_step =100
learning_rate =0.75
# discount factore gammma 
gamma = 0.96
"""
    initialisation of  q Table 
    env.observation_space.n --> n states 
    env.action_sapce.n --> n  actions
"""
Q =np.zeros((env.observation_space.n,env.action_space.n))
# define action for the agent to choose and learn 
def select_action(state):
    action = 0
    if np.random.uniform(0,1)< epsilon:
        # choose a random action 
        action = env.action_space.sample()
    else:
        # choose action white higth max values in Q table
        action = np.argmax(Q[state, :])
    return action
def agent_learn (state, next_state, reward , action):
    predict = Q[state, action]
    target = reward+gamma*np.max(Q[next_state, :])
    Q[state, action]= Q[state, action ]+learning_rate * (target - predict)


"""
begin explore the q table

"""

for episode in range(total_episodes):
    state =env.reset()
    t =0
    while t< max_step:
        env.render()
        action =select_action(state)
        next_state, reward, done , info = env.step(action)
        agent_learn(state, next_state, reward, action)
        
        state= next_state
        t +=1
        if done :
            break
        time.sleep(0.1)
print(Q)




