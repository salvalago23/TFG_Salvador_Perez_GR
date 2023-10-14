from agent import Agent

import matplotlib.pyplot as plt
import torch

from IPython import display

import gymnasium as gym

# Get the environment and extract the number of actions.
gym.register(id='grid-v0', entry_point='NNmodeledEnv:NNGridWorldEnv')
#Maze config
maze = [
    ['.', '.', '#', '.', 'G'],
    ['.', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['S', '.', '#', '.', '.'],
]

grid_model_path = '../data/models/modelo_entorno.h5'
reward_model_path = '../data/models/modelo_reward.h5'

# Test the environment
env = gym.make('grid-v0', maze=maze, grid_model_path=grid_model_path, reward_model_path=reward_model_path)#, max_episode_steps=500)



#load the weights from file
agent = Agent(state_size=8,action_size=4,seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rbg_array'))
        plt.axix('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state,reward,done,_ = env.step(action)
        if done:
            break

env.close()