import gymnasium as gym
#import gym
from gymnasium import spaces

#import pygame
#import numpy as np

# Register the environment
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
env = gym.make('grid-v0', maze=maze, grid_model_path=grid_model_path, reward_model_path=reward_model_path, max_episode_steps=500)

with open(f"../data/csv/history4.csv", 'a') as f:
    #f.write(f"step,y,x,action,next_y,next_x,reward,done\n")
    for i in range(1):
        obs, _ = env.reset()
        #env.render()

        t = 0
        done = False
        while not done:
            #pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            
            prev_state = [obs[0], obs[1], action]

            obs, rew, done, _, _ = env.step(action)

            f.write(f"{t},{prev_state[0]},{prev_state[1]},{prev_state[2]},{obs[0]},{obs[1]},{rew},{done}\n")
            
            t += 1
            #env.render()

            #pygame.time.wait(20)
        print("Agente", i+1, "terminado en", t, "pasos")

#env.close()