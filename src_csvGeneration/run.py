import gymnasium as gym
from gymnasium import spaces

#import pygame
#import numpy as np

# Register the environment
gym.register(id='grid-v0', entry_point='gridNewEnv:GridWorldEnv')

#Maze config
maze = [
    ['.', '.', '#', '.', 'G'],
    ['.', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['S', '.', '#', '.', '.'],
]

# Test the environment
env = gym.make('grid-v0',maze=maze, max_episode_steps=500)

with open(f"../data/csv/history.csv", 'a') as f:
    #f.write(f"step,y,x,action,next_y,next_x,reward,done\n")
    for i in range(3):
        obs, _ = env.reset()
        #env.render()

        t = 0
        done = False
        while not done:
            #pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            
            prev_state = [obs['agent'][0], obs['agent'][1], action]

            obs, rew, done, _, _ = env.step(action)

            #f.write(f"{t},{prev_state[0]},{prev_state[1]},{prev_state[2]},{obs['agent'][0]},{obs['agent'][1]},{rew},{done}\n")
            
            t += 1
            #env.render()

            #pygame.time.wait(20)
        print("Agente", i+1, "terminado en", t, "pasos")

#env.close()