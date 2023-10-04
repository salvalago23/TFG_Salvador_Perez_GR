import gym
#import pygame
#from gym import spaces
#import numpy as np

# Register the environment
gym.register(
    id='grid-v0',
    entry_point='gridEnv:MazeGameEnv',
    kwargs={'maze': None} 
)

#Maze config
maze = [
    ['.', '.', '#', '.', 'G'],
    ['.', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['S', '.', '#', '.', '.'],
]

# Test the environment
env = gym.make('grid-v0',maze=maze,disable_env_checker=True)


done = False

with open(f"../data/csv/history.csv", 'a') as f:
    #f.write(f"step,x,y,action,next_x,next_y,reward,done\n")
    for i in range(20):
        obs = env.reset()

        #env.render()

        for t in range(500):
            #pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            prev_state = [obs[0][0], obs[1][0], action]
            
            obs, rew, done, _ = env.step(action)

            f.write(f"{t},{prev_state[0]},{prev_state[1]},{prev_state[2]},{obs[0][0]},{obs[1][0]},{rew},{done}\n")

            #env.render()
            #print('Action:', action)
            #print('Observation:', obs)
            #print('Reward:', rew)
            #print('Done:', done)

            if done:
                break

            #pygame.time.wait(200)
        print("Agente", i+1, "terminado en", t, "pasos")

env.close()