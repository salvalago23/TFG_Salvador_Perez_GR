from agent import Agent
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import torch

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = Agent(state_size=2,action_size=4,seed=0)

def dqn(n_episodes = 100, max_t = 200, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.996):
    """Deep Q-Learning
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        
    """
    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state,_ = env.reset()
        score = 0
        for t in range(1, max_t+1):
            print()
            action = agent.act(state,eps)
            #print("State: ", state,"Action: ",action)
            next_state,reward,done,_,_ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our 
            ## replay buffer.
            state = next_state
            score += reward

            if done:
                print('Episode: {}\tSteps: {}'.format(i_episode,t))
                break

            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon
            #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
            #if i_episode %100==0:
                #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
                
            if np.mean(scores_window)>=1.0:
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                           np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
                break
    return scores

scores = dqn()

#I want to see the agent q values
for col in range(5):
    for row in range(5):
        state = np.array([row,col])
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        agent.qnetwork_local.eval()
        with torch.no_grad():
            action_values = agent.qnetwork_local(state)
        agent.qnetwork_local.train()
        print("State: ",state.cpu().data.numpy(),"Action values: ",action_values.cpu().data.numpy())


#Not worth printing the scores in this environment since they are always 0 except for the last step where it is 1 for
#reaching the goal state
#print(scores)

#plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()