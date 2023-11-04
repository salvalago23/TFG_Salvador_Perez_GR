import matplotlib.pyplot as plt
import numpy as np
import torch

from classes.DQNagentClass import DQNAgent, DDQNAgent
from envs.createEnvs import createNNEnv
from utilities.plots import create_grids, plot_trajectory
from utilities.jsonRW import writeJSON, readJSON

#CREATING THE ENVIRONMENT
shape = "14x14"             # "5x5" or "14x14"
env = createNNEnv(shape)

#Hyperparameters
train = True            # train or test
show_stats = False       # show stats
export_to_JSON = True   # write JSON file
render = False           # render the results after training

NUM_DQN_AGENTS = 10     # number of DQN agents
NUM_DDQN_AGENTS = 10     # number of DDQN agents

SEED = 0                # random seed. 0 for all
NUM_NEURONS_FC1 = 128   # number of neurons for the first fully connected layer
NUM_NEURONS_FC2 = 128   # number of neurons for the second fully connected layer

#EPISODES_PER_AGENT = 1000
#MAX_STEPS_PER_EPISODE = 25

episodes = [1000,2000,3000]
steps = [50,100,150]

EPS_START = 1.0         # epsilon start value
EPS_END = 0.01          # epsilon end value
EPS_DECAY = 0.996       # epsilon decay rate
#EPS_DECAY = EPS_START/(EPISODES_PER_AGENT/2)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

for n_episode in episodes:
    EPISODES_PER_AGENT = n_episode
    for n_steps in steps:
        MAX_STEPS_PER_EPISODE = n_steps
        print("Starting training with ", EPISODES_PER_AGENT, "episodes and ", MAX_STEPS_PER_EPISODE, "steps per episode")

        #CREATING THE AGENTS
        agents_arr = []         # array of agents
        scores_arr = []         # array of scores of the episodes
        durations_arr = []      # array of durations of the episodes
        starting_positions = [] # array of starting positions for each agent

        for i in range(NUM_DQN_AGENTS):
            path_to_save = "../data/agent_models/pytorch/DQNagent"+str(i+1)+".pt" # path to save the model
            agent = DQNAgent(path_to_save, env, SEED, NUM_NEURONS_FC1, NUM_NEURONS_FC2, EPISODES_PER_AGENT, MAX_STEPS_PER_EPISODE, EPS_START, EPS_END, EPS_DECAY, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
            agents_arr.append(agent) # append the agent to the array

        for i in range(NUM_DDQN_AGENTS):
            path_to_save = "../data/agent_models/pytorch/DDQNagent"+str(i+1)+".pt" # path to save the model
            agent = DDQNAgent(path_to_save, env, SEED, NUM_NEURONS_FC1, NUM_NEURONS_FC2, EPISODES_PER_AGENT, MAX_STEPS_PER_EPISODE, EPS_START, EPS_END, EPS_DECAY, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
            agents_arr.append(agent) # append the agent to the array


        #TRAINING
        if train:
            print("Starting training of", NUM_DQN_AGENTS, "DQN agents and", NUM_DDQN_AGENTS, "DDQN agents")
            
            for agent in agents_arr:
                env.unwrapped.randomize_start_pos()     # randomize the starting position of the agent in the grid environment
                if agents_arr.index(agent) < NUM_DQN_AGENTS:
                    print("DQN Agent", agents_arr.index(agent)+1,"/",NUM_DQN_AGENTS)    # if the agent is a DQN agent
                else:
                    print("DDQN Agent", agents_arr.index(agent)+1-NUM_DQN_AGENTS,"/",NUM_DDQN_AGENTS)  # if the agent is a DDQN agent 

                scores, durations = agent.train()
                scores_arr.append(scores)
                durations_arr.append(durations)
                starting_positions.append(env.unwrapped.start_pos)

        if train:
            for i in range(len(agents_arr)):
                value_grid, policy_grid, string_policy_grid = create_grids(env, Qnet=agents_arr[i].qnetwork_local)
                
                if export_to_JSON:
                    if i < NUM_DQN_AGENTS:
                        algorithm = "DQN"
                    else:
                        algorithm = "DDQN"

                    writeJSON(algorithm, EPISODES_PER_AGENT, MAX_STEPS_PER_EPISODE, shape, starting_positions[i], value_grid, policy_grid, string_policy_grid)
                