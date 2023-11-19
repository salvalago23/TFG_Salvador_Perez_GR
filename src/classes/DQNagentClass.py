import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple, deque 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.createEnvs import createNNEnv

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc1_unit, fc2_unit):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns form environment."""
    def __init__(self, id, algorithm, shape, fc1_unit, fc2_unit, n_episodes, max_steps, initial_eps, final_eps, eps_decay, buffer_size, batch_size, gamma, tau, lr, update_every):
        """Initialize an Agent object.
        Params
        =======
            id (int): id of the agent
            env (gym environment): environment to interact with
            fc1_unit (int): number of nodes in first hidden layer
            fc2_unit (int): number of nodes in second hidden layer
            n_episodes (int): maximum number of training episodes
            max_steps (int): maximum number of steps per episode
            initial_eps (float): starting value of epsilon, for epsilon-greedy action selection
            final_eps (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            buffer_size (int): replay buffer size
            batch_size (int): mini batch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_every (int): how often to update the network
        """
        self.id = id
        self.algorithm = algorithm

        self.shape = shape
        self.env = createNNEnv(shape, id=id, max_steps=max_steps)
        self.env.unwrapped.randomize_start_pos()
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self.epsilon = initial_eps
        self.final_epsilon = final_eps
        self.epsilon_decay = eps_decay

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        self.training_error = []

        #Q- Network
        self.qnetwork_local = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n, fc1_unit, fc2_unit).to(device)
        self.qnetwork_target = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n, fc1_unit, fc2_unit).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
    
        # Replay memory 
        self.memory = ReplayBuffer(self.env.action_space.n, self.buffer_size, self.batch_size)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
        self.total_steps = 0
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step+1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory)>self.batch_size:
                experience = self.memory.sample()
                self.learn(experience, self.gamma)

    def get_action(self, state):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > self.epsilon:
            #print("Action values: ",action_values.cpu().data.numpy())
            return np.argmax(action_values.cpu().data.numpy())
        else:
            #print("Random action")
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        #Compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma* labels_next*(1-dones))
        
        # Calculate temporal differences
        temporal_differences = labels - predicted_targets

        # Append temporal differences to the list
        self.training_error.extend(temporal_differences.detach().cpu().numpy())

        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.tau)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

    def train(self):
        """Deep Q-Learning
        Params
        ======
            n_episodes (int): maximum number of training epsiodes
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon 
            eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        """
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=self.n_episodes)

        for episode in tqdm(range(self.n_episodes)):
            done = False
            state, _ = self.env.reset()

            while not done:
                self.total_steps += 1
                action = self.get_action(state)
                #print("State: ", state,"Action: ",action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = truncated or terminated

                self.step(state, action, reward, next_state, done)
                ## above step decides whether we will train(learn) the network actor (local_qnetwork) or we will fill the replay buffer
                ## if len replay buffer is equal to the batch size then we will train the network or otherwise we will add experience tuple in our replay buffer.
                state = next_state
            
            self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)## decrease the epsilon

            #eps = max(self.eps_end, eps-self.eps_decay)

        # save the model weights
        #torch.save(self.qnetwork_local.state_dict(), self.path_to_save)


    def plot_results(self, rolling_length=1, rolling_error=1):
        print("Agent", self.id+1, "steps stats:", "\n  -Average:", round(np.mean(self.env.length_queue), 2), "\n  -Std dev:", round(np.std(self.env.length_queue), 2), "\n   -Median:", int(np.median(self.env.length_queue)), "\n     -Best:", np.min(self.env.length_queue))
        fig, axs = plt.subplots(ncols=3, figsize=(20, 5))

        axs[0].set_title("Episode rewards")
        axs[0].set_ylabel("Score")
        axs[0].set_xlabel("Episode #")
        axs[0].plot(range(len(self.env.return_queue)), np.array(self.env.return_queue).flatten())

        axs[1].set_title("Episode lengths")
        axs[1].set_ylabel("Steps")
        axs[1].set_xlabel("Episode #")
        length_moving_average = (np.convolve(np.array(self.env.length_queue).flatten(), np.ones(rolling_length), mode="valid") / rolling_length)
        axs[1].plot(range(len(length_moving_average)), length_moving_average)

        axs[2].set_title("Training Error")
        axs[2].set_ylabel("Value")
        axs[2].set_xlabel("Temporal difference #")
        training_error_moving_average = (np.convolve([float(e) for e in self.training_error], np.ones(rolling_error), mode="valid") / rolling_error)

        #training_error_moving_average = (np.convolve(np.array(self.training_error), np.ones(rolling_error), mode="valid") / rolling_error)
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

        fig.suptitle(f'Agent {self.id+1} - Stats')
        plt.show()

class DDQNAgent(DQNAgent):
    """Interacts with and learns form environment."""

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        #Compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)

        #################Updates for Double DQN learning###########################
        self.qnetwork_local.eval()
        with torch.no_grad():
            actions_q_local = self.qnetwork_local(next_state).detach().max(1)[1].unsqueeze(1).long()
            labels_next = self.qnetwork_target(next_state).gather(1,actions_q_local)
        self.qnetwork_local.train()
        ############################################################################

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma* labels_next*(1-dones))

        # Calculate temporal differences
        temporal_differences = labels - predicted_targets

        # Append temporal differences to the list
        self.training_error.extend(temporal_differences.detach().cpu().numpy())

        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.tau)


class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)