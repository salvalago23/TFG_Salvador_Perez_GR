import numpy as np
import random
from tqdm import tqdm
from collections import namedtuple, deque 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_unit, fc2_unit):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns form environment."""
    def __init__(self, path_to_save, env, seed, fc1_unit, fc2_unit, n_episodes, max_steps, initial_eps, final_eps, eps_decay, buffer_size, batch_size, gamma, tau, lr, update_every):
        """Initialize an Agent object.
        Params
        =======
            path_to_save (str): path to save the model weights
            env (gym environment): environment to interact with
            seed (int): random seed
            fc1_unit (int): number of nodes in first hidden layer
            fc2_unit (int): number of nodes in second hidden layer
            n_episodes (int): maximum number of training episodes
            max_steps (int): maximum number of timesteps per episode
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

        self.path_to_save = path_to_save
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.seed = random.seed(seed)
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.eps_start = initial_eps
        self.eps_end = final_eps
        self.eps_decay = eps_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        #Q- Network
        self.qnetwork_local = QNetwork(env.observation_space.shape[0], env.action_space.n, seed, fc1_unit, fc2_unit).to(device)
        self.qnetwork_target = QNetwork(env.observation_space.shape[0], env.action_space.n, seed, fc1_unit, fc2_unit).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
    
        # Replay memory 
        self.memory = ReplayBuffer(env.action_space.n, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>self.batch_size:
                experience = self.memory.sample()
                self.learn(experience, self.gamma)

    def act(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
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
        ## TODO: compute and minimize the loss
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
            max_steps (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon 
            eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        """
        scores = [] # list containing score from each episode
        durations = [] # list containing duration of each episode
        eps = self.eps_start

        #for i_episode in range(1, self.n_episodes+1):
        for episode in tqdm(range(self.n_episodes)):
            state,_ = self.env.reset()
            score = 0
            for t in range(1, self.max_steps+1):
                action = self.act(state,eps)
                #print("State: ", state,"Action: ",action)
                next_state,reward,done,_,_ = self.env.step(action)
                self.step(state,action,reward,next_state,done)
                ## above step decides whether we will train(learn) the network actor (local_qnetwork) or we will fill the replay buffer
                ## if len replay buffer is equal to the batch size then we will train the network or otherwise we will add experience tuple in our replay buffer.
                state = next_state
                score += reward

                if done or t == self.max_steps:
                    #print('Episode: {}\tSteps: {}'.format(i_episode,t))
                    scores.append(score)
                    durations.append(t)
                    break
            
            eps = max(eps*self.eps_decay,self.eps_end)## decrease the epsilon

        # save the model weights
        torch.save(self.qnetwork_local.state_dict(), self.path_to_save)

        return scores, durations


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
        ## TODO: compute and minimize the loss
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

        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.tau)


class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
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