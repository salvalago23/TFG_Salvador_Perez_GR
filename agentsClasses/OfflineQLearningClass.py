from collections import defaultdict
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from envs.CreateEnvs import createOfflineEnv

class OfflineQLearningAgent:
    def __init__(self, id, shape, n_models, n_episodes, max_steps, reward, learning_rate, start_epsilon, epsilon_decay, final_epsilon, discount_factor):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.
        Args:
            learning_rate: The learning rate
            start_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.id = id
        self.algorithm = "Q-Learning"

        self.shape = shape
        self.env = createOfflineEnv(shape, n_models, reward, id=id, max_steps=max_steps)
        #self.env.unwrapped.randomize_start_pos()

        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.total_steps = 0

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def train(self):
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=self.n_episodes)

        for episode in tqdm(range(self.n_episodes)):
            done = False
            obs, _ = self.env.reset()
            obs = tuple(obs)
            
            # play one episode
            while not done:
                self.total_steps += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = truncated or terminated
                next_obs = tuple(next_obs)

                # update the agent
                self.update(obs, action, reward, done, next_obs)

                obs = next_obs

            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def plot_results(self, rolling_length=1, rolling_error=1):
        fig, axs = plt.subplots(ncols=3, figsize=(20, 5))
        
        axs[0].set_title("Episode rewards")
        axs[0].set_ylabel("Score")
        axs[0].set_xlabel("Episode #")
        axs[0].plot(range(len(self.env.return_queue)), np.array(self.env.return_queue).flatten())

        axs[1].set_title("Episode lengths")
        axs[1].set_ylabel("Steps")
        axs[1].set_xlabel("Episode #")
        # compute and assign a rolling average of the data to provide a smoother graph
        length_moving_average = (np.convolve(np.array(self.env.length_queue).flatten(), np.ones(rolling_length), mode="valid") / rolling_length)
        axs[1].plot(range(len(length_moving_average)), length_moving_average)

        axs[2].set_title("Training Error")
        axs[2].set_ylabel("Value")
        axs[2].set_xlabel("Temporal difference #")
        training_error_moving_average = (np.convolve(np.array(self.training_error), np.ones(rolling_error), mode="same") / rolling_error)
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

        fig.suptitle(f'Agent {self.id+1} - Stats')
        plt.show()
