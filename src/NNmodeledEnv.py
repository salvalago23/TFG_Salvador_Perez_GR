import numpy as np
import pygame

import gym
from gym import spaces
import keras
import tensorflow as tf

class NNGridWorldEnv(gym.Env):
    def __init__(self, maze, grid_model_path, reward_model_path):
        self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(int)  # Starting position
        self.goal_pos = (np.concatenate(np.where(self.maze == 'G'))).astype(int)  # Goal position

        #self.current_pos = self.start_pos #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.size = self.num_rows  # The size of the square grid

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.num_rows - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.num_rows - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),#UP
            1: np.array([1, 0]),#DOWN
            2: np.array([0, -1]),#LEFT
            3: np.array([0, 1]),#RIGHT
        }

        # Load models
        print("Loading models...")

        self.grid_model = tf.keras.models.load_model(grid_model_path)
        self.reward_model = tf.keras.models.load_model(reward_model_path)

        print("Models loaded")

        # Initialize Pygame
        #pygame.init()
        self.cell_size = 125
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True

    def reset(self, seed=None, options=None):
        self._agent_location = self.start_pos 
        self._target_location = self.goal_pos

        observation = self._get_obs()

        return observation, {}
    

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        input_model = np.column_stack(np.array([self._agent_location[0], self._agent_location[1], action]))
        new_pos = self.grid_model.predict(input_model, verbose=0)
        reward = self.reward_model.predict(input_model, verbose=0)

        #round the values
        new_pos = np.array(np.round(new_pos[0]), dtype=int)
        reward = int(np.round(reward[0][0]))

        # We use `np.clip` to make sure we don't leave the grid
        new_pos = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self._agent_location = new_pos

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()


        return observation, reward, terminated, False, ""
    

    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  
        print(self._agent_location[0], self._agent_location[1])
        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                """try:
                    print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')"""

                if self.maze[row, col] == '#':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Goal position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
                if (self._agent_location[0] == row) and (self._agent_location[1] == col):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
    
        pygame.display.update()


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
