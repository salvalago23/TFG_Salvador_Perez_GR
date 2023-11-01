import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc1_unit=None, fc2_unit=None):
        super(NeuralNetwork, self).__init__()
        if fc1_unit is None:
            self.big = False
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, output_size)
        else:
            self.big = True
            self.fc1 = nn.Linear(input_size, fc1_unit)
            self.fc2 = nn.Linear(fc1_unit, fc2_unit)
            self.fc3 = nn.Linear(fc2_unit, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        if not self.big:
            x = self.fc2(x)
        else:
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        return x


class NNGridWorldEnv(gym.Env):
    def __init__(self, maze, grid_model_path, reward_model_path, randomStart, render):
        self.maze = np.array(maze["maze"])  # Maze represented as a 2D numpy array

        if randomStart:
            #There is a predifined start position with an S in the maze. Before assigning a random start position I have to convert the older one to "."
            self.maze[self.maze == 'S'] = '.'
            start_pos = maze["starting_pos"][np.random.randint(0, len(maze["starting_pos"]))]
            self.maze[start_pos[0], start_pos[1]] = 'S'
        
        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)  # Starting position
        self.goal_pos = (np.concatenate(np.where(self.maze == 'G'))).astype(np.int32)  # Goal position
        self.num_rows, self.num_cols = self.maze.shape

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.num_rows, self.num_cols]), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Load models
        print("Loading models...")
        if self.num_rows == 5:
            self.grid_model = NeuralNetwork(3, 2)
            self.reward_model = NeuralNetwork(3, 1)
        elif self.num_rows == 14:
            self.grid_model = NeuralNetwork(3, 2, 256, 128)
            self.reward_model = NeuralNetwork(3, 1, 64, 64)

        self.grid_model.load_state_dict(torch.load(grid_model_path))
        self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.grid_model.eval()
        self.reward_model.eval()
        print("Models loaded")

        if render:
        # Initialize Pygame
            pygame.init()
            if self.num_rows == 5:
                self.cell_size = 125
            elif self.num_rows == 14:
                self.cell_size = 25
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def _is_valid_position(self, pos):
        row, col = pos
        # If agent goes out of the grid or if the agent hits an obstacle
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols or self.maze[row, col] == '#':
            return False
        return True

    def reset(self, seed=None, options=None):
        self._agent_location = self.start_pos 
        self._target_location = self.goal_pos

        return self._agent_location, {}
    
    def step(self, action):
        #No me queda muy claro pq tiene que ser un [[[]]] y si es necesario, pero si no tiene esta forma el modelo de torch protesta
        #pq podria ser mas lento
        model_input = np.array([np.column_stack(np.array([float(self._agent_location[0]), float(self._agent_location[1]), float(action)]))])
        input_tensor = torch.tensor(model_input, dtype=torch.float32)

        #round the values
        new_pos = np.int32(np.round(self.grid_model(input_tensor).detach().numpy()[0][0]))
        reward = int(np.round(self.reward_model(input_tensor).detach().numpy()))
        
        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self._agent_location = new_pos

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)        

        return self._agent_location, reward, terminated, False, {}

    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  
        #print(self._agent_location[0], self._agent_location[1])
        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

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


class CSVGeneratorEnv(gym.Env):
    def __init__(self, maze, randomStart, render):
        self.maze = np.array(maze["maze"])  # Maze represented as a 2D numpy array
        #choose a random starting position between maze["starting_pos"]
        #convert the position in self. maze to a "S"
        if randomStart:
            #There is a predifined start position with an S in the maze. Before assigning a random start position I have to convert the older one to "."
            self.maze[self.maze == 'S'] = '.'
            start_pos = maze["starting_pos"][np.random.randint(0, len(maze["starting_pos"]))]
            self.maze[start_pos[0], start_pos[1]] = 'S'

        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)  # Starting position
        self.goal_pos = (np.concatenate(np.where(self.maze == 'G'))).astype(np.int32)  # Goal position
        self.num_rows, self.num_cols = self.maze.shape
        self.size = self.num_rows  # The size of the square grid

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.num_rows, self.num_cols]), dtype=np.int32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),#UP
            1: np.array([1, 0]),#DOWN
            2: np.array([0, -1]),#LEFT
            3: np.array([0, 1]),#RIGHT
        }

        if render:
            # Initialize Pygame
            pygame.init()
            if self.num_rows == 5:
                self.cell_size = 125
            elif self.num_rows == 14:
                self.cell_size = 25
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def _is_valid_position(self, pos):
        row, col = pos
        # If agent goes out of the grid or if the agent hits an obstacle
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols or self.maze[row, col] == '#':
            return False
        return True

    def reset(self, seed=None, options=None):
        self._agent_location = self.start_pos 
        self._target_location = self.goal_pos

        return self._agent_location, {}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        new_pos = np.clip(self._agent_location + direction, 0, self.size - 1)
        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self._agent_location = new_pos

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards

        return self._agent_location, reward, terminated, False, {}


    def test(self, input):
        position = np.array(input[0:2])
        action = input[2]
        # First we check if the position is not a wall
        if self.maze[position[0], position[1]] == '#':
            return "WALL"

        direction = self._action_to_direction[action]
        new_pos = np.clip(position + direction, 0, self.size - 1)
        
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return position


    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  
        #print(self._agent_location[0], self._agent_location[1])
        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

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
