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
    def __init__(self, maze, grid_model_path, reward_model_path, render, max_steps_per_episode):
        self.maze = np.array(maze["maze"])  # Maze represented as a 2D numpy array
        self.starting_positions = maze["starting_pos"] #list of possible starting positions
        
        self.max_steps_per_episode = max_steps_per_episode
        self.step_count = 0

        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)  # Starting position
        self.goal_pos = (np.concatenate(np.where(self.maze == 'G'))).astype(np.int32)  # Goal position
        self.num_rows, self.num_cols = self.maze.shape

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.num_rows, self.num_cols]), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Load models
        #print("Loading models...")
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
        #print("Models loaded")

        if render:
        # Initialize Pygame
            pygame.init()
            if self.num_rows == 5:
                self.cell_size = 125
            elif self.num_rows == 14:
                self.cell_size = 25
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def randomize_start_pos(self):
        # Aqui realmente con guardar las coordenadas en una variable ya valdria, pero pinto la "S" en el mapa para poder visualizarlo más tarde
        # There is a predifined start position with an S in the maze. Before assigning a random start position I have to convert the older one to "."
        
        self.maze[self.maze == 'S'] = '.'
        start_pos = self.starting_positions[np.random.randint(0, len(self.starting_positions))]
        self.maze[start_pos[0], start_pos[1]] = 'S'
        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)

    def _is_valid_position(self, pos):
        row, col = pos
        # If agent goes out of the grid or if the agent hits an obstacle
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols or self.maze[row, col] == '#':
            return False
        return True

    def reset(self, seed=None, options=None):
        self._agent_location = self.start_pos 
        self._target_location = self.goal_pos
        self.step_count = 0

        return self._agent_location, {}
    
    def step(self, action):
        self.step_count += 1

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
        
        # An episode is truncated if the agent has reached the maximum number of steps
        #do it with ternal operator
        truncated = True if self.step_count == self.max_steps_per_episode else False

        return self._agent_location, reward, terminated, truncated, {}

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
    def __init__(self, maze, render):
        self.maze = np.array(maze["maze"])  # Maze represented as a 2D numpy array
        self.starting_positions = maze["starting_pos"] #list of possible starting positions

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

    def randomize_start_pos(self):
        # Aqui realmente con guardar las coordenadas en una variable ya valdria, pero pinto la "S" en el mapa para poder visualizarlo más tarde
        # There is a predifined start position with an S in the maze. Before assigning a random start position I have to convert the older one to "."
        
        self.maze[self.maze == 'S'] = '.'
        start_pos = self.starting_positions[np.random.randint(0, len(self.starting_positions))]
        self.maze[start_pos[0], start_pos[1]] = 'S'
        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)

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




class OfflineGridWorldEnv(gym.Env):
    def __init__(self, maze, shape, n_models, reward, render, max_steps_per_episode):
        self.maze = np.array(maze["maze"])  # Maze represented as a 2D numpy array
        self.starting_positions = maze["starting_pos"] #list of possible starting positions
        
        self.max_steps_per_episode = max_steps_per_episode
        self.step_count = 0

        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)  # Starting position
        self.goal_pos = (np.concatenate(np.where(self.maze == 'G'))).astype(np.int32)  # Goal position
        
        self.num_rows, self.num_cols = self.maze.shape
        self.shape = shape

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.num_rows, self.num_cols]), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        self.reward = reward

        self.rewardHistory = []

        # Load models
        print("Loading models...")
        self.grid_models = []

        for i in range(n_models):
            if self.shape == "5x5":
                model = NeuralNetwork(3, 2)
            elif self.shape == "14x14":
                model = NeuralNetwork(3, 2, 128, 64)

            model.load_state_dict(torch.load("../data/offline_models/{}_{}.pt".format(self.shape, i)))
            model.eval()

            self.grid_models.append(model)
        
        print("Models loaded")




        if render:
        # Initialize Pygame
            pygame.init()
            if self.num_rows == 5:
                self.cell_size = 125
            elif self.num_rows == 14:
                self.cell_size = 25
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def randomize_start_pos(self):
        # Aqui realmente con guardar las coordenadas en una variable ya valdria, pero pinto la "S" en el mapa para poder visualizarlo más tarde
        # There is a predifined start position with an S in the maze. Before assigning a random start position I have to convert the older one to "."
        
        self.maze[self.maze == 'S'] = '.'
        start_pos = self.starting_positions[np.random.randint(0, len(self.starting_positions))]
        self.maze[start_pos[0], start_pos[1]] = 'S'
        self.start_pos = (np.concatenate(np.where(self.maze == 'S'))).astype(np.int32)

        print("Start position is: {}".format(self.start_pos))

    def _is_valid_position(self, pos):
        row, col = pos
        # If agent goes out of the grid or if the agent hits an obstacle
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols or self.maze[row, col] == '#':
            return False
        return True

    def reset(self, seed=None, options=None):
        self._agent_location = self.start_pos 
        self._target_location = self.goal_pos
        self.step_count = 0

        return self._agent_location, {}
    

    def step(self, action):
        self.step_count += 1

        #No me queda muy claro pq tiene que ser un [[[]]] y si es necesario, pero si no tiene esta forma el modelo de torch protesta
        #pq podria ser mas lento
        model_input = np.array([np.column_stack(np.array([float(self._agent_location[0]), float(self._agent_location[1]), float(action)]))])
        input_tensor = torch.tensor(model_input, dtype=torch.float32)

        posibilidades = []

        for model in self.grid_models:
            #resultado = model(input_tensor)
            #resultado = resultado.detach().numpy()

            posibilidades.append([np.int32(np.round(model(input_tensor).detach().numpy()[0][0][0])), np.int32(np.round(model(input_tensor).detach().numpy()[0][0][1]))])
         
        probability_dict = {}
        for p in posibilidades:
            if not str(p) in probability_dict:
                probability_dict[str(p)] = str(posibilidades.count(p)/len(posibilidades))
        
        #sort the dictionary by value from highest to lowest
        probability_dict = {k: v for k, v in sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)}
        
        highest_probability_state = list(probability_dict.keys())[0]
        highest_probability_value = float(list(probability_dict.values())[0])

        highest_probability_state = highest_probability_state.replace("[", "").replace("]", "").split(", ")

        #create a numpy array with the highest probability state
        new_pos = np.array([np.column_stack(np.array([np.int32(highest_probability_state[0]), np.int32(highest_probability_state[1])]))])[0][0]


        old_pos = self._agent_location

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self._agent_location = new_pos

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        if self.reward == 1:
            reward = 1 if terminated else 0#highest_probability_value  # Binary sparse rewards
        elif self.reward == 1000:
            if highest_probability_value < 0.5:
                reward = -1000
                self.rewardHistory.append([old_pos, action, new_pos])
            else:
                reward = 1 if terminated else 0
        #print("New position is: {}".format(self._agent_location))
        #print("Reward is: {}".format(reward))
        #input("Press Enter to continue...")

        # An episode is truncated if the agent has reached the maximum number of steps
        #do it with ternal operator
        truncated = True if self.step_count == self.max_steps_per_episode else False

        return self._agent_location, reward, terminated, truncated, {}




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