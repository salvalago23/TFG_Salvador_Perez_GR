import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from collections import defaultdict
import torch

def create_dicts_Qlearning(agent):
    state_value = defaultdict(float)
    policy = defaultdict(int)

    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    return state_value, policy

def create_dicts_DQN(Qnet, env):
    n_rows, n_cols = env.unwrapped.num_rows, env.unwrapped.num_cols
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state_value = defaultdict(float)
    policy = defaultdict(int)
    Qnet.eval()

    for col in range(n_cols):
        for row in range(n_rows):
            state = torch.from_numpy(np.array([row,col])).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = Qnet(state)
            obs = (row, col)
            state_value[obs] = float(np.max(action_values.cpu().data.numpy()[0]))
            policy[obs] = int(np.argmax(action_values.cpu().data.numpy()[0]))

    return state_value, policy

def create_grids(state_value, policy, env):
    n_rows, n_cols = env.unwrapped.num_rows, env.unwrapped.num_cols

    # create a grid for the state values
    value_grid = np.zeros((n_rows, n_cols))
    for obs, value in state_value.items():
        value_grid[obs] = value

    # create a grid for the policy. In each state we should show the action that has the highest int(np.argmax(action_values))
    policy_grid = np.zeros((n_rows, n_cols))
    for obs, action in policy.items():
        policy_grid[obs] = action

    string_policy_grid = np.chararray((n_rows, n_cols), unicode=True)
    for i in range(n_rows):
        for j in range(n_cols):
            #Grid positions [0,2], [1,2], [3,2], [4,2] are walls, so we dont want to show an action for them, instead show an X. 
            #We also dont want to show an action for the goal, so we show a G in the grid position [0,4]
            if env.unwrapped.maze[i][j] == '#':
                string_policy_grid[i][j] = 'X'
            elif env.unwrapped.maze[i][j] == 'G':
                string_policy_grid[i][j] = 'G'
            elif policy_grid[i][j] == 0:
                string_policy_grid[i][j] = '↑'
            elif policy_grid[i][j] == 1:
                string_policy_grid[i][j] = '↓'
            elif policy_grid[i][j] == 2:
                string_policy_grid[i][j] = '←'
            elif policy_grid[i][j] == 3:
                string_policy_grid[i][j] = '→'
            else:
                string_policy_grid[i][j] = 'X'
    #print(string_policy_grid)

    return value_grid, policy_grid, string_policy_grid


"""def plot_string_policy(string_policy_grid):
    map_data = np.array(string_policy_grid)
    
    colors = ['red', 'black', 'lightgreen', 'green', 'gray', 'blue']
    cmap = ListedColormap(colors)

    # Create an array of integers based on the character array
    numeric_map = np.zeros(map_data.shape, dtype=int)
    for i, char in enumerate(['↓', 'X', '→', 'G', '↑', '←']):
        numeric_map[map_data == char] = i

    # Create the plot using imshow
    fig, ax = plt.subplots()
    im = ax.imshow(numeric_map, cmap=cmap, norm=plt.Normalize(0, len(colors)))

    # Add character annotations
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            char = map_data[i, j]
            ax.text(j, i, char, ha='center', va='center', color='black', fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="gray", edgecolor="black", label="Up"),
        Patch(facecolor="red", edgecolor="black", label="Down"),
        Patch(facecolor="blue", edgecolor="black", label="Left"),
        Patch(facecolor="lightgreen", edgecolor="black", label="Right"),
        Patch(facecolor="green", edgecolor="black", label="Goal"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig"""

def plot_trajectory(string_policy_grid, start_pos):
    map_data = np.array(string_policy_grid)
    
    colors = ['red', 'black', 'lightgreen', 'green', 'gray', 'blue']
    cmap = ListedColormap(colors)

    # Create an array of integers based on the character array
    numeric_map = np.zeros(map_data.shape, dtype=int)
    for i, char in enumerate(['↓', 'X', '→', 'G', '↑', '←']):
        numeric_map[map_data == char] = i

    # Create the plot using imshow
    fig, ax = plt.subplots()
    im = ax.imshow(numeric_map, cmap=cmap, norm=plt.Normalize(0, len(colors)))

        # Add character annotations
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            char = map_data[i, j]
            ax.text(j, i, char, ha='center', va='center', color='black', fontsize=12)

    # Define possible movements (up, down, left, right)
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Initialize the trajectory with the starting position
    trajectory = [start_pos.tolist()]

    current_pos = start_pos
    while True:
        y, x = current_pos
        char = map_data[y, x]

        # Check if the character indicates a goal, wall, or outside the grid
        if char == "G" or char == "X" or not (0 <= y < map_data.shape[0] and 0 <= x < map_data.shape[1]):
            break

        # Determine the next position based on the policy
        if char == "↓":
            next_pos = np.array([y + 1, x])
        elif char == "↑":
            next_pos = np.array([y - 1, x])
        elif char == "→":
            next_pos = np.array([y, x + 1])
        elif char == "←":
            next_pos = np.array([y, x - 1])

        # Append the next position to the trajectory
        trajectory.append(next_pos.tolist())

        # Move to the next position
        current_pos = next_pos

    # Convert the trajectory points into lists of x and y coordinates
    trajectory = np.array(trajectory)
    trajectory_x, trajectory_y = trajectory[:, 0], trajectory[:, 1]

    # Plot the trajectory on top of the grid
    ax.plot(trajectory_y, trajectory_x, marker='o', color='purple', markersize=2, linestyle='-')

    # add a legend
    legend_elements = [
        Patch(facecolor="gray", edgecolor="black", label="Up"),
        Patch(facecolor="red", edgecolor="black", label="Down"),
        Patch(facecolor="blue", edgecolor="black", label="Left"),
        Patch(facecolor="lightgreen", edgecolor="black", label="Right"),
        Patch(facecolor="green", edgecolor="black", label="Goal"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig