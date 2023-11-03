import numpy as np
import json

from utilities.plots import plot_trajectory
import matplotlib.pyplot as plt

def get_json_file_path(algorithm, shape):
    
    if algorithm == "Q-Learning":
        json_file_path = "../data/agent_models/json/agents_data_QLearning"
    elif algorithm == "DQN":
        json_file_path = "../data/agent_models/json/agents_data_DQN"
    elif algorithm == "DDQN":
        json_file_path = "../data/agent_models/json/agents_data_DDQN"

    if shape == "5x5":
        json_file_path += "_5x5.json"
    elif shape == "14x14":
        json_file_path += "_14x14.json"

    return json_file_path


def writeJSON(algorithm, episodes, steps, shape, start_pos, value_grid, policy_grid, string_policy_grid):

    json_file_path = get_json_file_path(algorithm, shape)

    starting_position_list = start_pos.tolist()
    value_grid_list = value_grid.tolist()
    policy_grid_list = policy_grid.tolist()
    string_policy_grid_list = string_policy_grid.tolist()

    # Load and parse the existing JSON data if the file already exists
    existing_data = {}
    try:
        with open(json_file_path, "r") as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        pass  # File doesn't exist yet, so initialize with an empty list

    # Determine the next unique ID
    if "agents_data" in existing_data:
        next_id = len(existing_data["agents_data"]) + 1
    else:
        next_id = 1

    new_agent_data = {
        "id": next_id,
        "episodes": episodes,
        "max_steps": steps,
        "algorithm": algorithm,
        "shape": shape,
        "starting_position": starting_position_list,
        "string_policy_grid": string_policy_grid_list,
        "value_grid": value_grid_list,
        "policy_grid": policy_grid_list
    }

    # Create a list to store multiple entries (if it doesn't already exist)
    if "agents_data" not in existing_data:
        existing_data["agents_data"] = []

    # Append the new agent data to the list inside the dictionary
    existing_data["agents_data"].append(new_agent_data)

    # Save the updated data back to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

def readJSON(algorithm, shape):

    json_file_path = get_json_file_path(algorithm, shape)

    # Load and parse the JSON data
    with open(json_file_path, "r") as json_file:
        agent_data = json.load(json_file)

    # Access the data for each agent entry
    for entry in agent_data.get("agents_data", []):
        id = entry["id"]
        algorithm = entry["algorithm"]
        shape = entry["shape"]
        episodes = entry["episodes"]
        max_steps = entry["max_steps"]
        starting_position = np.array(entry["starting_position"])
        string_policy_grid = np.array(entry["string_policy_grid"])
        value_grid = np.array(entry["value_grid"])
        policy_grid = np.array(entry["policy_grid"])

        #print(f'Agent {id} info --> Algorithm: {algorithm}   Shape: {shape}   Episodes: {episodes}   Max steps: {max_steps}')

        fig = plot_trajectory(string_policy_grid, starting_position)
        fig.suptitle(f'Agent {id} - {algorithm}  Start pos: {starting_position}  Episodes: {episodes}  Max steps: {max_steps}')
        plt.show()

def delete_agent_data_by_id(algorithm, shape, agent_id):

    json_file_path = get_json_file_path(algorithm, shape)

    try:
        # Load and parse the existing JSON data
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        if "agents_data" in data:
            # Find and remove the entry with the specified ID
            new_agents_data = [entry for entry in data["agents_data"] if entry["id"] != agent_id]

            # Update the IDs for the remaining entries
            for idx, entry in enumerate(new_agents_data):
                entry["id"] = idx + 1

            # Update the "agents_data" list
            data["agents_data"] = new_agents_data

            # Save the updated data back to the JSON file
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
                
            print(f"Agent data with ID {agent_id} has been successfully deleted.")
        else:
            print("No agent data found in the JSON file.")
    
    except FileNotFoundError:
        print(f"JSON file '{json_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")