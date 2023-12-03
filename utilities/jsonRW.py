import numpy as np
import json

from utilities.plots import plot_trajectory
import matplotlib.pyplot as plt

def get_json_file_path(algorithm, shape):
    json_file_path = "../data/json/agents_data_"

    if algorithm == "Q-Learning":
        json_file_path += "QLearning"
    elif algorithm == "DQN":
        json_file_path += "DQN"
    elif algorithm == "DDQN":
        json_file_path += "DDQN"

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

        fig = plot_trajectory(string_policy_grid, starting_position, id, return_fig=True)
        fig.suptitle(f'Agent {id} - {algorithm}  Start pos: {starting_position}  Episodes: {episodes}  Max steps: {max_steps}')
        plt.show()

def readJSONById(algorithm, shape, agent_id):
    json_file_path = get_json_file_path(algorithm, shape)

    # Load and parse the JSON data
    with open(json_file_path, "r") as json_file:
        agent_data = json.load(json_file)

    # Find the agent entry with the specified ID
    agent_entry = None
    for entry in agent_data.get("agents_data", []):
        if entry["id"] == agent_id:
            agent_entry = entry
            break

    if agent_entry is None:
        print(f"Error: Agent with ID {agent_id} not found in the JSON data.")
        return

    id = agent_entry["id"]
    algorithm = agent_entry["algorithm"]
    shape = agent_entry["shape"]
    episodes = agent_entry["episodes"]
    max_steps = agent_entry["max_steps"]
    starting_position = np.array(agent_entry["starting_position"])
    string_policy_grid = np.array(agent_entry["string_policy_grid"])
    value_grid = np.array(agent_entry["value_grid"])
    policy_grid = np.array(agent_entry["policy_grid"])

    # Print agent information
    print(f'Agent {id} info --> Algorithm: {algorithm}   Shape: {shape}   Episodes: {episodes}   Max steps: {max_steps}')

    # Plot the trajectory for the specified agent
    fig = plot_trajectory(string_policy_grid, starting_position, id, return_fig=True)
    fig.suptitle(f'Agent {id} - {algorithm}  Start pos: {starting_position}  Episodes: {episodes}  Max steps: {max_steps}')
    plt.show()


def display_agent_data(algorithm, shape, page_number, items_per_page=10):
    try:
        json_file_path = get_json_file_path(algorithm, shape)

        # Load and parse the existing JSON data
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        if "agents_data" in data:
            agent_data = data["agents_data"]
            total_entries = len(agent_data)

            start_idx = (page_number - 1) * items_per_page
            end_idx = min(page_number * items_per_page, total_entries)

            if start_idx < total_entries:
                print(f"Displaying entries {start_idx + 1} to {end_idx} (out of {total_entries}):")
                for entry in agent_data[start_idx:end_idx]:
                    starting_position = np.array(entry["starting_position"])
                    string_policy_grid = np.array(entry["string_policy_grid"])
                    id = entry["id"]
                    algorithm = entry.get("algorithm", "Unknown")
                    episodes = entry.get("episodes", "Unknown")
                    max_steps = entry.get("max_steps", "Unknown")

                    fig = plot_trajectory(string_policy_grid, starting_position, id, return_fig=True)
                    fig.suptitle(f'Agent {id} - {algorithm}  Start pos: {starting_position}  Episodes: {episodes}  Max steps: {max_steps}')
                    plt.show()
            else:
                print("No more entries to display.")

        else:
            print("No agent data found in the JSON file.")

    except FileNotFoundError:
        print(f"JSON file '{json_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


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


#OFFLINE MODELS
def traj_families_from_json(algorithms, shape):
    families_dict = {}

    for algorithm in algorithms:
        json_file_path = get_json_file_path(algorithm, shape)

        # Load and parse the JSON data
        with open(json_file_path, "r") as json_file:
            agent_data = json.load(json_file)
            
        # Access the data for each agent entry
        for entry in agent_data.get("agents_data", []):

            starting_position = entry["starting_position"]
            string_policy_grid = np.array(entry["string_policy_grid"])

            transitions = []
            current_pos = starting_position
            map_data = string_policy_grid

            while True:
                y, x = current_pos
                # Check if the current position is outside the map
                if not (0 <= y < map_data.shape[0] and 0 <= x < map_data.shape[1]):
                    break

                char = map_data[y, x]
                # Check if the character indicates the goal or a wall
                if char == "G" or char == "X":
                    break

                # Determine the next position based on the policy
                if char == "↑":
                    next_pos = [y - 1, x]
                    action = 0
                elif char == "↓":
                    next_pos = [y + 1, x]
                    action = 1
                elif char == "←":
                    next_pos = [y, x - 1]
                    action = 2
                elif char == "→":
                    next_pos = [y, x + 1]
                    action = 3

                transitions.append([current_pos, action, next_pos])

                # Move to the next position
                current_pos = next_pos

            sy_values = []
            sx_values = []
            a_values = []
            sy1_values = []
            sx1_values = []

            for sas in transitions:
                sy_values.append(float(sas[0][0]))
                sx_values.append(float(sas[0][1]))
                a_values.append(float(sas[1]))
                sy1_values.append(float(sas[2][0]))
                sx1_values.append(float(sas[2][1]))
                #print(sas[0][0], sas[0][1], sas[1], sas[2][0], sas[2][1])
            
            # Si la familia ya existe en el diccionario, simplemente agrega los valores
            if tuple(starting_position) in families_dict:
                families_dict[tuple(starting_position)]["sy_values"].extend(sy_values)
                families_dict[tuple(starting_position)]["sx_values"].extend(sx_values)
                families_dict[tuple(starting_position)]["a_values"].extend(a_values)
                families_dict[tuple(starting_position)]["sy1_values"].extend(sy1_values)
                families_dict[tuple(starting_position)]["sx1_values"].extend(sx1_values)
                families_dict[tuple(starting_position)]["length"] = families_dict[tuple(starting_position)]["length"] + len(sy_values)
                families_dict[tuple(starting_position)]["count"] = families_dict[tuple(starting_position)]["count"] + 1
            else:
                # Si la familia no existe, crea una nueva entrada en el diccionario
                families_dict[tuple(starting_position)] = {
                    "sy_values": sy_values,
                    "sx_values": sx_values,
                    "a_values": a_values,
                    "sy1_values": sy1_values,
                    "sx1_values": sx1_values,
                    "length": len(sy_values),
                    "count": 0
                }

    return families_dict