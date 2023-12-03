import json
import numpy as np

from utilities.jsonRW import get_json_file_path

def trajectorize(start_pos, map_data):
    SASexperience = []
    current_pos = start_pos

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

        SASexperience.append([current_pos, action, next_pos])

        # Move to the next position
        current_pos = next_pos

    return SASexperience

def process_jsons(algorithms, shape):  #Ahora quiero obtener las trayectorias
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

            sy_values = []
            sx_values = []
            a_values = []
            sy1_values = []
            sx1_values = []

            for sas in trajectorize(starting_position, string_policy_grid):
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