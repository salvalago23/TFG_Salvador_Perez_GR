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
        else:
            raise ValueError("Invalid character in the policy grid")

        # Append the next position to the trajectory
        trajectory.append(next_pos.tolist())

        # Move to the next position
        current_pos = next_pos

    # Convert the trajectory points into lists of x and y coordinates
    trajectory = np.array(trajectory)
    trajectory_x, trajectory_y = trajectory[:, 0], trajectory[:, 1]

    # Plot the trajectory on top of the grid
    ax.plot(trajectory_y, trajectory_x, marker='o', color='purple', markersize=8, linestyle='-')

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