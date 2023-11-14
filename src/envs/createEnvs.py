import gymnasium as gym

#Maze configs
maze5x5 = { "starting_pos": [[0,0], [2,0], [4,0]],
             "maze":[
                    ['.', '.', '#', '.', 'G'],
                    ['.', '.', '#', '.', '.'],
                    ['.', '.', '.', '.', '.'],
                    ['.', '.', '#', '.', '.'],
                    ['S', '.', '#', '.', '.'],
                    ]
        }

maze14x14 = { "starting_pos": [[0,0], [5,0], [7,0], [13,0], [13,5], [13,8], [13,11]],
             "maze":[
                    ['.', '#', '.', '.', '.', '.', '.', '.', '#', '#', '#', '#', '.', 'G'],
                    ['.', '#', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '#', '.', '.', '#', '.', '.', '#', '.', '.'],
                    ['.', '#', '#', '#', '.', '#', '.', '.', '.', '.', '.', '#', '.', '.'],
                    ['.', '.', '#', '.', '.', '#', '#', '#', '#', '#', '.', '#', '#', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '.'],
                    ['#', '#', '#', '#', '#', '.', '#', '#', '.', '.', '.', '.', '#', '.'],
                    ['.', '.', '.', '.', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '#', '.', '.', '#', '.', '#', '.', '#', '#', '#'],
                    ['#', '.', '#', '.', '#', '.', '#', '#', '.', '#', '.', '.', '.', '.'],
                    ['.', '.', '#', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '#', '#', '.', '#', '.', '.', '.', '.', '#', '.', '#', '#', '.'],
                    ['.', '.', '.', '.', '#', '#', '.', '#', '.', '#', '.', '.', '#', '.'],
                    ['S', '.', '#', '.', '#', '.', '.', '#', '.', '#', '.', '.', '#', '.'],
                    ]
            }

mazes = {"5x5": maze5x5, "14x14": maze14x14}

def createCSVEnv(shape, render=False):
    # Register the environment
    gym.register(id='grid-v0', entry_point='envs.environments:CSVGeneratorEnv')

    if shape == "5x5":
        maze = mazes["5x5"]
    elif shape == "14x14":
        maze = mazes["14x14"]

    # Create the environment
    env = gym.make('grid-v0',maze=maze, render=render)#, max_episode_steps=500)

    return env

def createNNEnv(shape, render=False, id=0):
    # Register the environment
    id = 'gridNN-v' + str(id)

    gym.register(id=id, entry_point='envs.environments:NNGridWorldEnv')

    if shape == "5x5":
        maze = mazes["5x5"]
        grid_model_path = '../data/grid&reward_models/modelo_entorno5x5.pt'
        reward_model_path = '../data/grid&reward_models/modelo_reward5x5.pt'
    elif shape == "14x14":
        maze = mazes["14x14"]
        grid_model_path = '../data/grid&reward_models/modelo_entorno14x14.pt'
        reward_model_path = '../data/grid&reward_models/modelo_reward14x14.pt'

    # Create the environment
    env = gym.make(id, maze=maze, grid_model_path=grid_model_path, reward_model_path=reward_model_path, render=render)#, disable_env_checker=disable_env_checker, max_episode_steps=500)

    return env