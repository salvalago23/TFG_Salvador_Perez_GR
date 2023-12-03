import gymnasium as gym
from envs.GridMaps import maze5x5, maze14x14

mazes = {"5x5": maze5x5, "14x14": maze14x14}

def createCSVEnv(shape, render=False):
    # Register the environment
    gym.register(id='grid-v0', entry_point='envs.EnvCSV:CSVGeneratorEnv')

    if shape == "5x5":
        maze = mazes["5x5"]
    elif shape == "14x14":
        maze = mazes["14x14"]

    # Create the environment
    env = gym.make('grid-v0',maze=maze, render=render)#, max_episode_steps=500)

    return env

def createNNEnv(shape, max_steps=500, render=False, id=0):
    # Register the environment
    id = 'gridNN-v' + str(id)

    gym.register(id=id, entry_point='envs.EnvNN:NNGridWorldEnv')

    if shape == "5x5":
        maze = mazes["5x5"]
        grid_model_path = '../data/grid&reward_models/modelo_entorno5x5.pt'
        reward_model_path = '../data/grid&reward_models/modelo_reward5x5.pt'
    elif shape == "14x14":
        maze = mazes["14x14"]
        grid_model_path = '../data/grid&reward_models/modelo_entorno14x14.pt'
        reward_model_path = '../data/grid&reward_models/modelo_reward14x14.pt'

    # Create the environment
    env = gym.make(id, maze=maze, grid_model_path=grid_model_path, reward_model_path=reward_model_path, render=render, max_steps_per_episode=max_steps)

    return env

def createOfflineEnv(shape, n_models, reward, max_steps=500, render=False, id=0):
    # Register the environment
    id = 'gridOffline-v' + str(id)

    gym.register(id=id, entry_point='envs.EnvOffline:OfflineGridWorldEnv')

    if shape == "5x5":
        maze = mazes["5x5"]
    elif shape == "14x14":
        maze = mazes["14x14"]

    # Create the environment
    env = gym.make(id, maze=maze, shape=shape, n_models=n_models, reward=reward, render=render, max_steps_per_episode=max_steps)

    return env