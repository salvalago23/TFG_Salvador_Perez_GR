import gymnasium as gym

#Maze config
maze = [
    ['.', '.', '#', '.', 'G'],
    ['.', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['S', '.', '#', '.', '.'],
]

def createCSVEnv():
    # Register the environment
    gym.register(id='grid-v0', entry_point='envs.environments:CSVGeneratorEnv')
    # Create the environment
    env = gym.make('grid-v0',maze=maze, max_episode_steps=500)

    return env

def createNNEnv():
    # Register the environment
    gym.register(id='gridNN-v0', entry_point='envs.environments:NNGridWorldEnv')

    grid_model_path = '../data/models/modelo_entorno.pt'
    reward_model_path = '../data/models/modelo_reward.pt'

    # Create the environment
    env = gym.make('gridNN-v0', maze=maze, grid_model_path=grid_model_path, reward_model_path=reward_model_path, max_episode_steps=500)

    return env