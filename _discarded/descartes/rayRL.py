import ray
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQN
from NNmodeledEnv import NNGridWorldEnv  # Importa la clase desde el archivo personalizado
import gymnasium as gym

def create_custom_grid_env(config):
    return NNGridWorldEnv(
        maze=config["maze"],
        grid_model_path=config["grid_model_path"],
        reward_model_path=config["reward_model_path"]
    )

# Define el laberinto
maze = [
    ['.', '.', '#', '.', 'G'],
    ['.', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['S', '.', '#', '.', '.'],
]

grid_model_path = '../data/models/modelo_entorno.h5'
reward_model_path = '../data/models/modelo_reward.h5'

ray.init()

config = {
    "env": NNGridWorldEnv,  # Utiliza la clase directamente
    "num_workers": 1,
    "framework": "tf",
    "env_config": {
        "maze": maze,
        "grid_model_path": grid_model_path,
        "reward_model_path": reward_model_path
    }
}

# Entrenar el agente DQN
trainer = DQN(config=config)
results = tune.run(trainer, stop={"training_iteration": 100})

# Guardar el modelo entrenado
checkpoint_path = trainer.save()
print("Modelo guardado en", checkpoint_path)

ray.shutdown()
