# Proyecto de RL Offline basado en modelo con gestión de incertidumbre by Salvador Pérez Lago

Todo el código necesario para replicar los experimentos de este TFG están incluídos en las carpetas ./src/ y ./src2/, que contienen dos flujos de trabajo totalmente independientes. 

Para instalar las librerías necesarias basta con instalar el contenido del 'requirements.txt' con el comando "pip install -r requirements.txt" desde la terminal. Es recomendable crear un entorno virtual de 0 para realizar la instalación evitando problemas de incompatibilidad entre otras versiones de las librerías.

A continuación se describirá brevemente el contenido de cada una de las carpetas que componen el proyecto:

## _discarded
Contiene pruebas fallidas en las que se intentaron usar librerías de RL previamente existentes, además de una primera versión del proyecto que estaba implementando en TensorFlow, antes de decidir migrar todo a PyTorch.


## agentsClasses
Aquí están definidas las clases de los agentes DQN y DDQN (en DQNClass.py) y los Q-Learning (QLearningClass.py). También están las versiones de esas clases pero para interactuar con el entorno offline (OfflineDQNClass.py y OfflineQLearningClass.py).


## data
### csv
Contiene los dataset de las transiciones (generados con el primer entorno en ./src/1_CSVGeneration.ipynb) de cada uno de los dos mapas que existen, antes (history5x5.csv y history14x14.csv) y después (new5x5.csv y new14x14.csv) de eliminar aquellas transiciones que se repitan más de un número X de veces.

### EnvNNModels
Con los dataset de los csv se generan 2 modelos, un modelo de la dinámica del entorno y otro de la función de recompensa, para cada uno de los 2 mapas. En total son 4 redes PyTorch que se guardan en esta carpeta.

### json
Aquí se guardan en formato json la información de los agentes QLearning, DQN y DDQN de los 2 mapas, entrenados en ./src/4_1_QLearning.ipynb, ./src/4_2_QLearningV2.ipynb, ./src/5_1_DQNTrain.ipynb y ./src/5_2_DQNTrainV2.ipynb

Se guardan los siguientes datos de cada agente: ID, Episodios de entrenamiento, Máx. número de pasos por episodio, Algoritmo usado (QL, DQN o DDQN), Tamaño del grid (5x5 o 14x14), Coordenadas de la casilla inicial, Política entrenada, Tabla de valores de cada estado

### OfflineEnsembles
Carpetas que contienen los ensembles de modelos para la predicción de la incertidumbre en los 2 mapas. Las carpetas "csv" dentro de las carpetas que contienen los modelos, guardan ficheros en formato csv con la información de la evolución de la pérdida de validación y de entrenamiento durante la fase de entrenamiento de los modelos.

### txt
ficheros txt para debuguear más fácilmente los resultados de las predicciones de los ensembles online.


## envs



## img
Contiene imágenes resultantes de algunos de los experimentos (gráficas de entrenamientos de agentes, modelos de entorno, políticas de comportamiento de agentes...)


## src



### src2



### utilities
Programas que contienen funciones auxiliares para el resto:
-jsonRW.py: contiene funciones que permiten acceder y visualizar, así como editar la información contenidas en los archivos json de ./data/json/ desde el programa ./src/6_JsonManagement.ipynb

-plots.py: funciones para plotear las políticas y trayectorias de los agentes entrenados en los programas ./src/4_1_QLearning.ipynb, ./src/4_2_QLearningV2.ipynb, ./src/5_1_DQNTrain.ipynb y ./src/5_2_DQNTrainV2.ipynb

-transitionDatasetGeneration.py: a pesar de su nombre, además de contener funciones para generar los dataset para crear los modelos del entorno offline, también incluyen todo tipo de funciones auxiliares usadas en todos los programas de ./src2, como pueden ser la clase para el entrenamiento de los ensembles de modelos o las funciones que permiten visualizar el mapa completo con la incertidumbre de cada estado.