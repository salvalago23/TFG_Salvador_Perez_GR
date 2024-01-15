# Proyecto de RL Offline basado en modelo con gestión de incertidumbre by Salvador Pérez Lago

[id]: https://github.com/salvalago23/proyectoNN_salva  "Github"
[id2]: https://nubeusc-my.sharepoint.com/:f:/g/personal/salvador_perez_rai_usc_es/EmeTyvYVMS9OuyLWuvRf_eEBgIDsb3z1ewkRNZnxxQTSCw?e=OHTvFn  "OneDrive"

Todo el código y los modelos pueden ser descargados desde este repositorio de [Github][id] o bien desde [OneDrive][id2].
This is  reference-style link.

Los programas principales replicar los experimentos de este TFG están incluídos en las carpetas ./src/ y ./src2/, que contienen dos flujos de trabajo totalmente independientes. 

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
Ficheros txt para debuguear más fácilmente los resultados de las predicciones de los ensembles online.


## envs
Estos códigos contienen el entorno original SimpleGrid (EnvCSV.py), el que usa redes neuronales para modelar la dinámica del entorno y la función de recompensa (EnvNN.py) y el que usa como modelo el ensemble offline de redes neuronales (EnvOffline.py).

En el fichero GridMaps.py es donde están definidos como arrays de arrays de caracteres los mapas de los grids. Se pueden añadir o quitar fácilmente casillas no demostradas (aquellas que contienen un caracter 'x') para realizar experimentos, cambiar la configuración de los muros o la posición de las casillas de "Inicio" o "Meta". También existe la posibilidad de crear otros nuevos mapas con distintas dimensiones.

Por último, hay un archivo que permite contiene las funciones que permiten instanciar los entornos usando los mapas (CreateEnvs.py) y otro que contiene diferentes las diferentes arquitecturas de redes neuronales usadas a lo largo del proyecto (NeuralNetworks.py).

## img
Contiene imágenes resultantes de algunos de los experimentos (gráficas de entrenamientos de agentes, modelos de entorno, políticas de comportamiento de agentes...)


## src
Esta carpeta contiene los notebooks con los experimentos correspondientes con el primer enfoque del proyecto. Están ordenados alfabéticamente siguiendo el mismo orden secuencial con el que fueron creados y que debería seguir el proceso en caso de querer replicar los experimentos. A continuación se explicará el contenido de cada notebook:

### 1_CSVGeneration
En este archivo, haciendo uso del entorno de ./envs/EnvCSV.py se generan los dataset en formato csv con los que se entrenarán las redes neuronales para modelar la dinámica y la función de recompensa del entorno.

### 2_NNModelsTrain
Aquí se entrenan las redes neuronales haciendo uso de los csv creados por el anterior notebook, para poder crear el entorno de ./envs/EnvNN.py

### 3_NNRun
Aquí se puede comprobar que el entorno implementado con las redes neuronales funciona correctamente.

### 4_1QLearning y 4_2_QLearningV2
Programas para entrenar agentes de QLearning que sean capaces de resolver el problema de ir desde el Inicio a la Meta usando el entorno ./envs/EnvNN.py, de forma que se obtengan trayectorias que puedan usarse como "demostradores". La estructura del 4_2 es algo distinta a la de 4_1, para permitir entrenar muchos más agentes de cada vez y con distintas características de número de episodios y máximo número de pasos por episodio. Los resultados se guardan en los ficheros json de ./data/json/

### 5_1_DQNTrain y 5_2_DQNTrainV2
Lo mismo que los notebooks del punto anterior, pero para agentes DQN y DDQN.

### 6_JsonManagement
Este programa permite visualizar y modificar la información de los json creados en los notebooks previos.

### 7_OfflinesFamiliesModels
En este notebook me di cuenta de que el enfoque que estaba usando no me iba a dar resultados congruentes, por lo que decidí abandonarlo y pasar al enfoque de ./src2/
Como en cada mapa existe una serie de casillas de inicio definidas, entre las que se escoge una de ellas de forma aleatoria cada vez que se instancia el entorno, la idea era juntar los demostradores en grupos según su casilla de inicio. Así, con las trayectorias de cada grupo se generaría un modelo, y al juntar todos se podría obtener un modelo global que permitiese a otro agente resolver de cero el problema de llegar del Inicio a la Meta.


## src2
Esta carpeta se corresponde con el enfoque final del proyecto y, como la anterior, sus notebooks también están ordenados secuencialmente. A continuación, una breve explicación de cada uno:

### 1_OfflineEnsemblesGenerator
En este notebook, a partir de los mapas definidos en ./envs/GridMaps como arrays de arrays de caracteres, se generan todas las transiciones posibles. La clave es que podemos elegir casillas, en las que pondremos el caracter 'x', para que sean casillas no demostradas. Con esas transiciones, el dataset "D", se crean los modelos offline, un ensemble de modelos (inicializados con distinto seed, i.e. distinto vector de parámetros o pesos iniciales). Haciendo uso de estos ensembles en EnvOffline.py obtenemos un entorno en el que seremos capaces de discernir la incertidumbre entre los distintos estados, de manera que, aquellos que no habían sido incluídos en el dataset (los no demostrados 'x') tengan una incertidumbre mucho mayor que los que si. Así, penalizandolos cuando lleguen a estos estados, podremos entrenar agentes que consigan resolver el entorno mediante rutas que eviten pasar por ninguno de los estados no demostrados.

### 2_Experiments
Aquí se realizan una serie de pruebas descritas en la memoria para probar a aplicar distintas técnicas (basadas en el bagging) sobre el dataset generado anteriormente, y ver como afectaría a la eficacia de los ensembles generados con cada una.

### 3_Visualization
Con este notebook se pueden ver las gráficas de las pérdidas, tanto de validación como de entrenamiento, de los modelos que componen los ensembles durante su fase de entrenamiento. También se pueden ver los resultados de incertidumbre que nos dan los ensembles para cada casilla (y, si se quiere, para cada acción en cada una de ellas por separado) del grid gráficamente.

### 4_OfflineQLearning y 5_OfflineDQNTrain
Haciendo uso de los ensembles para modelar el entorno desde EnvOffline.py, podemos conseguir que agentes resuelvan el entorno ¡¡evitando las casillas no demostradas!!, tanto con QLearning como con DQN o DDQN, para los dos mapas.

## utilities
Programas que contienen funciones auxiliares para el resto:
-jsonRW.py: contiene funciones que permiten acceder y visualizar, así como editar la información contenidas en los archivos json de ./data/json/ desde el programa ./src/6_JsonManagement.ipynb

-plots.py: funciones para plotear las políticas y trayectorias de los agentes entrenados en los programas ./src/4_1_QLearning.ipynb, ./src/4_2_QLearningV2.ipynb, ./src/5_1_DQNTrain.ipynb y ./src/5_2_DQNTrainV2.ipynb

-transitionDatasetGeneration.py: a pesar de su nombre, además de contener funciones para generar los dataset para crear los modelos del entorno offline, también incluyen todo tipo de funciones auxiliares usadas en todos los programas de ./src2, como pueden ser la clase para el entrenamiento de los ensembles de modelos o las funciones que permiten visualizar el mapa completo con la incertidumbre de cada estado.