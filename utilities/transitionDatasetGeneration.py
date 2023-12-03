import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from envs.NeuralNetwork import NeuralNetwork

#Clase para simplificar la creación de los modelos a partir de las sublistas de transiciones
class ModelTrainer:
    def __init__(self, id, shape, T, n_epochs):
        self.id = id
        self.shape = shape

        self.T = T
        
        self.sy_values = np.array([], dtype = np.float32)
        self.sx_values = np.array([], dtype = np.float32)
        self.a_values = np.array([], dtype = np.float32)
        self.sy1_values = np.array([], dtype = np.float32)
        self.sx1_values = np.array([], dtype = np.float32)

        self.input_data = None
        self.target_data = None

        self.X_train_tensor = None
        self.X_test_tensor = None
        self.y_train_tensor = None
        self.y_test_tensor = None

        if shape == "5x5":
            self.model = NeuralNetwork(3, 2)
        elif shape == "14x14":
            self.model = NeuralNetwork(3, 2, 128, 64)
        
        self.n_epochs = n_epochs

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.losses = []

        self.generate_arrays()
    
    def generate_arrays(self):
        for t in self.T:
            self.sy_values = np.append(self.sy_values, t[0])
            self.sx_values = np.append(self.sx_values, t[1])
            self.a_values = np.append(self.a_values, t[2])
            self.sy1_values = np.append(self.sy1_values, t[3])
            self.sx1_values = np.append(self.sx1_values, t[4])

        self.input_data = np.column_stack((self.sy_values, self.sx_values, self.a_values))
        self.target_data = np.column_stack((self.sy1_values, self.sx1_values))

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.input_data, self.target_data, test_size=0.05)
        
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        for _ in range(self.n_epochs):
            # Forward pass
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
        
        print("Trained successfully!")
        print(f'Final loss: {self.losses[-1]}\n')

        #save the model with info of the shape and the id
        torch.save(self.model.state_dict(), f'../data/OfflineModels2/{self.shape}_{self.id}.pt')

    def test_loss(self):
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            test_loss = self.criterion(test_outputs, self.y_test_tensor)
            print(f'Test loss: {test_loss.item()}')

    def show_loss(self):
        plt.xlabel("# Epoch")
        plt.ylabel("Loss Magnitude")
        plt.plot(self.losses)
        plt.show()


#Esta funcion recibe un grid y devuelve todas las posibles transiciones que se pueden 
#hacer desde cada casilla del grid, para cada una de las 4 acciones posibles
def transition_generator(maze, with_walls = True):
    transiciones = []

    n_rows, n_cols = len(maze), len(maze[0])

    for y in range(n_rows):
        for x in range(n_cols):
            # Si la casilla actual es un muro o una casilla 'x' (forzada a no demostracion),
            # no se puede guardar la transición
            if maze[y][x] != '#' and maze[y][x] != 'x':
                for a in range(4):
                    if a == 0:  # Mover hacia el norte
                        y1, x1 = max(0, y - 1), x
                    elif a == 1:  # Mover hacia el sur
                        y1, x1 = min(n_rows - 1, y + 1), x
                    elif a == 2:  # Mover hacia el oeste
                        y1, x1 = y, max(0, x - 1)
                    elif a == 3:  # Mover hacia el este
                        y1, x1 = y, min(n_cols - 1, x + 1)

                    # Si la casilla siguiente es un muro y el booleano es válido,
                    # se debe guardar la transición de la casilla actual a sí misma
                    if maze[y1][x1] == '#':
                        if with_walls:
                            transiciones.append([y, x, a, y, x])
                    # Si la casilla siguiente es una casilla 'x' no vamos a guardar ninguna transición
                    elif maze[y1][x1] == 'x':
                        pass
                    # Si la siguiente no es ninguna de las anteriores, se guarda la transición
                    else:
                        transiciones.append([y, x, a, y1, x1])

    return transiciones


#FUNCIONES PARA CREAR SUBCONJUNTOS DE TRANSICIONES A PARTIR DE LA LISTA COMPLETA

#Esta funcion recibe una lista de transiciones, elimina un porcentaje de ellas y devuelve una sublista con las restantes
def transitions_families_generator_BASIC(transiciones_iniciales, porcentaje_retencion, n):
    n_transiciones_iniciales = len(transiciones_iniciales)
    m = int((1 - porcentaje_retencion) * n_transiciones_iniciales)

    nuevos_arrays = []

    for _ in range(n):
        # Crear una copia de las transiciones iniciales
        nuevas_transiciones = transiciones_iniciales.copy()

        # Eliminar aleatoriamente M transiciones
        transiciones_a_eliminar = random.sample(nuevas_transiciones, m)
        for t in transiciones_a_eliminar:
            nuevas_transiciones.remove(t)

        nuevos_arrays.append(nuevas_transiciones)

    return nuevos_arrays

# Misma función que la anterior pero con un porcentaje de retención fijo del 66%
# y que después genera el otro 33% de transiciones con repeticiones de las que ya tiene.
# Al final tendrá el mismo numero de transiciones que la lista original
def transitions_families_generator_WITH_REPETITION(transiciones_iniciales, n):
    n_transiciones_iniciales = len(transiciones_iniciales)
    n_borradas = int((1 - 0.66) * n_transiciones_iniciales)
    
    nuevos_arrays = []

    for _ in range(n):
        # Crear una copia de las transiciones iniciales
        nuevas_transiciones = transiciones_iniciales.copy()

        # Eliminar aleatoriamente 'n_retenidas' transiciones
        transiciones_a_eliminar = random.sample(nuevas_transiciones, n_borradas)
        for t in transiciones_a_eliminar:
            nuevas_transiciones.remove(t)

        # Repetir aleatoriamente 'n_borradas' transiciones
        transiciones_a_repetir = random.sample(nuevas_transiciones, n_borradas)
        nuevas_transiciones.extend(transiciones_a_repetir)

        nuevos_arrays.append(nuevas_transiciones)

    return nuevos_arrays

# Esta función es casi igual que la anterior, pero en vez de eliminarlas aleatoriamente, va seleccionando un subconjunto
# de transiciones de forma ordenada y ciclicamente (de la 0 a la N, de la N+1 a la 2N, ...). De esta manera se asegura que todas las transiciones se usan al menos una vez.
# Al final cada sublista se completa con repeticiones de las transiciones que ya tiene como antes.
def transitions_families_generator_FULLY_REPRESENTED(transiciones_iniciales, n):
    n_transiciones_iniciales = len(transiciones_iniciales)
    n_subset = int(0.66 * n_transiciones_iniciales)

    nuevos_arrays = []

    for i in range(n):

        # Crear una copia de las transiciones iniciales
        nuevas_transiciones = transiciones_iniciales.copy()
        
        start_index = (i * n_subset) % len(nuevas_transiciones)
        end_index = (start_index + n_subset) % len(nuevas_transiciones)

        if start_index < end_index:
            nuevas_transiciones = nuevas_transiciones[start_index:end_index]
        else:
            nuevas_transiciones = nuevas_transiciones[start_index:] + nuevas_transiciones[:end_index]        

        # Repetir aleatoriamente 'n_borradas' transiciones
        transiciones_a_repetir = random.sample(nuevas_transiciones, n_transiciones_iniciales - n_subset)
        nuevas_transiciones.extend(transiciones_a_repetir)

        nuevos_arrays.append(nuevas_transiciones)

    return nuevos_arrays

#Esta función muestra un conteo de cuantas veces aparece cada transición en las sublistas.
#De esta manera podemos asegurarnos de si las transiciones se han muestreado correctamente o no
def count_original_transitions(all_transitions, new_transitions):
    # Aplanar la lista de sublistas
    flattened_new_transitions = [item for sublist in new_transitions for item in sublist]

    # Inicializar un diccionario para contar las transiciones originales
    counts = defaultdict(int)

    # Contar la frecuencia de cada transición original
    for transition in flattened_new_transitions:
        counts[tuple(transition)] += 1

    index = 1
    # Imprimir los resultados
    for original_transition in all_transitions:
        count = counts.get(tuple(original_transition), 0)
        print(f"{index} Transición: {original_transition}, Apariciones: {count}")
        index += 1


#FUNCION PARA TESTEAR LOS MODELOS ENTRENADOS
#For a given state, predict the output for each action from the 9 models through polling.
#The possible states are sorted by their probability of being the next state, from highest to lowest.
def model_tester(state, models_arr):

    y = state[0]
    x = state[1]

    for i in range(4):
        posibilidades = []

        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32)
        
        print("Input: " + str([y, x, i]))

        for i in range(len(models_arr)):
            resultado = models_arr[i].model(test_input)
            resultado = resultado.detach().numpy()
            posibilidades.append([round(resultado[0][0]), round(resultado[0][1])])
        
        #Calculo el porcentaje de veces que aparece cada posibilidad
        probability_dict = {str(posibilidades.count(p)/len(posibilidades)*100) + "%": p for p in posibilidades}

        probability_dict = {}
        for p in posibilidades:
            if not str(p) in probability_dict:
                probability_dict[str(p)] = str(posibilidades.count(p)/len(posibilidades))

        #Ordeno el diccionario por las probabilidades de mayor a menor
        probability_dict = {k: v for k, v in sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)}
        
        highest_probability = list(probability_dict.keys())[0]
        #obtener el valor de la probabilidad del mas alto
        valor = float(list(probability_dict.values())[0])

        #convertirla de nuevo a lista
        highest_probability = highest_probability.replace("[", "").replace("]", "").split(", ")

        highest_probability = [highest_probability[0], highest_probability[1]]
        
        print("Predicciones: ")
        for p in probability_dict:
            print(p + ": ", round(float(probability_dict[p])*100, 3),"%")

        #print("Probabilidad " + str(p) + ": " + str(posibilidades.count(p)/len(posibilidades)*100) + "%")
        
        print("--------------------------------------------------\n")

#FUNCION PARA GENERAR UN .TXT CON TODAS LAS PREDICCIONES DEL ENSEMBLE DE MODELOS
#PARA TODO EL GRID
def model_tester_output_txt(grid, models_arr):
    with open('../data/txt/ModelsTestOutput.txt', 'w') as archivo:
        
        archivo.write("Grid: \n")   
        for i in range(len(grid)):
            archivo.write("\t\t" + str(grid[i]) + ",#" + str(i) + "\n")

        if len(grid) == 5:
            archivo.write("\t\t# 0    1    2    3    4\n\n")
        elif len(grid) == 14:
            archivo.write("\t\t# 0    1    2    3    4    5    6    7    8    9    10   11   12   13\n\n")

        archivo.write("Acciones:\n \t\t\t\t0\n \t\t\t\t↑\n \t\t\t 2 ← → 3\n \t\t\t\t↓\n \t\t\t\t1\n\n\n")

        for fila in grid:
            for x in range(5):
                if fila[x] == 'x':
                    y = grid.index(fila)
                    #print("y: ", y, "x: ", x)

                    archivo.write("Casilla: [" + str(y) + "," + str(x) + "]\n")

                    for i in range(4):
                        posibilidades = []

                        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32)
                        
                        #print("Input: " + str([y, x, i]))
                        archivo.write("\tAccion: " + str(i) + "\n")
                        for i in range(len(models_arr)):
                            resultado = models_arr[i].model(test_input)
                            resultado = resultado.detach().numpy()
                            posibilidades.append([round(resultado[0][0]), round(resultado[0][1])])
                        
                        #Calculo el porcentaje de veces que aparece cada posibilidad
                        probability_dict = {str(posibilidades.count(p)/len(posibilidades)*100) + "%": p for p in posibilidades}

                        probability_dict = {}
                        for p in posibilidades:
                            if not str(p) in probability_dict:
                                probability_dict[str(p)] = str(posibilidades.count(p)/len(posibilidades))

                        #Ordeno el diccionario por las probabilidades de mayor a menor
                        probability_dict = {k: v for k, v in sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)}
                        
                        highest_probability = list(probability_dict.keys())[0]
                        #obtener el valor de la probabilidad del mas alto
                        valor = float(list(probability_dict.values())[0])

                        #convertirla de nuevo a lista
                        highest_probability = highest_probability.replace("[", "").replace("]", "").split(", ")

                        highest_probability = [highest_probability[0], highest_probability[1]]
                        
                        archivo.write("\tPredicciones:\n")
                        for p in probability_dict:
                            archivo.write("\t\t" + p + ": "+ str(round(float(probability_dict[p])*100, 3))+"%\n")

                        #print("Probabilidad " + str(p) + ": " + str(posibilidades.count(p)/len(posibilidades)*100) + "%")
                        
                        archivo.write("\t---------------------\n")
                    archivo.write("-------------------------\n\n")


#FUNCION PARA MOSTRAR UNA GRÁFICA CON LAS PROBABILIDADES DE CADA ACCIÓN PARA CADA CASILLA DEL GRID
def probMapper(grid, n_models, lower_threshold = 35.0, higher_threshold = 60.0):
    print("Loading models...")
    grid_models = []

    if len(grid) == 5:
        shape = "5x5"
    elif len(grid) == 14:
        shape = "14x14"

    for i in range(n_models):
        if shape == "5x5":
            model = NeuralNetwork(3, 2)
        elif shape == "14x14":
            model = NeuralNetwork(3, 2, 128, 64)

        model.load_state_dict(torch.load("../data/OfflineModels2/{}_{}.pt".format(shape, i)))
        model.eval()

        grid_models.append(model)
    
    print("Models loaded")

    empty_grid = [['' for i in range(len(grid[0]))] for j in range(len(grid))]

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == "#":
                empty_grid[y][x] = "#" 
            else:
                for i in range(4):
                    posibilidades = []

                    test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32)
                    
                    for i in range(len(grid_models)):
                        resultado = grid_models[i](test_input)
                        resultado = resultado.detach().numpy()
                        posibilidades.append([round(resultado[0][0]), round(resultado[0][1])])
                    
                    #Calculo el porcentaje de veces que aparece cada posibilidad
                    probability_dict = {str(posibilidades.count(p)/len(posibilidades)*100) + "%": p for p in posibilidades}

                    probability_dict = {}
                    for p in posibilidades:
                        if not str(p) in probability_dict:
                            probability_dict[str(p)] = str(posibilidades.count(p)/len(posibilidades))

                    #Ordeno el diccionario por las probabilidades de mayor a menor
                    probability_dict = {k: v for k, v in sorted(probability_dict.items(), key=lambda item: item[1], reverse=True)}
                    
                    highest_probability = list(probability_dict.keys())[0]

                    #convertirla de nuevo a lista
                    highest_probability = highest_probability.replace("[", "").replace("]", "").split(", ")

                    highest_probability = [highest_probability[0], highest_probability[1]]

                    for p in probability_dict:
                        empty_grid[y][x] = empty_grid[y][x] + str(p) + ": "+ str(round(float(probability_dict[p])*100, 2))+"% "
                        break
                    #print("Probabilidad " + str(p) + ": " + str(posibilidades.count(p)/len(posibilidades)*100) + "%")


    #now I want to plot the grid, each cell containing their string

    binary_grid = np.where(np.array(empty_grid) == '#', 0, 1)
    
    if shape == "5x5":
        plt.figure(figsize=(8, 8))
    elif shape == "14x14":
        plt.figure(figsize=(16, 16))
    # Create a plot
    plt.imshow(binary_grid, cmap='gray', interpolation='nearest')
    # Set the locations of gridlines explicitly to have them at non
    # -even indices
    plt.xticks(np.arange(-0.5, len(binary_grid[0]), 1), [])
    plt.yticks(np.arange(-0.5, len(binary_grid), 1), [])

    plt.grid(True, color='black', linewidth=2, which='both', linestyle='-', alpha=0.5)

    # Draw diagonal lines in each cell to divide it into sectors
    for i in range(len(binary_grid)):
        for j in range(len(binary_grid[0])):

            if binary_grid[i, j] == 1:  # Check if the cell is white
                plt.plot([j - 0.5, j + 0.5], [i - 0.5, i + 0.5], color='black', linewidth=1)
                plt.plot([j + 0.5, j - 0.5], [i - 0.5, i + 0.5], color='black', linewidth=1)

                text = empty_grid[i][j].split("%")
                text1state = text[0].split(":")[0].replace(" ", "")
                text1prob = text[0].split(":")[1] + "%"
                text2state = text[1].split(":")[0].replace(" ", "")
                text2prob = text[1].split(":")[1] + "%"
                text3state = text[2].split(":")[0].replace(" ", "")
                text3prob = text[2].split(":")[1] + "%"
                text4state = text[3].split(":")[0].replace(" ", "")
                text4prob = text[3].split(":")[1] + "%"


                # Add text based on the sector
                if float(text[0].split(":")[1]) > higher_threshold:
                    color1 = 'green'
                elif float(text[0].split(":")[1]) < lower_threshold:
                    color1 = 'red'
                else:
                    color1 = 'black'
                if float(text[1].split(":")[1]) > higher_threshold:
                    color2 = 'green'
                elif float(text[1].split(":")[1]) < lower_threshold:
                    color2 = 'red'
                else:
                    color2 = 'black'
                if float(text[2].split(":")[1]) > higher_threshold:
                    color3 = 'green'
                elif float(text[2].split(":")[1]) < lower_threshold:
                    color3 = 'red'
                else:
                    color3 = 'black'
                if float(text[3].split(":")[1]) > higher_threshold:
                    color4 = 'green'
                elif float(text[3].split(":")[1]) < lower_threshold:
                    color4 = 'red'
                else:
                    color4 = 'black'

                plt.text(j, i - .4, text1state, ha='center', va='center', fontsize=6, color=color1)  # North
                plt.text(j, i - .25, text1prob, ha='center', va='center', fontsize=6, color=color1)  # North

                plt.text(j, i + .25, text2prob, ha='center', va='center', fontsize=6, color=color2)  # South
                plt.text(j, i + .4, text2state, ha='center', va='center', fontsize=6, color=color2)  # South

                plt.text(j - .3, i - .1, text3state, ha='center', va='center', fontsize=6, color=color3)  # West
                plt.text(j - .3, i + .05, text3prob, ha='center', va='center', fontsize=6, color=color3)  # West

                plt.text(j + .3, i - .1, text4state, ha='center', va='center', fontsize=6, color=color4)  # East
                plt.text(j + .3, i + .05, text4prob, ha='center', va='center', fontsize=6, color=color4)  # East


    #  Add row numbering on the left from top to bottom
    for i, label in enumerate(range(len(binary_grid))):
        plt.text(-0.8, i, str(label), ha='right', va='center', fontsize=12, color='black')

    # Add column numbering at the bottom
    for j, label in enumerate(range(len(binary_grid[0]))):
        plt.text(j, len(binary_grid) - 0.3, str(label), ha='center', va='top', fontsize=12, color='black')

    # Show the plot
    plt.show()