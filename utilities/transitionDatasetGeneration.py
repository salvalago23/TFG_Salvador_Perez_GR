import numpy as np
import random
from collections import defaultdict

import os
import csv
import pandas as pd
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt

from envs.NeuralNetwork import OfflineNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_tensor(transitions):
    sy_values = np.array([t[0] for t in transitions], dtype=np.float32)
    sx_values = np.array([t[1] for t in transitions], dtype=np.float32)
    a_values = np.array([t[2] for t in transitions], dtype=np.float32)
    sy1_values = np.array([t[3] for t in transitions], dtype=np.float32)
    sx1_values = np.array([t[4] for t in transitions], dtype=np.float32)

    X = np.column_stack((sy_values, sx_values, a_values))
    y = np.column_stack((sy1_values, sx1_values))

    return torch.FloatTensor(X), torch.FloatTensor(y)

#Clase para simplificar la creación de los modelos a partir de las sublistas de transiciones
class ModelTrainer:
    def __init__(self, train_dataset, val_dataset, n_epochs, hidden_size, rand_seed = False):
        if rand_seed:
            self.seed = random.randint(1, 10000)
            torch.manual_seed(self.seed)

        self.X_train, self.y_train = convert_to_tensor(train_dataset)
        self.X_val, self.y_val = convert_to_tensor(val_dataset)

        self.n_epochs = n_epochs

        self.train_losses = []
        self.val_losses = []

        # Create DataLoader for training
        batch_size = 64  # Adjust the batch size based on your needs
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True)

        # Create DataLoader for validation
        self.val_loader = DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=batch_size, shuffle=False)

        self.model = OfflineNN(hidden_size).to(device)

    def fit(self):        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in tqdm(range(self.n_epochs)):
            # Forward pass
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_train = self.model(inputs)
                loss_train = criterion(outputs_train, targets)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                self.train_losses.append(loss_train.item())

            # Validation (every 1000 epochs)
            if epoch % 1000 == 0:
                self.val_losses.append(self.calculate_validation_loss())

        print("Trained successfully!")
        print(f'Final loss: {self.train_losses[-1]}\n')

    def show_loss(self):
        plt.xlabel("# Epoch")
        plt.ylabel("Loss Magnitude")
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(np.arange(0, self.n_epochs, self.val_steps), self.val_losses, 'ro-', label='Validation Loss')
        plt.legend()
        plt.show()

    def show_val_loss(self):
        plt.xlabel("# Epoch")
        plt.ylabel("Loss Magnitude")
        plt.plot(np.arange(0, self.n_epochs, self.val_steps), self.val_losses, 'ro-', label='Validation Loss')
        plt.legend()
        plt.show()

    def calculate_validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            outputs_val = self.model(self.X_val.to(device))
            loss_val = nn.MSELoss()(outputs_val, self.y_val.to(device)).item()
        self.model.train()
        return loss_val

def store_models(models_arr, folder_name):
    # Create the directory if it doesn't exist
    folder_path = f'../data/OfflineEnsembles/{folder_name}/'
    os.makedirs(folder_path, exist_ok=True)

    # Save each model in the given folder
    for i, model in enumerate(models_arr):
        model_path = os.path.join(folder_path, f'model_{i}.pt')
        torch.save(model.model.state_dict(), model_path)

    # Save train and val CSV files
    csv_train_file = os.path.join(folder_path, 'csv', 'train_loss.csv')
    csv_val_file = os.path.join(folder_path, 'csv', 'val_loss.csv')

    train_data = []
    val_data = []

    for i, model in enumerate(models_arr):
        train_data.append(model.train_losses)
        val_data.append(model.val_losses)

    column_titles = [f'model_{i}' for i in range(len(models_arr))]

    # Create the 'csv' directory if it doesn't exist
    csv_folder_path = os.path.join(folder_path, 'csv')
    os.makedirs(csv_folder_path, exist_ok=True)

    # Save train CSV
    with open(csv_train_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_titles)
        csv_writer.writerows(zip(*train_data))

    # Save val CSV
    with open(csv_val_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_titles)
        csv_writer.writerows(zip(*val_data))

def load_models(folder_name):
    folder_path = f'../data/OfflineEnsembles/{folder_name}/'

    if not os.path.exists(folder_path):
        print("ERROR: The folder " + folder_path + " doesn't exist")
        return
    
    # List all files in the folder
    model_files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]

    # Create an empty list to store loaded models
    models_arr = []
    print('Loading models from folder:', folder_path)

    # Iterate through each model file and load the model
    for model_file in model_files:
        print('Loading model:', model_file)

        # Create an instance of your model
        model = OfflineNN(hidden_size=512)
        
        # Load the weights
        model.load_state_dict(torch.load(os.path.join(folder_path, model_file)))
        
        # Move the model to the specified device
        model.to(device)
        model.eval()

        # Append the loaded model to the list
        models_arr.append(model)
    
    return models_arr

def plot_train_losses(folder_name):
    folder_path = f'../data/OfflineEnsembles/{folder_name}/'

    if not os.path.exists(folder_path):
        print("ERROR: The folder " + folder_path + " doesn't exist")
        return
    
    csv_file_name = os.path.join(folder_path, 'csv', 'train_loss.csv')

    # Read the data from the CSV file using pandas
    df = pd.read_csv(csv_file_name)

    # Plot each column with different colors
    colors = cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan',
                'magenta', 'teal', 'lime', 'gold', 'olive', 'navy', 'maroon', 'sienna', 'slateblue'])
    
    for i, column in enumerate(df.columns):
        color=next(colors)
        plt.plot(df[column], label=column, color=color)

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Data from CSV File')

    # Show the plot
    plt.show()

def plot_val_losses(folder_name):
    folder_path = f'../data/OfflineEnsembles/{folder_name}/'

    if not os.path.exists(folder_path):
        print("ERROR: The folder " + folder_path + " doesn't exist")
        return
    
    csv_file_name = os.path.join(folder_path, 'csv', 'val_loss.csv')

    # Read the data from the CSV file using pandas
    df = pd.read_csv(csv_file_name)

    # Plot each column with different colors
    colors = cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan',
                'magenta', 'teal', 'lime', 'gold', 'olive', 'navy', 'maroon', 'sienna', 'slateblue'])
    
    for i, column in enumerate(df.columns):
        color=next(colors)
        plt.plot(df[column], label=column, color=color)

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Data from CSV File')

    # Show the plot
    plt.show()

#FUNCIONES PARA CREAR SUBCONJUNTOS DE TRANSICIONES A PARTIR DE LA LISTA COMPLETA

#Esta funcion recibe un grid y devuelve todas las posibles transiciones que se pueden 
#hacer desde cada casilla del grid, para cada una de las 4 acciones posibles
def transition_generator(maze):
    train_transitions = []
    val_transitions = []

    n_rows, n_cols = len(maze), len(maze[0])

    for y in range(n_rows):
        for x in range(n_cols):
            for a in range(4):
                if a == 0:  # Mover hacia el norte
                    y1, x1 = max(0, y - 1), x
                elif a == 1:  # Mover hacia el sur
                    y1, x1 = min(n_rows - 1, y + 1), x
                elif a == 2:  # Mover hacia el oeste
                    y1, x1 = y, max(0, x - 1)
                elif a == 3:  # Mover hacia el este
                    y1, x1 = y, min(n_cols - 1, x + 1)

                if maze[y][x] != '#' and maze[y][x] != 'x':
                    if maze[y1][x1] == 'x':
                        val_transitions.append([y, x, a, y1, x1])
                    elif maze[y1][x1] == '#':
                        train_transitions.append([y, x, a, y, x])
                    else:
                        train_transitions.append([y, x, a, y1, x1])
                else:
                    if maze[y1][x1] == '#':
                        val_transitions.append([y, x, a, y, x])
                    else:
                        val_transitions.append([y, x, a, y1, x1])

    return train_transitions, val_transitions

# Esta función va seleccionando un subconjunto de transiciones de forma ordenada y ciclicamente (de la 0 a la N, de la N+1 a la 2N, ...).
# De esta manera se asegura que todas las transiciones se usan al menos una vez. Las transiciones que sobran se añaden al conjunto de validacion.
# Al final cada sublista se completa con repeticiones de las transiciones que ya tiene como antes.
def transitions_families_generator(train_transitions, val_transitions, n):
    n_transiciones_iniciales = len(train_transitions)
    n_subset = int(0.66 * n_transiciones_iniciales)

    new_train_transitions = []
    new_val_transitions = []

    for i in range(n):
        # Crear una copia de las transiciones iniciales
        new_train = train_transitions.copy()
        new_val = val_transitions.copy()

        start_index = (i * n_subset) % len(new_train)
        end_index = (start_index + n_subset) % len(new_train)

        if start_index < end_index:
            new_train = new_train[start_index:end_index]
            new_val.extend(new_train[end_index:])
        else:
            new_train = new_train[start_index:] + new_train[:end_index]
            new_val.extend(new_train[end_index:start_index])

        # Repetir aleatoriamente 'n_borradas' transiciones
        transiciones_a_repetir = random.sample(new_train, n_transiciones_iniciales - n_subset)
        new_train.extend(transiciones_a_repetir)

        new_train_transitions.append(new_train)
        new_val_transitions.append(new_val)

    return new_train_transitions, new_val_transitions

#FUNCION PARA MOSTRAR UNA GRÁFICA CON LAS PROBABILIDADES DE CADA ACCIÓN PARA CADA CASILLA DEL GRID
def probMapper(grid, models, threshold = 50.0):

    if len(grid) < 8:
        shape = "5x5"
    else:
        shape = "14x14"

    #print("Models loaded")

    empty_grid = [['' for i in range(len(grid[0]))] for j in range(len(grid))]

    n_rows, n_cols = len(grid), len(grid[0])

    with torch.no_grad():

        for y in range(n_rows):
            for x in range(n_cols):
                if grid[y][x] == "#":
                    empty_grid[y][x] = "#" 
                else:
                    for i in range(4):
                        posibilidades = []

                        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32).to(device)
                        
                        for i in range(len(models)):
                            resultado = models[i](test_input)
                            resultado = resultado.cpu().detach().numpy()
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

    binary_grid = np.where(np.array(empty_grid) == '#', 0, 1)

    if shape == "5x5":
        plt.figure(figsize=(8, 8))
        fontsize1 = 8
        fontsize2 = 10
    elif shape == "14x14":
        plt.figure(figsize=(16, 16))
        fontsize1 = 6
        fontsize2 = 8
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

                WrongTransitionsCount = 0
                
                for a in range(4):

                    if a == 0:  # Mover hacia el norte
                        if binary_grid[max(0, i - 1), j] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = max(0, i - 1), j

                        if y1 != int(text1state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text1state.replace("[", "").replace("]", "").split(",")[1]):
                            color1state = 'orange'
                            WrongTransitionsCount += 1
                        else:
                            color1state = 'green'
                    
                    elif a == 1:  # Mover hacia el sur
                        if binary_grid[min(n_rows - 1, i + 1), j] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = min(n_rows - 1, i + 1), j

                        if y1 != int(text2state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text2state.replace("[", "").replace("]", "").split(",")[1]):
                            color2state = 'orange'
                            WrongTransitionsCount += 1
                        else:
                            color2state = 'green'
                    
                    elif a == 2:  # Mover hacia el oeste
                        if binary_grid[i, max(0, j - 1)] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = i, max(0, j - 1)
                        
                        if y1 != int(text3state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text3state.replace("[", "").replace("]", "").split(",")[1]):
                            color3state = 'orange'
                            WrongTransitionsCount += 1
                        else:
                            color3state = 'green'
                    
                    elif a == 3:  # Mover hacia el este
                        if binary_grid[i, min(n_cols - 1, j + 1)] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = i, min(n_cols - 1, j + 1)
                        
                        if y1 != int(text4state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text4state.replace("[", "").replace("]", "").split(",")[1]):
                            color4state = 'orange'
                            WrongTransitionsCount += 1
                        else:
                            color4state = 'green'

                LowPercentageCount = 0
                # Add text based on the sector
                if float(text[0].split(":")[1]) >= threshold:
                    color1prob = 'green'
                else:
                    color1prob = 'red'
                    LowPercentageCount += 1
                if float(text[1].split(":")[1]) >= threshold:
                    color2prob = 'green'
                else:
                    color2prob = 'red'
                    LowPercentageCount += 1
                if float(text[2].split(":")[1]) >= threshold:
                    color3prob = 'green'
                else:
                    color3prob = 'red'
                    LowPercentageCount += 1
                if float(text[3].split(":")[1]) >= threshold:
                    color4prob = 'green'
                else:
                    color4prob = 'red'
                    LowPercentageCount += 1

                plt.text(j, i - .4, text1state, ha='center', va='center', fontsize=fontsize1, color=color1state)  # North
                plt.text(j, i - .25, text1prob, ha='center', va='center', fontsize=fontsize1, color=color1prob)  # North

                plt.text(j, i + .25, text2state, ha='center', va='center', fontsize=fontsize1, color=color2state)  # South
                plt.text(j, i + .4, text2prob, ha='center', va='center', fontsize=fontsize1, color=color2prob)  # South

                plt.text(j - .3, i - .1, text3state, ha='center', va='center', fontsize=fontsize1, color=color3state)  # West
                plt.text(j - .3, i + .05, text3prob, ha='center', va='center', fontsize=fontsize1, color=color3prob)  # West

                plt.text(j + .3, i - .1, text4state, ha='center', va='center', fontsize=fontsize1, color=color4state)  # East
                plt.text(j + .3, i + .05, text4prob, ha='center', va='center', fontsize=fontsize1, color=color4prob)  # East
            
            else:
                plt.text(j, i, '[' + str(i) + ',' + str(j) + ']', ha='center', va='center', fontsize=fontsize2, color='white')

    #  Add row numbering on the left from top to bottom
    for i, label in enumerate(range(len(binary_grid))):
        plt.text(-0.8, i, str(label), ha='right', va='center', fontsize=12, color='black')

    # Add column numbering at the bottom
    for j, label in enumerate(range(len(binary_grid[0]))):
        plt.text(j, len(binary_grid) - 0.3, str(label), ha='center', va='top', fontsize=12, color='black')

    plt.show()

#FUNCION PARA MOSTRAR UNA GRÁFICA CON LAS PROBABILIDADES DE CADA ACCIÓN PARA CADA CASILLA DEL GRID
def stdMapper(grid, models, threshold = 0.3):

    if len(grid) < 8:
        shape = "5x5"
    else:
        shape = "14x14"

    #print("Models loaded")

    empty_grid = [['' for i in range(len(grid[0]))] for j in range(len(grid))]
    #std grid should be an [n_rows][n_cols][4] array
    std_grid = [[[[] for i in range(4)] for j in range(len(grid[0]))] for k in range(len(grid))]
    std_grid2 = [[[[] for i in range(4)] for j in range(len(grid[0]))] for k in range(len(grid))]

    n_rows, n_cols = len(grid), len(grid[0])

    with torch.no_grad():

        for y in range(n_rows):
            for x in range(n_cols):
                if grid[y][x] == "#":
                    empty_grid[y][x] = "#" 
                else:
                    for i in range(4):
                        posibilidades = []

                        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32).to(device)
                        
                        for m in range(len(models)):
                            resultado = models[m](test_input)
                            resultado = resultado.cpu().detach().numpy()
                            posibilidades.append([round(resultado[0][0]), round(resultado[0][1])])
                            std_grid[y][x][i].append(resultado[0])

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
                            break #I only need the first one
                        #print("Probabilidad " + str(p) + ": " + str(posibilidades.count(p)/len(posibilidades)*100) + "%")
    
    for y in range(n_rows):
        for x in range(n_cols):
            for a in range(4):
                #std_grid2[y][x][a] = np.linalg.norm(np.std(std_grid[y][x][a], axis=0))
                if std_grid[y][x][a]:
                    std_grid2[y][x][a] = np.linalg.norm(np.std(std_grid[y][x][a], axis=0))
                else:
                    std_grid2[y][x][a] = 0

    binary_grid = np.where(np.array(empty_grid) == '#', 0, 1)

    if shape == "5x5":
        plt.figure(figsize=(8, 8))
        fontsize1 = 8
        fontsize2 = 10
    elif shape == "14x14":
        plt.figure(figsize=(16, 16))
        fontsize1 = 6
        fontsize2 = 8
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

                text1prob = round(np.linalg.norm(np.std(std_grid[i][j][0], axis=0)), 3)

                text2state = text[1].split(":")[0].replace(" ", "")

                text2prob = round(np.linalg.norm(np.std(std_grid[i][j][1], axis=0)), 3)

                text3state = text[2].split(":")[0].replace(" ", "")

                text3prob = round(np.linalg.norm(np.std(std_grid[i][j][2], axis=0)), 3)

                text4state = text[3].split(":")[0].replace(" ", "")

                text4prob = round(np.linalg.norm(np.std(std_grid[i][j][3], axis=0)), 3)

                for a in range(4):

                    if a == 0:  # Mover hacia el norte
                        if binary_grid[max(0, i - 1), j] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = max(0, i - 1), j

                        if y1 != int(text1state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text1state.replace("[", "").replace("]", "").split(",")[1]):
                            color1state = 'orange'
                        else:
                            color1state = 'green'
                    
                    elif a == 1:  # Mover hacia el sur
                        if binary_grid[min(n_rows - 1, i + 1), j] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = min(n_rows - 1, i + 1), j

                        if y1 != int(text2state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text2state.replace("[", "").replace("]", "").split(",")[1]):
                            color2state = 'orange'
                        else:
                            color2state = 'green'
                    
                    elif a == 2:  # Mover hacia el oeste
                        if binary_grid[i, max(0, j - 1)] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = i, max(0, j - 1)
                        
                        if y1 != int(text3state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text3state.replace("[", "").replace("]", "").split(",")[1]):
                            color3state = 'orange'
                        else:
                            color3state = 'green'
                    
                    elif a == 3:  # Mover hacia el este
                        if binary_grid[i, min(n_cols - 1, j + 1)] == 0:
                            y1, x1 = i, j
                        else:
                            y1, x1 = i, min(n_cols - 1, j + 1)
                        
                        if y1 != int(text4state.replace("[", "").replace("]", "").split(",")[0]) or x1 != int(text4state.replace("[", "").replace("]", "").split(",")[1]):
                            color4state = 'orange'
                        else:
                            color4state = 'green'

                # Add text based on the sector
                if np.linalg.norm(np.std(std_grid[i][j][0], axis=0)) <= threshold:
                    color1prob = 'green'
                else:
                    color1prob = 'red'

                if np.linalg.norm(np.std(std_grid[i][j][1], axis=0)) <= threshold:
                    color2prob = 'green'
                else:
                    color2prob = 'red'

                if np.linalg.norm(np.std(std_grid[i][j][2], axis=0)) <= threshold:
                    color3prob = 'green'
                else:
                    color3prob = 'red'

                if np.linalg.norm(np.std(std_grid[i][j][3], axis=0)) <= threshold:
                    color4prob = 'green'
                else:
                    color4prob = 'red'

                plt.text(j, i - .4, text1state, ha='center', va='center', fontsize=fontsize1, color=color1state)  # North
                plt.text(j, i - .25, text1prob, ha='center', va='center', fontsize=fontsize1, color=color1prob)  # North

                plt.text(j, i + .25, text2state, ha='center', va='center', fontsize=fontsize1, color=color2state)  # South
                plt.text(j, i + .4, text2prob, ha='center', va='center', fontsize=fontsize1, color=color2prob)  # South

                plt.text(j - .3, i - .1, text3state, ha='center', va='center', fontsize=fontsize1, color=color3state)  # West
                plt.text(j - .3, i + .05, text3prob, ha='center', va='center', fontsize=fontsize1, color=color3prob)  # West

                plt.text(j + .3, i - .1, text4state, ha='center', va='center', fontsize=fontsize1, color=color4state)  # East
                plt.text(j + .3, i + .05, text4prob, ha='center', va='center', fontsize=fontsize1, color=color4prob)  # East
            
            else:
                plt.text(j, i, '[' + str(i) + ',' + str(j) + ']', ha='center', va='center', fontsize=fontsize2, color='white')

    #  Add row numbering on the left from top to bottom
    for i, label in enumerate(range(len(binary_grid))):
        plt.text(-0.8, i, str(label), ha='right', va='center', fontsize=12, color='black')

    # Add column numbering at the bottom
    for j, label in enumerate(range(len(binary_grid[0]))):
        plt.text(j, len(binary_grid) - 0.3, str(label), ha='center', va='top', fontsize=12, color='black')

    plt.show()

#FUNCION PARA MOSTRAR UNA GRÁFICA CON LAS PROBABILIDADES DE CADA ACCIÓN PARA CADA CASILLA DEL GRID
def stdMeanMapper(grid, models, threshold = 0.3):

    if len(grid) < 8:
        shape = "5x5"
    else:
        shape = "14x14"

    #print("Models loaded")

    empty_grid = [['' for i in range(len(grid[0]))] for j in range(len(grid))]
    #std grid should be an [n_rows][n_cols][4] array
    std_grid = [[[[] for i in range(4)] for j in range(len(grid[0]))] for k in range(len(grid))]
    std_grid2 = [[[[] for i in range(4)] for j in range(len(grid[0]))] for k in range(len(grid))]

    n_rows, n_cols = len(grid), len(grid[0])

    with torch.no_grad():

        for y in range(n_rows):
            for x in range(n_cols):
                if grid[y][x] == "#":
                    empty_grid[y][x] = "#" 
                else:
                    for i in range(4):
                        posibilidades = []

                        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32).to(device)
                        
                        for m in range(len(models)):
                            resultado = models[m](test_input)
                            resultado = resultado.cpu().detach().numpy()
                            posibilidades.append([round(resultado[0][0]), round(resultado[0][1])])
                            std_grid[y][x][i].append(resultado[0])

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
                            break #I only need the first one
                        #print("Probabilidad " + str(p) + ": " + str(posibilidades.count(p)/len(posibilidades)*100) + "%")
    
    for y in range(n_rows):
        for x in range(n_cols):
            for a in range(4):
                if std_grid[y][x][a]:
                    std_grid2[y][x][a] = np.linalg.norm(np.std(std_grid[y][x][a], axis=0))
                else:
                    std_grid2[y][x][a] = 0

    binary_grid = np.where(np.array(empty_grid) == '#', 0, 1)

    if shape == "5x5":
        plt.figure(figsize=(8, 8))
        fontsize1 = 8
        fontsize2 = 10
    elif shape == "14x14":
        plt.figure(figsize=(16, 16))
        fontsize1 = 6
        fontsize2 = 8
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
                text1prob = round(np.linalg.norm(np.std(std_grid[i][j][0], axis=0)), 3)
                text2prob = round(np.linalg.norm(np.std(std_grid[i][j][1], axis=0)), 3)
                text3prob = round(np.linalg.norm(np.std(std_grid[i][j][2], axis=0)), 3)
                text4prob = round(np.linalg.norm(np.std(std_grid[i][j][3], axis=0)), 3)

                textprob = round(np.mean([text1prob, text2prob, text3prob, text4prob]), 3)

                if textprob <= threshold:
                    colortextprob = "green"
                else:
                    colortextprob = "red"

                plt.text(j, i, textprob, ha='center', va='center', fontsize=fontsize1, color=colortextprob)

            
            else:
                plt.text(j, i, '[' + str(i) + ',' + str(j) + ']', ha='center', va='center', fontsize=fontsize2, color='white')

    #  Add row numbering on the left from top to bottom
    for i, label in enumerate(range(len(binary_grid))):
        plt.text(-0.8, i, str(label), ha='right', va='center', fontsize=12, color='black')

    # Add column numbering at the bottom
    for j, label in enumerate(range(len(binary_grid[0]))):
        plt.text(j, len(binary_grid) - 0.3, str(label), ha='center', va='top', fontsize=12, color='black')

    plt.show()


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