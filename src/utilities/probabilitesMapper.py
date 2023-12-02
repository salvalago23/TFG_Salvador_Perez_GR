import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc1_unit=None, fc2_unit=None):
        super(NeuralNetwork, self).__init__()
        if fc1_unit is None:
            self.big = False
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, output_size)
        else:
            self.big = True
            self.fc1 = nn.Linear(input_size, fc1_unit)
            self.fc2 = nn.Linear(fc1_unit, fc2_unit)
            self.fc3 = nn.Linear(fc2_unit, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        if not self.big:
            x = self.fc2(x)
        else:
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        return x


def probMapper(grid, shape, n_models):

    print("Loading models...")
    grid_models = []

    for i in range(n_models):
        if shape == "5x5":
            model = NeuralNetwork(3, 2)
        elif shape == "14x14":
            model = NeuralNetwork(3, 2, 128, 64)

        model.load_state_dict(torch.load("../data/offline_models/{}_{}.pt".format(shape, i)))
        model.eval()

        grid_models.append(model)
    
    print("Models loaded")

    y, x = 2, 3

    for i in range(4):
        posibilidades = []

        test_input = torch.tensor([[float(y), float(x), float(i)]], dtype=torch.float32)
        
        print("Input: " + str([y, x, i]))

        for i in range(len(families_arr)):
            resultado = families_arr[i].model(test_input)
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