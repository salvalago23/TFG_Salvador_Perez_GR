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

class OfflineNN(nn.Module):
    def __init__(self, hidden_size):
        super(OfflineNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x