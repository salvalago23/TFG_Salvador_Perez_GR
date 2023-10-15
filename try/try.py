import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

# Set the path to the CSV file
path_csv = '../data/csv/history.csv'

# Lists to store the values of the columns 's', 'reward', and other relevant columns
sy_values = []
sx_values = []
a_values = []
sy1_values = []
sx1_values = []
reward_values = []

# Read the data from the CSV file
with open(path_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        step, y, x, action, next_y, next_x, reward, done = row
        sy_values.append(float(y))
        sx_values.append(float(x))
        a_values.append(float(action))
        sy1_values.append(float(next_y))
        sx1_values.append(float(next_x))
        reward_values.append(float(reward))

# Convert the lists to NumPy arrays
sy_array = np.array(sy_values)
sx_array = np.array(sx_values)
a_array = np.array(a_values)
sy1_array = np.array(sy1_values)
sx1_array = np.array(sx1_values)
reward_array = np.array(reward_values)

input_data = np.column_stack((sy_array, sx_array, a_array))
target_data = np.column_stack((sy1_array, sx1_array))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# Define a PyTorch neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init()
        self.fc1 = nn.Linear(input_size, 48)
        self.fc2 = nn.Linear(48, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = NeuralNetwork(3, 2)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Training loop
num_epochs = 50
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

print("Model trained!")

# Evaluate the model on the test data
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test loss: {test_loss.item()}')

# Plot the training loss
plt.xlabel("# Epoch")
plt.ylabel("Loss Magnitude")
plt.plot(losses)

# Save the trained model
torch.save(model.state_dict(), '../data/models/modelo_entorno.pth')

# Perform predictions
for k in range(2):
    for i in range(4):
        test_input = torch.tensor([[4.0, float(k), float(i)]], dtype=torch.float32)
        result = model(test_input)
        result = result.numpy()
        print("Input vector [y, x, a]:", [4, k, i])
        print("Result:", result[0])