import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('./datatwo/data1.csv')
print(df.shape)
df.describe()

# Extract features and target
X = df.iloc[:, :3].values
y = df.iloc[:, 3:].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define a simple neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = FeedForwardNN(input_size=3, hidden_size=32)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation
print('\nEVALUATION:\n')

# Predictions on training data
pred_train_tensor = model(X_train_tensor)
pred_train = pred_train_tensor.detach().numpy()
print('Train: ', np.sqrt(mean_squared_error(y_train, pred_train)))

# Single predictions
with torch.no_grad():
    print("Single predict:")
    inp = torch.FloatTensor([[1, 1, 0]])
    print(model(inp).numpy())

    print("Single predict:")
    inp = torch.FloatTensor([[2, 1, 1]])
    print(model(inp).numpy())
