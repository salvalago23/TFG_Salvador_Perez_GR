import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('./datatwo/data1.csv') 
print(df.shape)
df.describe()


train_x = df.drop(df.columns[[3, 4]], axis = 1)

train_y = df.drop(df.columns[[0, 1, 2]], axis = 1)


#Feed-Forward Neural Network Model
model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))  
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear')) 

model.compile(optimizer = 'adam', loss = 'mse', metrics=['mae','accuracy'])

history = model.fit(train_x, train_y, epochs=2000)

# Evaluation
print('\nEVALUATION:\n')

pred_train= model.predict(train_x)
print('Train: ', np.sqrt(mean_squared_error(train_y,pred_train)))

print("Single predict:")
inp = np.array([[1, 1, 0]])
print(model.predict(inp))

print("Single predict:")
inp = np.array([[2, 1, 1]])
print(model.predict(inp))