import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

# Nombre del archivo CSV
path_csv = '../data/csv/history.csv'

# Listas para almacenar los valores de las columnas 's' y 'reward'
sy_values = []
sx_values = []
a_values = []
sy1_values = []
sx1_values = []
reward_values = []

# Leer el archivo CSV
with open(path_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Quitar la primera fila donde están los nombres de las columnas

    for row in csv_reader:
        step, y, x, action, next_y, next_x, reward, done =  row
        sy_values.append(float(y))
        sx_values.append(float(x))
        a_values.append(float(action))
        sy1_values.append(float(next_y))
        sx1_values.append(float(next_x))
        reward_values.append(float(reward))

# Convertir las listas en NumPy arrays
sy_array = np.array(sy_values)
sx_array = np.array(sx_values)
a_array = np.array(a_values)
sy1_array = np.array(sy1_values)
sx1_array = np.array(sx1_values)
reward_array = np.array(reward_values)

input_data2 = np.column_stack((sy_array, sx_array, a_array))
target_data2 = reward_array
X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data2, target_data2, test_size=0.2, random_state=42)


oculta1 = keras.layers.Dense(units=48, input_shape=(3,), activation='relu')
salida2 = keras.layers.Dense(units=1)

modelo2 = keras.Sequential([oculta1, salida2])

modelo2.compile(optimizer='adam', loss='mean_squared_error')

historial2 = modelo2.fit(X_train2, y_train2, epochs=50, batch_size=32, verbose=False, validation_data=(X_test1, y_test1))
print("Modelo entrenado!")

test_loss1 = modelo2.evaluate(X_test2, y_test2)
print(f'Test loss: {test_loss1}')

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial2.history["loss"])

modelo2.save('../data/models/_modelo_entorno.h5')

for k in range(2):
    for i in range(4):
        #print("Hagamos una predicción!")
        test_input = np.array([[4,k,i]])
        resultado = modelo2.predict(test_input)
        print("El vector de entrada [y,x,a] es ",[4,k,i])
        print("El resultado es " + str(resultado))