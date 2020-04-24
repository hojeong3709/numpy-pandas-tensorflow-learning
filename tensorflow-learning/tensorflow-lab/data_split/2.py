import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn import model_selection

data = np.loadtxt("../../data/diabetes1.csv", skiprows=1, delimiter=",", dtype=np.float32)
# print(data)

x_data = data[:, :-1]
y_data = data[:, -1:]

x_train, y_train, x_test, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)
print("Train Data for learning: ", x_train.shape)
print("Train Data for Test: ", y_train.shape)
print("Label for learning ", x_test.shape)
print("Label for Test ", y_test.shape)

IO = Dense(units=1, input_shape=[8], activation="sigmoid")
model = Sequential([IO])
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

history = model.fit(x_train, x_test, epochs=100)

print(model.evaluate(y_train, y_test))