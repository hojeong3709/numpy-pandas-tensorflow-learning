import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

data = np.loadtxt("../../data/cross-entropy.txt", dtype=np.float32)

x_data = data[:, 1:3]
y_data = data[:, 3:]

print(x_data.shape)
print(y_data.shape)

IO = Dense(units=3, input_shape=[2], activation="cross-entropy")
model = Sequential([IO])
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

history = model.fit(x_data, y_data, epochs=1000)

print(IO.get_weights())

p = model.predict( np.array([[3., 6.]]))
print(history.history['acc'][-1])
