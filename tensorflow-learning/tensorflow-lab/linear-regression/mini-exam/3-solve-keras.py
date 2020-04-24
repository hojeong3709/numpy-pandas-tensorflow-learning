import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# https://keras.io/models/model/

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

'''
Early Stopping Module
'''
from tensorflow.keras.callbacks import EarlyStopping

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

trees = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1, dtype=np.float32)

x_data = trees[:, :-1]
y_data = trees[:, -1:]

IO = Dense(units=1, input_shape=[2])
model = Sequential(IO)
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01))
history = model.fit(x_data, y_data, epochs=1000)

print(model.predict(np.array([[8.8, 63], [10.5, 72]])))