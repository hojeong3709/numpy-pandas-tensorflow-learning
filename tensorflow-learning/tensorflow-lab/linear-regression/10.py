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

C = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=np.float32)
F = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=np.float32)

# label 갯수 --> units
# feature 갯수 --> input_shape
D = Dense(units=1, input_shape=[1])
model = Sequential(D)
model.compile(loss="mse", optimizer=Adam(learning_rate=0.1))

early_stop = EarlyStopping(monitor="loss", patience=50, min_delta=0.1)

history = model.fit(C, F, epochs=1000, callbacks=[early_stop])

print(history.history["loss"])
