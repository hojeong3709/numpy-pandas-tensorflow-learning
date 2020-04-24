import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(30, inplace=True)

# print(df)
# print(df.values)

'''
7. 나이에 따른 생사를 예측하시오(텐서플로우, 케라스)
'''

x_data = np.float32(df["Age"].values).reshape(-1, 1)
y_data = np.float32(df["Survived"].values).reshape(-1, 1)

layer = Dense(units=1, input_shape=[1], activation="sigmoid")

model = Sequential()
model.add(layer)
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

history = model.fit(x_data, y_data, epochs=1000)
print(model.evaluate(x_data, y_data))