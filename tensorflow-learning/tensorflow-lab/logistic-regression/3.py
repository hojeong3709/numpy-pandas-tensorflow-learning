import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

# 값차이가 상당히 크기 때문에 정규화가 필수
# 정규화된 데이터
data = np.loadtxt("../../data/diabetes1.csv", skiprows=1, delimiter=",", dtype=np.float32)
print(data)

x_data = data[:, :-1]
y_data = data[:, -1:]

print(x_data.shape)
print(y_data.shape)

IO = Dense(units=1, input_shape=[8], activation="sigmoid")
model = Sequential([IO])
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

history = model.fit(x_data, y_data, epochs=100)

print(model.predict(x_data))
print(model.predict_classes(x_data))
print(history.history['acc'][-1])

