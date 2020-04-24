'''
Keras - Neural Network
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import warnings

warnings.filterwarnings('ignore')
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 중간층 input shape 값은 생략가능
# layer1 = Dense(10, input_dim=2, activation='sigmoid')
layer1 = Dense(units=10, input_shape=[2], activation='sigmoid')
layer2 = Dense(units=10, input_shape=[10], activation='sigmoid')
output_layer = Dense(units=1, input_shape=[10], activation="sigmoid")

model = Sequential()
model.add(layer1)
model.add(layer2)
model.add(output_layer)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_data, y_data, epochs=2000)

print(model.predict_classes(x_data))
print(model.evaluate(x_data, y_data))
