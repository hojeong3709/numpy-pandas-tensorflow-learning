import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn import datasets

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

iris = datasets.load_iris()
data = iris["data"]

x_data = data[:, 1:]
y_data = data[:, :1]

IO = Dense(units=1, input_shape=[3])
model = Sequential([IO])
# https://keras.io/models/model/
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01))
history = model.fit(x_data, y_data, epochs=1000)

print(history.history["loss"])
print(IO.get_weights())

# predict 할때 ndarray가 입력되도록
result = model.predict(np.array([[3.5, 1.4, 0.2]]))
print(result)

predict_y_data = model.predict(x_data)

plt.plot(y_data, 'b-')
plt.plot(predict_y_data, 'r-')
plt.show()
