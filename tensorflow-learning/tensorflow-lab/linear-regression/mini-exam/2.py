# electric.csv를 읽어서 w,b를 구하고
# 실측데이터 scatter, 예측데이터는 라인차트를 그리시요.
# 전기생산량이 5인경우 전기사용량을 예측하시오
# 전기생산량, 전기사용량

# Keras 버전으로
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

e = np.loadtxt("../../../data/electric.csv", delimiter=",", skiprows=1, dtype=np.float32, encoding='UTF8')

x = e[:, 1]
y = e[:, 2]

print(x)
print(y)
# Document
# https://keras.io/models/model/
model = Sequential(Dense(units=1, input_shape=[1]))
model.compile(loss="mse", optimizer=Adam(learning_rate=0.01))
history = model.fit(x, y, epochs=500)

print(history.history["loss"])
plt.plot(history.history["loss"])
plt.show()

print(model.predict([5]))