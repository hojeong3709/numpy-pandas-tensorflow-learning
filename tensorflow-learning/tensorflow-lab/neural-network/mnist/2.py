import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

mnist = input_data.read_data_sets('data/', one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

model = Sequential()
model.add(Dense(units=64, input_dim=784, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10)

print(model.evaluate(x_test, y_test))

import matplotlib.pyplot as plt

plt.imshow(x_test[0].reshape(28, 28))
plt.show()

# x_test[0] --> 1차원 벡터
# matmix mulptiply 해야 하므로 2차원 매트릭스로 변환 필요
# predict = model.predict(x_test[0].reshape(1, 784)).argmax()
predict = model.predict(x_test[0].reshape(1, -1)).argmax()