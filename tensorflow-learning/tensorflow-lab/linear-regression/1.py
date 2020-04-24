import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# import os
# print(os.getcwd())

cars = np.loadtxt("../../data/cars.csv", delimiter=",", dtype=np.int32)
# print(cars)
print(cars.shape)

x = cars[:, 0]
y = cars[:, 1]

x1, y1 = cars.T
print(cars.T)

print(x1)
print(y1)

X = tf.placeholder(dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

w = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

hx = w * X + b
cost = tf.reduce_mean(tf.square(hx - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={X:x})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X:x}))

# 위에서 학습이 끝났으므로 W와 b값이 도출됨.
# 자동차 속도가 30과 50일때 제동거리 예측가능
print(sess.run(hx, feed_dict={X:30}))
print(sess.run(hx, feed_dict={X:50}))

plt.title("cars")
plt.xlabel("speed")
plt.ylabel("distance")
plt.scatter(x, y)
plt.plot(x, sess.run(hx, feed_dict={X:x}), "r--")
plt.show()