import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn import datasets
from sklearn import preprocessing

'''
정규화
1. min, max normalization
2. stadardizaion normalization
'''

data = [[828, 920, 1234567, 1020, 1111],
        [824, 910, 2345612, 1090, 1234],
        [880, 900, 3456123, 1010, 1000],
        [870, 990, 2312123, 1001, 1122],
        [860, 980, 3223123, 1008, 1133],
        [850, 970, 2432123, 1100, 1221]]
data = np.float32(data)

scale = preprocessing.MinMaxScaler()
data = scale.fit_transform(data)

x_data = data[:, :-1]
y_data = data[:, -1:]

print(x_data)
print(y_data)

w = tf.Variable(tf.random_uniform([4, 1]))
b = tf.Variable(tf.random_uniform([1]))

X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.constant(y_data, dtype=tf.float32)

hx = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(hx - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X: x_data})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X: x_data}))

print(sess.run(w))
print(sess.run(b))

# 예측값
print(sess.run(hx, feed_dict={X: x_data}))

# 실제값
print(y_data)

# 역정규화
data = [[828, 920, 1234567, 1020, 1111],
        [824, 910, 2345612, 1090, 1234],
        [880, 900, 3456123, 1010, 1000],
        [870, 990, 2312123, 1001, 1122],
        [860, 980, 3223123, 1008, 1133],
        [850, 970, 2432123, 1100, 1221]]

data = np.float32(data)
y1 = data[:, -1:]
ny = preprocessing.MinMaxScaler()
y1 = ny.fit_transform(y1)

xx = scale.transform(([[828, 920, 1234567, 1020, None]]))
xx = xx[:, :-1]
print(xx)

yy = sess.run(hx, feed_dict={X: xx})
print(ny.inverse_transform(yy))
