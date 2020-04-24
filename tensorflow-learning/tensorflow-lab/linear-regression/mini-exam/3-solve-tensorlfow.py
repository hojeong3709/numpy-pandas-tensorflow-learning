import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing

'''
1. trees.csv를 읽어들여서 아래에 대해
 Volume을 예측해 보세요.(텐서, 케라스)
Girth 8.8, 10.5
Height 63, 72
'''
trees = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1, dtype=np.float32)

x_data = trees[:, :-1]
y_data = trees[:, -1:]
print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)
import matplotlib.pyplot as plt

# W = tf.Variable(tf.random_uniform([2, 1]))
# b = tf.Variable(tf.random_uniform([1]))

W = tf.get_variable(name="w1", shape=[2, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name="b1", shape=[1], initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# Y = tf.constant(y_data, dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
hx = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hx - Y))
optimizer = tf.train.AdamOptimizer(0.1)

# 1. learning rate 조절
# 2. Optimizer 변경
# 3. Normalization
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in np.arange(1000):
    _cost, _train = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if not i % 100:
        print(i, _cost)


print(sess.run(W))
print(sess.run(b))

print(sess.run(hx, feed_dict={X: np.array([[8.8, 63], [10.5, 72]])}))
