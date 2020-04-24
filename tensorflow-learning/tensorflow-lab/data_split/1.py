import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')


data = np.loadtxt("../../data/diabetes1.csv", skiprows=1, delimiter=",", dtype=np.float32)
# print(data)

x_data = data[:, :-1]
y_data = data[:, -1:]

split_data = model_selection.train_test_split(x_data, y_data, test_size=0.3)
print(np.array(split_data).shape)

x_train, y_train, x_test, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)
print(data.shape)
print("Train Data for learning: ", x_train.shape)
print("Train Data for Test: ", y_train.shape)
print("Label for learning ", x_test.shape)
print("Label for Test ", y_test.shape)

X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_uniform([8, 1]))
b = tf.Variable(tf.random_uniform([1]))

z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

cost = tf.reduce_mean(Y * (-tf.log(hx)) + (1-Y)*(-tf.log(1-hx)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_train, Y: x_test})
    if not i % 100:
        print(i, _c)


h = sess.run(hx, feed_dict={X: y_train, Y: y_test})
predicted = h > 0.5
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_test), dtype=tf.float32))
print(sess.run(accuracy))

