'''
Tensorflow - Neural Network
'''

import tensorflow as tf
import numpy as np

# XOR 문제
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

# logistic..
cost_i = Y * -tf.log(hx) + (1 - Y) * -tf.log(1 - hx)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hx > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
    print(i, _c)

print(sess.run(predicted, feed_dict={X: x_data, Y: y_data}))
print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
