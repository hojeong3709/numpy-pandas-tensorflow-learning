'''
Tensorflow - Neural Network
'''

import tensorflow as tf
import numpy as np

# XOR 문제
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# layer1
W1 = tf.Variable(tf.random_normal([2, 2]))  # wide
b1 = tf.Variable(tf.random_normal([2]))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)  # input : 4 X 2 weight : 2 X 2 ==> 4 X 2

# output layer
W2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
z = tf.matmul(layer1, W2) + b2  # input : 4 X 2 weight : 2 X 1 ==> 4 X 1
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

for i in range(10000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
    if not i % 100:
        print(i, _c)

print(sess.run(predicted, feed_dict={X: x_data, Y: y_data}))
print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
