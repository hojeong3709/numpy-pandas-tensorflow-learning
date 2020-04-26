import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 예습, 복습 횟수
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 3]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.get_variable(name='w1', shape=[2, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name='b1', shape=[1],
                    initializer=tf.contrib.layers.xavier_initializer())

# he
# tf.contrib.layers.variance_scaling_initializer()


z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

cost_i = Y * (-tf.log(hx)) + (1 - Y) * (-tf.log(1 - hx))
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
    if not i % 100:
        print(i, _c)

print(sess.run(W))
print(sess.run(b))

predicted = tf.cast(hx > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

print(sess.run(predicted, feed_dict={X: x_data}))
print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
