import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(10, dtype=tf.float32)
X = tf.constant(x, dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

hx = W * X
cost = tf.reduce_mean(tf.square(hx - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in np.arange(100):
    sess.run(train)
    print(i, sess.run(cost))

print(sess.run(W))
