import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]

# X = tf.constant(x, dtype=tf.float32)
X = tf.placeholder(dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

W = tf.Variable(10., dtype=tf.float32)
b = tf.Variable(10., dtype=tf.float32)

hx = W * X + b

cost = tf.reduce_mean(tf.square(hx - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in np.arange(400):
    sess.run(train, feed_dict={X:x})
    print(i, sess.run(cost, feed_dict={X:x}))

print(sess.run(W))

# x가 5일때의 값을 예측하시오
print(sess.run(hx, feed_dict={X:5}))