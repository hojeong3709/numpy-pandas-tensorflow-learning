import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

x = [[1., 0], [0, 2.], [3., 0], [0, 4.], [5., 0]]
y = [1, 2, 3, 4, 5]

w = tf.Variable(tf.random_uniform([2, 1]))
b = tf.Variable(tf.random_uniform([1]))

hx = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hx - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    sess.run(train)
    print(i, sess.run(cost))

print(sess.run(w))
print(sess.run(b))

# 5시간 공부 3시간 출석시 점수를 예측하시오
print(sess.run(tf.matmul([[5., 3.]], w) + b))
