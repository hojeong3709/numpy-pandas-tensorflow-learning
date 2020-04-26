import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 특성데이터가 두개
x1 = [1, 0, 3, 0, 5]  # 공부한 시간
x2 = [0, 2, 0, 4, 0]  # 출석한 일수
y = [1, 2, 3, 4, 5]  # 점수

w1 = tf.Variable(tf.random_uniform([1]))
w2 = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

hx = w1 * x1 + w2 * x2 + b
cost = tf.reduce_mean(tf.square(hx - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    sess.run(train)
    print(i, sess.run(cost))

print(sess.run(w1))
print(sess.run(w2))
print(sess.run(b))
