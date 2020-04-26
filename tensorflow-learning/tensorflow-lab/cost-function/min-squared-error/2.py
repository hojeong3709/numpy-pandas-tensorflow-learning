import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]
w = -1

X = tf.constant(x)
Y = tf.constant(y)

hypothesis = w * X
sq = tf.square(hypothesis - Y)
cost = tf.reduce_mean(sq)

sess = tf.Session()
print(sess.run(sq))
print(sess.run(cost))
