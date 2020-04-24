import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn import datasets
from sklearn import preprocessing


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

patience = 16
min_delta = 0.001
hist_loss = list()

# Early Stopping
# Cost 변화가 미미할때 조기종료
for i in np.arange(400):
    sess.run(train)
    c = sess.run(cost)
    print(i, c)
    hist_loss.append(c)
    if i > 0:
        pass
    else:
        pass




