import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from sklearn import preprocessing

# 값차이가 상당히 크기 때문에 정규화가 필수
# 정규화된 데이터
data = np.loadtxt("../../../data/diabetes1.csv", skiprows=1, delimiter=",", dtype=np.float32)
print(data)

x_data = data[:, :-1]
y_data = data[:, -1:]

print(x_data.shape)
print(y_data.shape)

W = tf.Variable(tf.random_uniform([8, 1]))
b = tf.Variable(tf.random_uniform([1]))

# W = tf.get_variable(name='w1', shape=[8, 1],
#                     initializer=tf.contrib.layers.xavier_initializer())
# b = tf.get_variable(name='b1', shape=[1],
#                     initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

cost = tf.reduce_mean(Y * (-tf.log(hx)) + (1-Y) * (-tf.log(1-hx)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
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

print(sess.run(W))
print(sess.run(b))

print(sess.run(predicted, feed_dict={X: x_data}))
print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))