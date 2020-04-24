import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = np.loadtxt('../../data/data-01.csv', delimiter=",")
# print(data)

# 2차원 데이터로 뽑아오기
# matrix 연산을 하기 위해서는 2차원 텐서로 되어야 한다.
x_data = data[:, :-1]
# y_data = data[:, -1:]
y_data = data[:, [-1]]
print("x_data shape : ", x_data.shape)
print("y_data shape : ", y_data.shape)

w = tf.Variable(tf.random_uniform([3, 1]))
b = tf.Variable(tf.random_uniform([1]))
X = tf.placeholder(dtype=tf.float32, shape=(None, 3))
y = tf.constant(y_data, dtype=tf.float32)

hx = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(hx - y))
optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={X: x_data})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X: x_data}))

sess.run(w)

# 100, 98, 81 경우 점수 예측
print(sess.run(hx, feed_dict={X: [[100, 98, 81]]}))

# 73, 80, 75
# 93, 88, 93 경우 점수 예측
print(sess.run(hx, feed_dict={X:[[73, 80, 75], [93, 88, 93]]}))





