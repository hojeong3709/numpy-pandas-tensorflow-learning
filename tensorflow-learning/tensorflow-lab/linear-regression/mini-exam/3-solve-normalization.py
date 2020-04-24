import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing

'''
1. trees.csv를 읽어들여서 아래에 대해
 Volume을 예측해 보세요.(텐서, 케라스)
Girth 8.8, 10.5
Height 63, 72
'''

data = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1, dtype=np.float32)
total_scale = preprocessing.MinMaxScaler()
dataN = total_scale.fit_transform(data)
x_data = dataN[:, :-1]
y_data = dataN[:, -1:]

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

# W = tf.Variable(tf.random_uniform([2, 1]))
# b = tf.Variable(tf.random_uniform([1]))

W = tf.get_variable(name="w1", shape=[2, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name="b1", shape=[1], initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# Y = tf.constant(y_data, dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

hx = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hx - Y))
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in np.arange(1000):
    _cost, _train = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if not i % 100:
        print(i, _cost)

print(sess.run(W))
print(sess.run(b))

# print(sess.run(hx, feed_dict={X: np.array([[8.8, 63], [10.5, 72]])}))

# 정규화
# 전체 스케일 크기가 N X 3 형태이므로 맨뒤에 None 값이 들어가도록
predict_data = total_scale.transform([[8.8, 63, None], [10.5, 72, None]])
predict_data = predict_data[:, :-1]
predict_result = sess.run(hx, feed_dict={X: predict_data})
print(predict_result)

# 역정규화
label_scale = preprocessing.MinMaxScaler()
yN = label_scale.fit_transform(data[:, -1:])
print(label_scale.inverse_transform(predict_result))





