'''
아이리스(붓꽃) 데이터에 대한 데이터이다:
1. Sepal Length: 꽃받침의 길이 정보이다.
2. Sepal Width: 꽃받침의 너비 정보이다.
3. Petal Length: 꽃잎의 길이 정보이다.
4. Petal Width: 꽃잎의 너비 정보이다.
Species 꽃의 종류 정보이다.
setosa / versicolor / virginica 의 3종류로 구분된다.
'''

import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()

x_data = iris["data"]
y_data = iris["target"]

print(x_data.shape)
print(y_data.shape)

y_one_hot = np.eye(3)[y_data]
print(y_one_hot)

features = 4
classes = 3

W = tf.Variable(tf.random_uniform([features, classes]))
b = tf.Variable(tf.random_uniform([classes]))

X = tf.placeholder(dtype=tf.float32, shape=[None, features])
Y = tf.placeholder(dtype=tf.float32, shape=[None, classes])

z = tf.matmul(X, W) + b
hx = tf.nn.softmax(z)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx), axis=1))
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X: x_data})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X: x_data}))

print(sess.run(W))
print(sess.run(b))

predict = sess.run(z, feed_dict={X: [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.7, 3.8, 6.7, 2.2]]})

# 예측
print(np.unique(iris["target_names"]))
print(predict.argmax(axis=1))
for i in predict.argmax(axis=1):
    print(iris["target_names"][i])

# 정확도
hx = sess.run(hx, feed_dict={X: x_data})
print(np.equal(iris["target"], hx.argmax(axis=1)).mean())

accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(iris["target"], hx.argmax(axis=1)), dtype=tf.float32)))
print(accuracy)
