import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('data/', one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# layer1
W1 = tf.get_variable('w1', [784, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1', [100],
                     initializer=tf.contrib.layers.xavier_initializer())

# Gradient Vanishing 문제를 해결하기 위해서
# cross-entropy --> relu : activation function을 relu로
z1 = tf.matmul(X, W1) + b1
hy1 = tf.nn.relu(z1)

# output layer
W2 = tf.get_variable('w2', [100, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2', [10],
                     initializer=tf.contrib.layers.xavier_initializer())
z2 = tf.matmul(hy1, W2) + b2
hy = tf.nn.softmax(z2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z2, labels=Y)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_train, Y: y_train})
    if not i % 100:
        print(i, _c)

import matplotlib.pyplot as plt

plt.imshow(x_test[0].reshape(28, 28))
plt.show()

# softmax로 값이 나오기 때문에 argmax로 제일 큰 확률값으로 예측값 뽑기
predict = sess.run(hy, feed_dict={X: x_test}).argmax(axis=1)

# label Data는 One Hot Vector 이므로 argmax로 제일 큰값에 대한 인덱스를 뽑기 ( 정답뽑기 )
accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(predict, y_test.argmax(axis=1)))), feed_dict={X: x_test, Y: y_test})
