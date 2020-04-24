import tensorflow as tf
import numpy as np

data = np.loadtxt("../../data/softmax.txt", dtype=np.float32)

x = data[:, 1:3]
y = data[:, 3:]

# [입력갯수, 분류갯수]
X = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3]))
b = tf.Variable(tf.random_uniform([3]))

z = tf.matmul(X, W) + b
hy = tf.nn.softmax(z)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X: x})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X: x}))

print(sess.run(W))
print(sess.run(b))

h = sess.run(hy, feed_dict={X: [[3., 6.]]})
print(h)

index = h.argmax()
print(index)

grades = ["A", "B", "C"]
print(grades[index])