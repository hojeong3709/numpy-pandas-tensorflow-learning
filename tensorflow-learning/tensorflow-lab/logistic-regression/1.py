import numpy as np
import tensorflow as tf

# 출석일수, 공부한시간
x_data = np.array([[1., 3.],
                   [2., 2.],
                   [3., 1.],
                   [4., 6.],
                   [5., 5.],
                   [6., 4.]])

y = np.array([[0], [0], [0], [1], [1], [1]])

print(x_data.shape)
print(y.shape)

X = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 1]))
b = tf.Variable(tf.random_uniform([1]))

z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

cost_i = y * (-tf.log(hx)) + (1 - y) * (-tf.log(1 - hx))
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_data})
    if not i % 100:
        print(i, _c)

print(sess.run(W))
print(sess.run(b))

print(sess.run(z, feed_dict={X: np.array([[3., 5.]])}))
print(sess.run(hx, feed_dict={X: np.array([[3., 5.], [1., 1.]])}))

result = sess.run(hx, feed_dict={X: np.array([[3., 5.], [1., 1.]])})

# True -> 1 False -> 0
result1 = tf.cast(result > 0.5, dtype=np.int32)
print(sess.run(result1))
