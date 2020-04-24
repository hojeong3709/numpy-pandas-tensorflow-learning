import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 문제1
# simple3 파일을 읽어
# hx = wx + b
# w, b 값을 구하고 x가 5인 경우의 값을 예측하시오.

f = np.loadtxt("simple3.txt", dtype=np.int32, skiprows=1, delimiter=",")
print(f)

x = f[:, 0]
y = f[:, 1]

print(x)
print(y)

X = tf.placeholder(dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

# xaiver or he 알고리즘으로 적절한 초기값을 지정할 수 있다.
W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

hx = W * X + b

cost = tf.reduce_mean(tf.square(hx - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    _train, _cost = sess.run([train, cost], feed_dict={X: x})
    print(i, _cost)
    plt.plot(i, _cost, "ro")

print("W : ", sess.run(W))
print("b : ", sess.run(b))

print("X가 5인경우 값 : ", sess.run(hx, feed_dict={X: 5}))

# 문제2
# 위의 결과로 그래프를 그리시오.
# x축, y축 (hx)
plt.show()

plt.plot(x, sess.run(hx, feed_dict={X: x}))
plt.show()

# 문제3
# y = (5x +2 ) ^ 2를 미분하시오
# 코딩이 아닌 도출과정을 적으시오.

# 풀이 ==> 2 * (5x + 2) * 5 (편미분)