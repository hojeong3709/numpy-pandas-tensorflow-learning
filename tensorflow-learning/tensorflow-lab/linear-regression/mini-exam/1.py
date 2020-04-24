# electric.csv를 읽어서 w,b를 구하고
# 실측데이터 scatter, 예측데이터는 라인차트를 그리시요.
# 전기생산량이 5인경우 전기사용량을 예측하시오
# 전기생산량, 전기사용량

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

e = np.loadtxt("../../../data/electric.csv", delimiter=",", skiprows=1, dtype=np.float32, encoding='UTF8')

x = e[:, 1]
y = e[:, 2]

print(x)
print(y)

X = tf.placeholder(dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

w = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.random_uniform([1]))

hx = w * X + b
cost = tf.reduce_mean(tf.square(hx - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={X:x})
    if not i % 100:
        print(i, sess.run(cost, feed_dict={X:x}))


plt.title("electrics")
plt.xlabel("전기생산량")
plt.ylabel("전기사용량")
plt.scatter(x, y)
plt.plot(x, sess.run(hx, feed_dict={X: x}), "r--")
plt.show()

# 위에서 학습이 끝났으므로 W와 b값이 도출됨.
# 전기생산량 5일때 제동거리 예측가능
print(sess.run(hx, feed_dict={X: 5}))

