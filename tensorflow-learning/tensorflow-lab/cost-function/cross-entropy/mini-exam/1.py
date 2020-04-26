'''
1. sonar data set이용하여 광물분류를 예측하시요
(R, M)
0) 마지막광물분류를 제외한 나머지 컬럼은 초음파 측정치 임
1)eary stopping적용할것)
2)train 과 train_test_data_split set 을 이용할것(acc 구할것)
tensor 버전, keras 버전
2개 버전을 작성할것
'''

import numpy as np
import tensorflow as tf
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')

data = np.loadtxt("../../../data/sonar.csv", dtype=np.str, delimiter=",")

y_data = data[:, -1:]
y_data_list = list()
for n in y_data:
    if n == 'R':
        y_data_list.append(0)
    else:
        y_data_list.append(1)

# m = map(lambda n: 0 if n == "R" else 1, data[:, -1])

x_data = np.float32(data[:, :-1])
y_data = np.array(y_data_list, dtype=np.float32).reshape(len(y_data_list), 1)
# print(y_data)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)
# print(x_train)
# print(x_test)

print(y_train.shape)
print(y_test.shape)


# W = tf.Variable(tf.random_uniform([60, 1]))
# b = tf.Variable(tf.random_uniform([1]))

W = tf.get_variable("weights", [60, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable("bias", [1], initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(dtype=tf.float32, shape=[None, 60])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

z = tf.matmul(X, W) + b
hx = tf.sigmoid(z)

cost = tf.reduce_mean(Y * (-tf.log(hx)) + (1-Y) * (-tf.log(1-hx)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

history = list()
min_delta = 0.001
patience = 100
cnt = 0

for i in range(10000):
    _t, _c = sess.run([train, cost], feed_dict={X: x_train, Y: y_train})
    print(i, _c)

    history.append(_c)
    if history[-1] - history[i] > min_delta:
        cnt = 0
    else:
        cnt = cnt + 1

    if cnt > patience:
        print("Early Stop")
        break

# print(sess.run(W))
# print(sess.run(b))

predict = sess.run(hx, feed_dict={X: x_test, Y: y_test})
predict_result = predict > 0.5
# print(predict_result)

accuracy = (predict_result == y_test).mean()
print(accuracy)
accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(predict_result, y_test), dtype=tf.float32)), feed_dict={X: x_test, Y: y_test})
print(accuracy)






