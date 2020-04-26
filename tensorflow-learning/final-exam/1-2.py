'''
타이타닉
1. 생존자와 사망자에 대한 갯수를 구하시오
2. 등급별(pclass) 평균 생존률을 구하시오
( 등급과 생존율에 대한 pariplot을 그리시오 )
3. SibSp(가족과탑승) 의 평균 생존율을 구하시오
4. 혼자탑승(alone)한 인원의 평균 생존율을 구하시오
5. 성별 평균 생존율을 구하시오
6. 나이분류 컬럼을 추가하여 아래와 같이 출력하시오
1~15(미성년자), 15~25(청년), 25~35(중년),
35~60(장년), 60~(노년)  으로 표시하시요.
=================
  나이    나이분류
    20          청년
=================
train, test 구분... 정확, 측정...
7. 나이에 따른 생사를 예측하시오(텐서플로우, 케라스)

survived : 1 (생존), 0 (죽음)
sibsp : 같이 탑승인원, 0 ( 혼자탑승 )
parch : 직계가족
'''

import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.examples.tutorials.mnist import input_data

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(30, inplace=True)
# print(df)
# print(df.values)

'''
7. 나이에 따른 생사를 예측하시오(텐서플로우, 케라스)
'''

x_data = np.float32(df["Age"].values).reshape(-1, 1)
y_data = np.int32(df["Survived"].values)
print(x_data.shape)

e = np.eye(2)
y_one_hot = e[y_data]
print(y_one_hot.shape)

feature = 1
classes = 2

W = tf.Variable(tf.random_uniform([feature, classes]))
b = tf.Variable(tf.random_uniform([classes]))

X = tf.placeholder(dtype=tf.float32, shape=[None, feature])
Y = tf.placeholder(dtype=tf.float32, shape=[None, classes])

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
# cost-function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# softmax와 cast까지 같이 해주는 함수
# mean
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

predict = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predict, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_one_hot.argmax(axis=1)), dtype=tf.float32))

for i in range(1000):
    _t, _c, _a = sess.run([train, cost, accuracy], feed_dict={X: x_data, Y: y_one_hot})
    if not i % 100:
        print(i, _c, _a)

predict = sess.run(hypothesis, feed_dict={X: x_data})

# predict
print(predict.argmax(axis=1))

# accuracy
# print(np.mean(predict.argmax(axis=1) == y_one_hot.argmax(axis=1)))