import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn import datasets
from sklearn import preprocessing

'''
1. trees.csv를 읽어들여서 아래에 대해
 Volume을 예측해 보세요.(텐서, 케라스)
Girth 8.8, 10.5
Height 63, 72
'''
trees = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1, dtype=np.float32)
# 정규화
total_scale = preprocessing.MinMaxScaler()
trees = total_scale.fit_transform(trees)

# 스케일을 하게 되면 0값이 생성되게 되는데 이게 학습에 문제가 되지 않을까?

x_data = trees[:, :-1]
y_data = trees[:, -1:]
print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

W = tf.Variable(tf.random_uniform([2, 1]))
b = tf.Variable(tf.random_uniform([1]))

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.constant(y_data, dtype=tf.float32)

hx = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hx - Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in np.arange(1000):
    _cost, _train = sess.run([cost, train], feed_dict={X: x_data})
    if not i % 100:
        print(i, _cost)

# predict
# Girth 8.8, 10.5
# Height 63, 72
# --> [[8.8, 63], [10.5, 72]]

print("학습된 Weight : ", sess.run(W))
print("학습된 bias : ", sess.run(b))

# 입력데이터에 대한 스케일 조정
predict_data = np.float32(np.array([[8.8, 63], [10.5, 72]]))
# 새로운 스케일러가 필요한것일까? 기존에 있던것에 적용해서 변환하는 것이 맞겠지?
# predict_data_scale = preprocessing.MinMaxScaler()
# predict_data_scale.fit_transform(predict_data)
# scaled_predict_data = predict_data_scale.transform(predict_data)
scaled_predict_data = total_scale.fit_transform(predict_data)
predict_result = sess.run(hx, feed_dict={X: scaled_predict_data})
print("(Tensorflow) 스케일된 예측값 : ", predict_result)

# 원본데이터 라벨에 대해 스케일 조정 학습
trees = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1)
y_data = np.float32(trees[:, -1:])
label_data_scale = preprocessing.MinMaxScaler()
scaled_label_data = label_data_scale.fit_transform(y_data)

# 복원
inverse_predict_result = label_data_scale.inverse_transform(predict_result)
print("(Tensorflow) 복원된 예측값 : ", inverse_predict_result)


model = Sequential(Dense(units=1, input_shape=[2]))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01))
history = model.fit(x_data, y_data, epochs=1000)
# print(history.history["loss"])
predict_result = model.predict(scaled_predict_data)
print("(Keras) 스케일된 예측값 : ", predict_result)
inverse_predict_result = label_data_scale.inverse_transform(predict_result)
print("(Keras) 복원된 예측값 : ", inverse_predict_result)

'''
2. volume이 40 이상이면 크다
30이상이면 보통 미만이면 적음으로
아래와 같이 출력하시요
volume  정도
============
 10.3    적음
...
'''

trees = np.loadtxt("../../../data/trees.csv", delimiter=",", skiprows=1, dtype=np.float32)

volume = trees[:, -1]
big_index = np.where(volume >= 40)
medium_index = np.where((30 <= volume) & (volume < 40))
small_index = np.where(volume < 30)

'''
3. Height  가 가장 작은값과 큰값을
구하시요

4. girth(테두리) 가 가장큰 top5를
출력하세요(girth, height, volume)
'''



