import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import warnings

warnings.filterwarnings('ignore')

'''
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율
MEDV : 주택가격

문제
다음 데이터가 주어졌을 때 주택가격 예측하기
predict_data = [[0.02729 0 7.07 0 0.469 7.185 61.1 4.9671 2 242 17.8 392.83 4.03]]
'''
data = np.loadtxt("../../../data/BostonHousing.csv", skiprows=1, delimiter=",", dtype=np.float32)

x_data = data[:, :-1]
y_data = data[:, -1:]

print(x_data.shape)
print(y_data.shape)

layer1 = Dense(units=20, input_shape=[13])
layer2 = Dense(units=20, input_shape=[20])
output_layer = Dense(units=1, input_shape=[20])

model = Sequential()
model.add(layer1)
model.add(layer2)
model.add(output_layer)

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_data, y_data, epochs=2000)

print(model.predict(x_data))
print(model.evaluate(x_data, y_data))

predict_data = np.array([[0.02729, 0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242, 17.8, 392.83, 4.03]])
print(model.predict(predict_data))
